"""
v04: Backbone Exploration
==========================
Compare three backbones using the same ArcFace setup as v03.
Goal: find which architecture captures jaguar spot patterns best.

Backbones tested:
  efficientnetv2_s      — Fused-MBConv, ~22M params, trains faster than B0
  convnext_small        — 7x7 depthwise conv, large texture receptive field
  vit_small_patch16_224 — global self-attention, holistic pattern matching

Performance optimisations vs v03:
  1. Image pre-cache: all 1895 images loaded to RAM once (PNG decode happens
     once, not once per epoch per worker). Eliminates the disk I/O bottleneck
     that made v02/v03 take 8-10h for ~1895 small images.
  2. Mixed precision (AMP): torch.cuda.amp halves memory/compute on T4/P100.
  Together these should cut wall-clock time to ~1.5-2h per backbone.

All backbones:
  ArcFace(m=0.5, s=30), embedding_dim=512, img_size=224
  Stage 1: 5 epochs frozen | Stage 2: 25 epochs full fine-tune
  Best backbone selected by val mAP, retrained on full data for submission.
"""

import sys, os, time

REPO_URL = 'https://github.com/Smooth-Cactus0/jaguar-re-identification.git'
REPO_DIR = '/kaggle/working/repo'

if os.path.exists(REPO_DIR):
    os.system(f'git -C {REPO_DIR} pull --ff-only')
else:
    os.system(f'git clone {REPO_URL} {REPO_DIR}')
sys.path.insert(0, REPO_DIR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.data      import (JaguarDataset, get_transforms, encode_labels,
                            get_identity_splits, build_image_cache)
from src.models    import EmbeddingModel
from src.losses    import ArcFaceLoss
from src.inference import extract_embeddings, make_submission
from src.evaluate  import compute_map, print_results, save_benchmark

# -- Paths ---------------------------------------------------------------------
KAGGLE_INPUT = Path('/kaggle/input/jaguar-re-id')
TRAIN_DIR    = KAGGLE_INPUT / 'train' / 'train'
TEST_DIR     = KAGGLE_INPUT / 'test'  / 'test'
OUT_DIR      = Path('/kaggle/working/output')
OUT_DIR.mkdir(exist_ok=True)

# -- Shared config -------------------------------------------------------------
EMBEDDING_DIM   = 512
IMG_SIZE        = 224
BATCH_SIZE      = 64   # AMP enables larger batches; 64 fits all backbones
N_CLASSES       = 31
LR_HEAD         = 1e-3
LR_BACKBONE     = 1e-5
EPOCHS_S1       = 5
EPOCHS_S2       = 25
WEIGHT_DECAY    = 1e-4
ARC_MARGIN      = 0.5
ARC_SCALE       = 30.0
GRAD_CLIP_NORM  = 1.0
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_AMP         = DEVICE == 'cuda'   # mixed precision only on GPU
VERSION         = 'v04'
CV_FOLD         = 0

BACKBONES = [
    'tf_efficientnetv2_s',    # TF-converted weights, reliably available in all timm versions
    'convnext_small',
    'vit_small_patch16_224',
]

print('\n' + '='*60)
print(f'  {VERSION}: Backbone Exploration')
print(f'  Comparing : {", ".join(BACKBONES)}')
print(f'  ArcFace   : m={ARC_MARGIN}, s={ARC_SCALE}')
print(f'  Device    : {DEVICE}  |  AMP: {USE_AMP}')
print(f'  Epochs    : S1={EPOCHS_S1} + S2={EPOCHS_S2} per backbone')
print(f'  Batch size: {BATCH_SIZE}')
print('='*60)

# -- Load data -----------------------------------------------------------------
print('\n[1] Loading data...')
train_df = pd.read_csv(KAGGLE_INPUT / 'train.csv')
test_df  = pd.read_csv(KAGGLE_INPUT / 'test.csv')
labels, label_to_idx, idx_to_label = encode_labels(train_df['ground_truth'])

for fold, train_idx, val_idx in get_identity_splits(train_df, n_splits=5):
    if fold == CV_FOLD:
        break
print(f'  Train fold: {len(train_idx)} | Val fold: {len(val_idx)}')

# -- Pre-cache ALL images to RAM (eliminates PNG decode bottleneck) ------------
print('\n[2] Pre-caching images to RAM...')
t_cache = time.time()
all_train_files  = train_df['filename'].tolist()
test_files_flat  = sorted(pd.unique(test_df[['query_image', 'gallery_image']].values.ravel()))
train_cache = build_image_cache(all_train_files, TRAIN_DIR, img_size=IMG_SIZE, verbose=True)
test_cache  = build_image_cache(test_files_flat,  TEST_DIR,  img_size=IMG_SIZE, verbose=True)
print(f'  Cache built in {time.time()-t_cache:.0f}s')

# Transforms: cached=True skips Resize (already done in build_image_cache)
tf_train = get_transforms(mode='train', img_size=IMG_SIZE, cached=True)
tf_val   = get_transforms(mode='val',   img_size=IMG_SIZE, cached=True)

# Build datasets (all share the same cache — zero extra memory overhead)
train_ds = JaguarDataset(
    filenames=train_df['filename'].iloc[train_idx].tolist(),
    img_dir=TRAIN_DIR, labels=labels[train_idx], transform=tf_train,
    image_cache=train_cache)
val_ds = JaguarDataset(
    filenames=train_df['filename'].iloc[val_idx].tolist(),
    img_dir=TRAIN_DIR, labels=labels[val_idx], transform=tf_val,
    image_cache=train_cache)
full_ds = JaguarDataset(
    filenames=all_train_files, img_dir=TRAIN_DIR, labels=labels,
    transform=tf_val, image_cache=train_cache)
full_train_ds = JaguarDataset(
    filenames=all_train_files, img_dir=TRAIN_DIR, labels=labels,
    transform=tf_train, image_cache=train_cache)
test_ds_all = JaguarDataset(
    filenames=test_files_flat, img_dir=TEST_DIR, labels=None,
    transform=tf_val, image_cache=test_cache)

train_loader     = DataLoader(train_ds,     batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=2, pin_memory=True)
val_loader       = DataLoader(val_ds,       batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=2, pin_memory=True)


# -- Model + training helpers --------------------------------------------------

class ArcFaceModel(nn.Module):
    def __init__(self, backbone_name, embedding_dim, pretrained=True):
        super().__init__()
        self.backbone = EmbeddingModel(backbone_name, pretrained=pretrained)
        native_dim    = self.backbone.out_dim
        self.proj     = nn.Sequential(
            nn.Linear(native_dim, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )
        self.out_dim = embedding_dim

    def forward(self, x):
        return F.normalize(self.proj(self.backbone(x)), p=2, dim=1)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(True)


def run_epoch(model, arcface_loss, loader, optimizer, scheduler,
              is_train, device, scaler):
    """Single epoch with AMP (no-op scaler when USE_AMP=False)."""
    model.train(is_train)
    arcface_loss.train(is_train)
    total_loss, n = 0.0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            if is_train:
                optimizer.zero_grad()
            with autocast(enabled=USE_AMP):
                embs = model(imgs)
                loss = arcface_loss(embs, targets)
            if is_train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(arcface_loss.parameters()),
                    max_norm=GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            total_loss += loss.item() * len(imgs)
            n          += len(imgs)
    return total_loss / n


def train_backbone(backbone_name):
    """Full two-stage ArcFace training for one backbone. Returns best val mAP."""
    print(f'\n{"="*55}')
    print(f'  Backbone: {backbone_name}')
    print(f'{"="*55}')

    model   = ArcFaceModel(backbone_name, EMBEDDING_DIM).to(DEVICE)
    arcface = ArcFaceLoss(EMBEDDING_DIM, N_CLASSES, ARC_MARGIN, ARC_SCALE).to(DEVICE)
    scaler  = GradScaler(enabled=USE_AMP)
    n_total = sum(p.numel() for p in model.parameters())
    print(f'  Native dim: {model.backbone.out_dim} | Total params: {n_total:,}')

    # Stage 1: head + ArcFace only
    model.freeze_backbone()
    head_params = list(filter(lambda p: p.requires_grad, model.parameters())) + \
                  list(arcface.parameters())
    opt1 = AdamW(head_params, lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    sch1 = OneCycleLR(opt1, max_lr=LR_HEAD,
                      steps_per_epoch=len(train_loader), epochs=EPOCHS_S1)
    for ep in range(1, EPOCHS_S1 + 1):
        t0 = time.time()
        tr = run_epoch(model, arcface, train_loader, opt1, sch1, True,  DEVICE, scaler)
        va = run_epoch(model, arcface, val_loader,   opt1, sch1, False, DEVICE, scaler)
        print(f'  S1 ep {ep}/{EPOCHS_S1} | loss {tr:.4f}/{va:.4f} | {time.time()-t0:.0f}s')

    # Stage 2: full fine-tune
    model.unfreeze_backbone()
    opt2 = AdamW([
        {'params': model.backbone.parameters(), 'lr': LR_BACKBONE},
        {'params': model.proj.parameters(),     'lr': LR_HEAD},
        {'params': arcface.parameters(),        'lr': LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)
    sch2 = OneCycleLR(opt2, max_lr=[LR_BACKBONE, LR_HEAD, LR_HEAD],
                      steps_per_epoch=len(train_loader), epochs=EPOCHS_S2)

    best_val_map  = 0.0
    best_state    = None

    for ep in range(1, EPOCHS_S2 + 1):
        t0 = time.time()
        tr = run_epoch(model, arcface, train_loader, opt2, sch2, True,  DEVICE, scaler)
        va = run_epoch(model, arcface, val_loader,   opt2, sch2, False, DEVICE, scaler)

        if ep % 5 == 0 or ep == EPOCHS_S2:
            val_embs    = extract_embeddings(model, val_ds, BATCH_SIZE, DEVICE,
                                             l2_normalise=False, desc='')
            val_results = compute_map(val_embs, labels[val_idx])
            vm = val_results['map']
            marker = ''
            if vm > best_val_map:
                best_val_map = vm
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                marker = ' **'
            print(f'  S2 ep {ep}/{EPOCHS_S2} | loss {tr:.4f}/{va:.4f} | '
                  f'mAP {vm:.4f}{marker} | {time.time()-t0:.0f}s')
        else:
            print(f'  S2 ep {ep}/{EPOCHS_S2} | loss {tr:.4f}/{va:.4f} | '
                  f'{time.time()-t0:.0f}s')

    model.load_state_dict(best_state)

    # Full-train mAP
    full_embs    = extract_embeddings(model, full_ds, BATCH_SIZE, DEVICE,
                                      l2_normalise=False, desc='Full train')
    full_results = compute_map(full_embs, labels)
    print(f'  {backbone_name}: val_mAP={best_val_map:.4f}, full_mAP={full_results["map"]:.4f}')

    return best_val_map, full_results, model


# -- Run backbone search -------------------------------------------------------
results_all  = {}   # backbone -> (val_map, full_results, model)
t_total = time.time()

for backbone in BACKBONES:
    val_map, full_results, model = train_backbone(backbone)
    results_all[backbone] = (val_map, full_results, model)
    # Free GPU memory before next backbone
    del model
    torch.cuda.empty_cache()

print(f'\nAll backbones trained in {(time.time()-t_total)/60:.0f} min')

# -- Summary -------------------------------------------------------------------
print(f'\n{"="*60}')
print(f'  {VERSION} — Backbone Comparison Summary')
print(f'  {"Backbone":<30} {"Val mAP":>8} {"Full mAP":>10}')
print(f'  {"-"*52}')
best_backbone = None
best_val = 0.0
for bb, (vm, fr, _) in results_all.items():
    marker = ' <-- best' if vm == max(v[0] for v in results_all.values()) else ''
    print(f'  {bb:<30} {vm:>8.4f} {fr["map"]:>10.4f}{marker}')
    if vm > best_val:
        best_val      = vm
        best_backbone = bb
print(f'{"="*60}')

# Save benchmark rows
for bb, (vm, fr, _) in results_all.items():
    save_benchmark(
        OUT_DIR / 'benchmarks_v04.csv',
        {
            'version':        f'{VERSION}_{bb.split("_")[0]}',
            'backbone':       bb,
            'loss':           f'ArcFace (m={ARC_MARGIN}, s={ARC_SCALE})',
            'embedding_dim':  EMBEDDING_DIM,
            'img_size':       IMG_SIZE,
            'augmentation':   'none (v05 adds augmentation)',
            'cv_map_val':     round(vm, 5),
            'cv_map_full':    round(fr['map'], 5),
            'best_identity':  idx_to_label[max(fr['per_identity_ap'],
                                               key=fr['per_identity_ap'].get)],
            'worst_identity': idx_to_label[min(fr['per_identity_ap'],
                                               key=fr['per_identity_ap'].get)],
            'epochs_s1':      EPOCHS_S1,
            'epochs_s2':      EPOCHS_S2,
            'notes':          f'Backbone search; best={best_backbone}; AMP+cache',
        }
    )


# -- Visualisations ------------------------------------------------------------
print('\n[Vis] Generating backbone comparison chart...')

bbs   = list(results_all.keys())
vmaps = [results_all[b][0]       for b in bbs]
fmaps = [results_all[b][1]['map'] for b in bbs]
x     = np.arange(len(bbs))
w     = 0.35

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].bar(x - w/2, vmaps, w, color='lightsteelblue', label='Val fold mAP')
axes[0].bar(x + w/2, fmaps, w, color='steelblue',      label='Full train mAP')
axes[0].axhline(0.68287, color='coral', linestyle='--', linewidth=1.5,
                label='v03 EfficientNet-B0 (full mAP=0.683)')
axes[0].set_xticks(x)
axes[0].set_xticklabels([b.replace('_', '\n') for b in bbs], fontsize=9)
axes[0].set_ylabel('Identity-balanced mAP'); axes[0].set_ylim(0, 1)
axes[0].set_title('v04: Backbone Comparison', fontsize=12, fontweight='bold')
axes[0].legend()

# Per-identity AP for the best backbone
best_per_id = {idx_to_label[k]: v
               for k, v in results_all[best_backbone][1]['per_identity_ap'].items()}
ids_sorted  = sorted(best_per_id, key=lambda x: -best_per_id[x])
xp = np.arange(len(ids_sorted))
axes[1].bar(xp, [best_per_id[i] for i in ids_sorted], color='steelblue')
axes[1].axhline(results_all[best_backbone][1]['map'], color='navy',
                linestyle='--', linewidth=1,
                label=f'mAP={results_all[best_backbone][1]["map"]:.4f}')
axes[1].set_xticks(xp)
axes[1].set_xticklabels(ids_sorted, rotation=45, ha='right', fontsize=7)
axes[1].set_ylabel('AP'); axes[1].set_ylim(0, 1)
axes[1].set_title(f'Per-identity AP: {best_backbone}', fontsize=11, fontweight='bold')
axes[1].legend(fontsize=9)

plt.suptitle('v04: Backbone Search Results', fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(OUT_DIR / f'{VERSION}_backbone_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved -> v04_backbone_comparison.png')


# -- Full retrain with best backbone + submission ------------------------------
print(f'\n[Final] Retraining best backbone ({best_backbone}) on ALL data...')

full_loader = DataLoader(full_train_ds, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=2, pin_memory=True)

model_final   = ArcFaceModel(best_backbone, EMBEDDING_DIM).to(DEVICE)
arcface_final = ArcFaceLoss(EMBEDDING_DIM, N_CLASSES, ARC_MARGIN, ARC_SCALE).to(DEVICE)
scaler_final  = GradScaler(enabled=USE_AMP)

# Stage 1
model_final.freeze_backbone()
head_f = list(filter(lambda p: p.requires_grad, model_final.parameters())) + \
         list(arcface_final.parameters())
opt_f1 = AdamW(head_f, lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
sch_f1 = OneCycleLR(opt_f1, max_lr=LR_HEAD,
                    steps_per_epoch=len(full_loader), epochs=EPOCHS_S1)
for ep in range(1, EPOCHS_S1 + 1):
    loss = run_epoch(model_final, arcface_final, full_loader,
                     opt_f1, sch_f1, True, DEVICE, scaler_final)
    print(f'  Full S1 ep {ep}/{EPOCHS_S1} | loss {loss:.4f}')

# Stage 2
model_final.unfreeze_backbone()
opt_f2 = AdamW([
    {'params': model_final.backbone.parameters(), 'lr': LR_BACKBONE},
    {'params': model_final.proj.parameters(),     'lr': LR_HEAD},
    {'params': arcface_final.parameters(),        'lr': LR_HEAD},
], weight_decay=WEIGHT_DECAY)
sch_f2 = OneCycleLR(opt_f2, max_lr=[LR_BACKBONE, LR_HEAD, LR_HEAD],
                    steps_per_epoch=len(full_loader), epochs=EPOCHS_S2)
for ep in range(1, EPOCHS_S2 + 1):
    loss = run_epoch(model_final, arcface_final, full_loader,
                     opt_f2, sch_f2, True, DEVICE, scaler_final)
    if ep % 5 == 0:
        print(f'  Full S2 ep {ep}/{EPOCHS_S2} | loss {loss:.4f}')

# Submission
model_final.eval()
test_embs = extract_embeddings(model_final, test_ds_all, BATCH_SIZE, DEVICE,
                                l2_normalise=False, desc='Test')
fname_to_idx = {f: i for i, f in enumerate(test_files_flat)}
make_submission(
    test_df=test_df, test_embeddings=test_embs,
    filename_to_idx=fname_to_idx,
    output_path=OUT_DIR / f'submission_{VERSION}.csv')

torch.save(model_final.state_dict(), OUT_DIR / f'model_{VERSION}_{best_backbone}.pth')

print(f'\n{"="*60}')
print(f'  {VERSION} COMPLETE')
print(f'  Best backbone: {best_backbone}  (val_mAP={best_val:.4f})')
for bb, (vm, fr, _) in results_all.items():
    print(f'  {bb:<30}: val={vm:.4f}  full={fr["map"]:.4f}')
print(f'  Total time   : {(time.time()-t_total)/60:.0f} min')
print(f'  Outputs in   : {OUT_DIR}')
print(f'{"="*60}')
print('\nDownload /kaggle/working/output/ zip and push figures + results to GitHub.')
