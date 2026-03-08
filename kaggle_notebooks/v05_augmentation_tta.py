"""
v05: Augmentation + Test-Time Augmentation
==========================================
Best backbone from v04: ConvNeXt-Small (full_mAP=0.8382, LB=0.797).

Two improvements over v04:
  1. TRAINING AUGMENTATION — use augment_level='heavy' during training:
       RandomRotation(10°), aggressive ColorJitter, RandomErasing(p=0.25)
     Forces the model to learn robust features rather than memorising
     exact viewpoint/lighting conditions.

  2. TEST-TIME AUGMENTATION (TTA) — at inference, run each image through
     N augmented views and average the L2-normalised embeddings:
       - original centre crop
       - horizontal flip
       - top-left crop, top-right crop, bottom-centre crop
     Averaging unit-norm vectors on the hypersphere is equivalent to
     finding the centroid direction — more robust than a single view.

Both changes require no architecture modification; same
ConvNeXt-Small + ArcFace(m=0.5, s=30) + 512-dim projection.
Caching + AMP retained for speed.
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
from torchvision import transforms
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.data      import (JaguarDataset, get_transforms, encode_labels,
                            get_identity_splits, build_image_cache,
                            IMAGENET_MEAN, IMAGENET_STD)
from src.models    import EmbeddingModel
from src.losses    import ArcFaceLoss
from src.inference import make_submission
from src.evaluate  import compute_map, print_results, save_benchmark

# -- Paths ---------------------------------------------------------------------
KAGGLE_INPUT = Path('/kaggle/input/competitions/jaguar-re-id')
TRAIN_DIR    = KAGGLE_INPUT / 'train' / 'train'
TEST_DIR     = KAGGLE_INPUT / 'test'  / 'test'
OUT_DIR      = Path('/kaggle/working/output')
OUT_DIR.mkdir(exist_ok=True)

# -- Config --------------------------------------------------------------------
BACKBONE       = 'convnext_small'   # winner from v04
EMBEDDING_DIM  = 512
IMG_SIZE       = 224
BATCH_SIZE     = 64
N_CLASSES      = 31
LR_HEAD        = 1e-3
LR_BACKBONE    = 1e-5
EPOCHS_S1      = 5
EPOCHS_S2      = 25
WEIGHT_DECAY   = 1e-4
ARC_MARGIN     = 0.5
ARC_SCALE      = 30.0
GRAD_CLIP_NORM = 1.0
TTA_VIEWS      = 5     # number of augmented views for TTA
DEVICE         = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_AMP        = DEVICE == 'cuda'
VERSION        = 'v05'
CV_FOLD        = 0

print('\n' + '='*60)
print(f'  {VERSION}: Augmentation + TTA')
print(f'  Backbone   : {BACKBONE}')
print(f'  ArcFace    : m={ARC_MARGIN}, s={ARC_SCALE}')
print(f'  Aug level  : heavy')
print(f'  TTA views  : {TTA_VIEWS}')
print(f'  Device     : {DEVICE}  |  AMP: {USE_AMP}')
print(f'  Epochs     : S1={EPOCHS_S1} + S2={EPOCHS_S2}')
print('='*60)

# -- Load data -----------------------------------------------------------------
print('\n[1] Loading data...')
train_df = pd.read_csv(KAGGLE_INPUT / 'train.csv')
test_df  = pd.read_csv(KAGGLE_INPUT / 'test.csv')
labels, label_to_idx, idx_to_label = encode_labels(train_df['ground_truth'])

for fold, train_idx, val_idx in get_identity_splits(train_df, n_splits=5):
    if fold == CV_FOLD:
        break

# -- Pre-cache images ----------------------------------------------------------
print('\n[2] Pre-caching images...')
all_train_files = train_df['filename'].tolist()
test_files_flat = sorted(pd.unique(test_df[['query_image', 'gallery_image']].values.ravel()))
train_cache = build_image_cache(all_train_files, TRAIN_DIR, img_size=IMG_SIZE, verbose=True)
test_cache  = build_image_cache(test_files_flat,  TEST_DIR,  img_size=IMG_SIZE, verbose=True)

# Heavy augmentation for training; clean val transforms
tf_train = get_transforms(mode='train', img_size=IMG_SIZE, augment_level='heavy', cached=True)
tf_val   = get_transforms(mode='val',   img_size=IMG_SIZE, cached=True)

train_ds = JaguarDataset(
    filenames=train_df['filename'].iloc[train_idx].tolist(),
    img_dir=TRAIN_DIR, labels=labels[train_idx],
    transform=tf_train, image_cache=train_cache)
val_ds = JaguarDataset(
    filenames=train_df['filename'].iloc[val_idx].tolist(),
    img_dir=TRAIN_DIR, labels=labels[val_idx],
    transform=tf_val, image_cache=train_cache)
full_ds = JaguarDataset(
    filenames=all_train_files, img_dir=TRAIN_DIR, labels=labels,
    transform=tf_val, image_cache=train_cache)
full_train_ds = JaguarDataset(
    filenames=all_train_files, img_dir=TRAIN_DIR, labels=labels,
    transform=tf_train, image_cache=train_cache)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)
print(f'  Train: {len(train_ds)} | Val: {len(val_ds)}')


# -- TTA utilities -------------------------------------------------------------

def get_tta_transforms(img_size: int):
    """
    Returns a list of deterministic transforms for TTA.
    All produce (img_size, img_size) tensors from a (img_size+32) cached PIL image.

    Views:
      0: centre crop (same as val)
      1: horizontal flip + centre crop
      2: top-left crop
      3: top-right crop
      4: bottom-centre crop
    """
    norm = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    pad  = img_size + 32  # cached size
    off  = 32             # = pad - img_size; offset for corner crops

    return [
        transforms.Compose([transforms.CenterCrop(img_size)] + norm),
        transforms.Compose([transforms.RandomHorizontalFlip(p=1.0),
                             transforms.CenterCrop(img_size)] + norm),
        transforms.Compose([transforms.Lambda(lambda img: img.crop((0, 0, img_size, img_size)))] + norm),
        transforms.Compose([transforms.Lambda(lambda img: img.crop((off, 0, pad, img_size)))] + norm),
        transforms.Compose([transforms.Lambda(lambda img: img.crop((off//2, off, off//2+img_size, pad)))] + norm),
    ]


def extract_embeddings_tta(model, filenames, img_dir, cache, img_size,
                            batch_size, device):
    """
    Extract L2-normalised embeddings with TTA.

    For each image, runs TTA_VIEWS augmented views through the model and
    averages the resulting unit-norm embeddings. The average is re-normalised
    to stay on the hypersphere.

    Returns: (N, embedding_dim) numpy array
    """
    tta_tfs = get_tta_transforms(img_size)
    model.eval()

    all_embs = []   # will be (N, TTA_VIEWS, D)

    for tf in tta_tfs:
        ds = JaguarDataset(filenames=filenames, img_dir=img_dir,
                           labels=None, transform=tf, image_cache=cache)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
        view_embs = []
        with torch.no_grad():
            for imgs in loader:
                imgs = imgs.to(device)
                with autocast(enabled=USE_AMP):
                    embs = model(imgs)   # already L2-normalised
                view_embs.append(embs.cpu().numpy())
        all_embs.append(np.concatenate(view_embs, axis=0))

    # Stack: (TTA_VIEWS, N, D) -> mean -> (N, D) -> re-normalise
    stacked = np.stack(all_embs, axis=0)   # (V, N, D)
    avg     = stacked.mean(axis=0)         # (N, D)
    norms   = np.linalg.norm(avg, axis=1, keepdims=True).clip(min=1e-8)
    return avg / norms


# -- Model ---------------------------------------------------------------------

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


print('\n[3] Building model...')
model   = ArcFaceModel(BACKBONE, EMBEDDING_DIM).to(DEVICE)
arcface = ArcFaceLoss(EMBEDDING_DIM, N_CLASSES, ARC_MARGIN, ARC_SCALE).to(DEVICE)
scaler  = GradScaler(enabled=USE_AMP)
print(f'  Backbone dim: {model.backbone.out_dim} | '
      f'Total params: {sum(p.numel() for p in model.parameters()):,}')


# -- Training ------------------------------------------------------------------

def run_epoch(model, arcface_loss, loader, optimizer, scheduler,
              is_train, device, scaler):
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
                loss = arcface_loss(model(imgs), targets)
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


print('\n[4] Stage 1: head only...')
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

print('\n[5] Stage 2: full fine-tune...')
model.unfreeze_backbone()
opt2 = AdamW([
    {'params': model.backbone.parameters(), 'lr': LR_BACKBONE},
    {'params': model.proj.parameters(),     'lr': LR_HEAD},
    {'params': arcface.parameters(),        'lr': LR_HEAD},
], weight_decay=WEIGHT_DECAY)
sch2 = OneCycleLR(opt2, max_lr=[LR_BACKBONE, LR_HEAD, LR_HEAD],
                  steps_per_epoch=len(train_loader), epochs=EPOCHS_S2)

best_val_map = 0.0
best_state   = None

for ep in range(1, EPOCHS_S2 + 1):
    t0 = time.time()
    tr = run_epoch(model, arcface, train_loader, opt2, sch2, True,  DEVICE, scaler)
    va = run_epoch(model, arcface, val_loader,   opt2, sch2, False, DEVICE, scaler)

    if ep % 5 == 0 or ep == EPOCHS_S2:
        # Val mAP without TTA (fast)
        val_embs    = []
        model.eval()
        with torch.no_grad():
            for imgs, _ in val_loader:
                with autocast(enabled=USE_AMP):
                    val_embs.append(model(imgs.to(DEVICE)).cpu().numpy())
        val_embs    = np.concatenate(val_embs, axis=0)
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
        print(f'  S2 ep {ep}/{EPOCHS_S2} | loss {tr:.4f}/{va:.4f} | {time.time()-t0:.0f}s')

model.load_state_dict(best_state)
print(f'  Loaded best checkpoint (val_mAP={best_val_map:.4f})')


# -- Evaluate: val fold without TTA and with TTA -------------------------------
print('\n[6] Evaluating val fold (no TTA vs TTA)...')

# Without TTA
val_embs_plain = []
model.eval()
with torch.no_grad():
    for imgs, _ in val_loader:
        with autocast(enabled=USE_AMP):
            val_embs_plain.append(model(imgs.to(DEVICE)).cpu().numpy())
val_embs_plain = np.concatenate(val_embs_plain, axis=0)
res_plain = compute_map(val_embs_plain, labels[val_idx])
print(f'  Val mAP (no TTA) : {res_plain["map"]:.4f}')

# With TTA
val_filenames = train_df['filename'].iloc[val_idx].tolist()
val_embs_tta  = extract_embeddings_tta(
    model, val_filenames, TRAIN_DIR, train_cache, IMG_SIZE, BATCH_SIZE, DEVICE)
res_tta = compute_map(val_embs_tta, labels[val_idx])
print(f'  Val mAP (TTA x{TTA_VIEWS}): {res_tta["map"]:.4f}  '
      f'(delta: {res_tta["map"] - res_plain["map"]:+.4f})')

print_results(res_tta, idx_to_label, title=f'{VERSION} - Fold {CV_FOLD} Val (TTA)')

# Full train mAP for reference
full_embs    = []
full_loader_eval = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=2, pin_memory=True)
model.eval()
with torch.no_grad():
    for imgs, _ in full_loader_eval:
        with autocast(enabled=USE_AMP):
            full_embs.append(model(imgs.to(DEVICE)).cpu().numpy())
full_embs    = np.concatenate(full_embs, axis=0)
full_results = compute_map(full_embs, labels)
print(f'\n  Full train mAP (no TTA): {full_results["map"]:.4f}')
print(f'  v04 ConvNeXt was 0.8382 — delta: {full_results["map"]-0.8382:+.4f}')

save_benchmark(
    OUT_DIR / 'benchmarks_v05.csv',
    {
        'version':        VERSION,
        'backbone':       BACKBONE,
        'loss':           f'ArcFace (m={ARC_MARGIN}, s={ARC_SCALE})',
        'embedding_dim':  EMBEDDING_DIM,
        'img_size':       IMG_SIZE,
        'augmentation':   'heavy',
        'cv_map_val':     round(res_tta['map'], 5),
        'cv_map_val_noTTA': round(res_plain['map'], 5),
        'cv_map_full':    round(full_results['map'], 5),
        'best_identity':  idx_to_label[max(res_tta['per_identity_ap'],
                                           key=res_tta['per_identity_ap'].get)],
        'worst_identity': idx_to_label[min(res_tta['per_identity_ap'],
                                           key=res_tta['per_identity_ap'].get)],
        'epochs_s1':      EPOCHS_S1,
        'epochs_s2':      EPOCHS_S2,
        'notes':          f'ConvNeXt-Small + heavy aug + TTA x{TTA_VIEWS}',
    }
)


# -- Visualisations ------------------------------------------------------------
print('\n[7] Generating visualisations...')

# Per-identity AP: TTA vs no-TTA vs v04
v05_per_id = {idx_to_label[k]: v for k, v in res_tta['per_identity_ap'].items()}
v04_per_id_ref = {idx_to_label[k]: v for k, v in res_plain['per_identity_ap'].items()}
ids_sorted = sorted(v05_per_id.keys(), key=lambda x: -v05_per_id[x])
x = np.arange(len(ids_sorted))
w = 0.38

fig, ax = plt.subplots(figsize=(16, 5))
ax.bar(x - w/2, [v04_per_id_ref.get(i, 0) for i in ids_sorted], w,
       color='lightsteelblue', label='v05 no TTA', edgecolor='white')
ax.bar(x + w/2, [v05_per_id[i] for i in ids_sorted], w,
       color='steelblue', label=f'v05 TTA x{TTA_VIEWS}', edgecolor='white')
ax.axhline(res_tta['map'], color='navy', linestyle='--', linewidth=1,
           label=f'v05 TTA mAP={res_tta["map"]:.4f}')
ax.axhline(res_plain['map'], color='gray', linestyle=':', linewidth=1,
           label=f'v05 no-TTA mAP={res_plain["map"]:.4f}')
ax.set_xticks(x); ax.set_xticklabels(ids_sorted, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Average Precision'); ax.set_ylim(0, 1)
ax.set_title('v05: Per-Identity AP — TTA vs No TTA', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(OUT_DIR / f'{VERSION}_per_identity_ap.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved -> v05_per_identity_ap.png')

# t-SNE
print('  Computing t-SNE...')
from sklearn.manifold import TSNE
import seaborn as sns

tsne   = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, n_jobs=-1)
embs2d = tsne.fit_transform(full_embs)
palette = sns.color_palette('tab20', N_CLASSES) + \
          sns.color_palette('tab20b', max(0, N_CLASSES - 20))

fig, ax = plt.subplots(figsize=(12, 10))
for lbl in range(N_CLASSES):
    mask = labels == lbl
    ax.scatter(embs2d[mask, 0], embs2d[mask, 1], color=palette[lbl],
               label=idx_to_label[lbl], s=20, alpha=0.7, edgecolors='none')
ax.set_title(f'v05: t-SNE — ConvNeXt-Small + Heavy Aug', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=6, ncol=2, markerscale=1.5,
          framealpha=0.7, title='Identity')
ax.axis('off')
plt.tight_layout()
fig.savefig(OUT_DIR / f'{VERSION}_tsne.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved -> v05_tsne.png')


# -- Full retrain + TTA submission ---------------------------------------------
print('\n[8] Retraining on ALL data + TTA submission...')

full_loader = DataLoader(full_train_ds, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=2, pin_memory=True)

model_final   = ArcFaceModel(BACKBONE, EMBEDDING_DIM).to(DEVICE)
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

# TTA submission
print(f'  Extracting test embeddings with TTA x{TTA_VIEWS}...')
model_final.eval()
test_embs_tta = extract_embeddings_tta(
    model_final, test_files_flat, TEST_DIR, test_cache, IMG_SIZE, BATCH_SIZE, DEVICE)
fname_to_idx  = {f: i for i, f in enumerate(test_files_flat)}
make_submission(
    test_df=test_df, test_embeddings=test_embs_tta,
    filename_to_idx=fname_to_idx,
    output_path=OUT_DIR / f'submission_{VERSION}.csv')

torch.save(model_final.state_dict(), OUT_DIR / f'model_{VERSION}.pth')

print(f'\n{"="*60}')
print(f'  {VERSION} COMPLETE')
print(f'  Val mAP no-TTA: {res_plain["map"]:.4f}')
print(f'  Val mAP TTA x{TTA_VIEWS}: {res_tta["map"]:.4f}')
print(f'  Full train mAP : {full_results["map"]:.4f}')
print(f'  Outputs in     : {OUT_DIR}')
print(f'{"="*60}')
print('\nDownload /kaggle/working/output/ zip and push figures + results to GitHub.')
