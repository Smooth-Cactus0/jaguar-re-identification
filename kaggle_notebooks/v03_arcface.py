"""
v03: ArcFace Metric Learning
==============================
Train EfficientNet-B0 with ArcFace loss — the standard approach for
closed-set re-identification tasks.

Key differences from v02 (classifier):
  Loss:    ArcFace (angular margin m=0.5, scale s=30) vs CrossEntropy
  Embedding: 512-dim projected (Linear + BN) vs 1280-dim raw
  Normalise: embeddings L2-normalised DURING training (required by ArcFace)
  LR:      backbone lr=1e-5 (lower than v02's 1e-4) — ArcFace gradients larger
  Monitor: val mAP every VAL_MAP_INTERVAL epochs; best checkpoint by mAP
  Clip:    gradient norm clipped at 1.0 for ArcFace stability

Strategy:
  Stage 1 (5 epochs)  — frozen backbone, train projection + ArcFace head
  Stage 2 (30 epochs) — full fine-tune, differential LR
  Then retrain on ALL data for final submission.

Run on Kaggle (T4 or P100 GPU):
  Attach the jaguar-re-id competition dataset.
  Outputs land in /kaggle/working/output/ — download the zip.
"""

import sys, os, time

# -- Clone / update repo -------------------------------------------------------
REPO_URL = 'https://github.com/Smooth-Cactus0/jaguar-re-identification.git'
REPO_DIR = '/kaggle/working/repo'

if os.path.exists(REPO_DIR):
    os.system(f'git -C {REPO_DIR} pull --ff-only')
else:
    os.system(f'git clone {REPO_URL} {REPO_DIR}')
sys.path.insert(0, REPO_DIR)

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.data      import JaguarDataset, get_transforms, encode_labels, get_identity_splits
from src.models    import EmbeddingModel
from src.losses    import ArcFaceLoss
from src.inference import extract_embeddings, make_submission
from src.evaluate  import compute_map, print_results, save_benchmark

# -- Paths (Kaggle layout) -----------------------------------------------------
KAGGLE_INPUT = Path('/kaggle/input/jaguar-re-id')
TRAIN_DIR    = KAGGLE_INPUT / 'train' / 'train'
TEST_DIR     = KAGGLE_INPUT / 'test'  / 'test'
OUT_DIR      = Path('/kaggle/working/output')
OUT_DIR.mkdir(exist_ok=True)

# -- Config --------------------------------------------------------------------
BACKBONE        = 'efficientnet_b0'
EMBEDDING_DIM   = 512          # projected dim (1280 -> 512 via Linear + BN)
IMG_SIZE        = 224
BATCH_SIZE      = 64
N_CLASSES       = 31
LR_HEAD         = 1e-3         # Stage 1 and ArcFace head in Stage 2
LR_BACKBONE     = 1e-5         # lower than v02 — ArcFace gradients are stronger
EPOCHS_S1       = 5
EPOCHS_S2       = 30
WEIGHT_DECAY    = 1e-4
ARC_MARGIN      = 0.5          # angular margin in radians (~28.6°)
ARC_SCALE       = 30.0         # feature scale; lower for small-scale re-ID
GRAD_CLIP_NORM  = 1.0          # gradient clipping for ArcFace stability
VAL_MAP_INTERVAL = 5           # compute val mAP every N epochs of stage 2
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'
VERSION         = 'v03'
CV_FOLD         = 0

print('\n' + '='*60)
print(f'  {VERSION}: ArcFace Metric Learning')
print(f'  Backbone      : {BACKBONE}')
print(f'  Embedding dim : {EMBEDDING_DIM}')
print(f'  ArcFace       : margin={ARC_MARGIN}, scale={ARC_SCALE}')
print(f'  Device        : {DEVICE}')
print(f'  Epochs        : S1={EPOCHS_S1} + S2={EPOCHS_S2}')
print(f'  Batch size    : {BATCH_SIZE}')
print('='*60)

# -- Load data -----------------------------------------------------------------
print('\n[1/8] Loading data...')
train_df = pd.read_csv(KAGGLE_INPUT / 'train.csv')
test_df  = pd.read_csv(KAGGLE_INPUT / 'test.csv')
labels, label_to_idx, idx_to_label = encode_labels(train_df['ground_truth'])
print(f'  Train: {len(train_df)} images, {N_CLASSES} identities')

# -- CV split ------------------------------------------------------------------
print(f'\n[2/8] Creating CV split (fold {CV_FOLD}/5)...')
for fold, train_idx, val_idx in get_identity_splits(train_df, n_splits=5):
    if fold == CV_FOLD:
        break

tf_train = get_transforms(mode='train', img_size=IMG_SIZE)
tf_val   = get_transforms(mode='val',   img_size=IMG_SIZE)

train_ds = JaguarDataset(
    filenames=train_df['filename'].iloc[train_idx].tolist(),
    img_dir=TRAIN_DIR, labels=labels[train_idx], transform=tf_train)
val_ds = JaguarDataset(
    filenames=train_df['filename'].iloc[val_idx].tolist(),
    img_dir=TRAIN_DIR, labels=labels[val_idx], transform=tf_val)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)
print(f'  Train: {len(train_ds)} imgs | Val: {len(val_ds)} imgs')


# -- Model architecture --------------------------------------------------------
print('\n[3/8] Building model...')


class ArcFaceModel(nn.Module):
    """
    EfficientNet backbone + projection head for ArcFace training.

    Forward always returns L2-normalised embeddings — ArcFaceLoss takes
    them directly. Same interface is used for inference (extract_embeddings).

    Projection: Linear(native_dim, embedding_dim, bias=False) + BatchNorm1d
    bias=False because BN provides its own learnable offset.
    """

    def __init__(self, backbone_name: str, embedding_dim: int,
                 pretrained: bool = True):
        super().__init__()
        self.backbone  = EmbeddingModel(backbone_name, pretrained=pretrained)
        native_dim     = self.backbone.out_dim
        self.proj      = nn.Sequential(
            nn.Linear(native_dim, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )
        self.out_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)                          # (B, native_dim)
        emb  = self.proj(feat)                           # (B, embedding_dim)
        return F.normalize(emb, p=2, dim=1)              # (B, embedding_dim), unit norm

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(True)


model   = ArcFaceModel(BACKBONE, EMBEDDING_DIM).to(DEVICE)
arcface = ArcFaceLoss(EMBEDDING_DIM, N_CLASSES, margin=ARC_MARGIN, scale=ARC_SCALE).to(DEVICE)

total_backbone = sum(p.numel() for p in model.backbone.parameters())
total_proj     = sum(p.numel() for p in model.proj.parameters())
total_arc      = sum(p.numel() for p in arcface.parameters())
print(f'  Backbone params : {total_backbone:,}')
print(f'  Projection params: {total_proj:,}')
print(f'  ArcFace params  : {total_arc:,}')


# -- Training utilities --------------------------------------------------------

def run_epoch(model, arcface_loss, loader, optimizer, scheduler, is_train, device):
    """Run one epoch; return average ArcFace loss."""
    model.train(is_train)
    arcface_loss.train(is_train)
    total_loss, n = 0.0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            if is_train:
                optimizer.zero_grad()
            embs = model(imgs)                # (B, 512), L2-normalised
            loss = arcface_loss(embs, targets)
            if is_train:
                loss.backward()
                # Clip gradients — ArcFace margin creates sharp spikes early in training
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(arcface_loss.parameters()),
                    max_norm=GRAD_CLIP_NORM)
                optimizer.step()
                scheduler.step()
            total_loss += loss.item() * len(imgs)
            n          += len(imgs)
    return total_loss / n


def compute_val_map(model, val_ds, val_labels, batch_size, device):
    """Extract val embeddings and return identity-balanced mAP."""
    embs    = extract_embeddings(model, val_ds, batch_size=batch_size,
                                 device=device, l2_normalise=False,  # already normalised
                                 desc='Val mAP')
    results = compute_map(embs, val_labels, identity_balanced=True)
    return results


# -- Stage 1: head-only training -----------------------------------------------
print('\n[4/8] Stage 1: projection + ArcFace head only (backbone frozen)...')
model.freeze_backbone()
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'  Trainable params: {trainable:,} + {total_arc:,} (ArcFace) = {trainable+total_arc:,}')

# Include ArcFace weight matrix in optimiser
head_params = list(filter(lambda p: p.requires_grad, model.parameters())) + \
              list(arcface.parameters())
opt_s1 = AdamW(head_params, lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
sch_s1 = OneCycleLR(opt_s1, max_lr=LR_HEAD,
                    steps_per_epoch=len(train_loader), epochs=EPOCHS_S1)

history = {'train_loss': [], 'val_loss': []}
for ep in range(1, EPOCHS_S1 + 1):
    t0 = time.time()
    tr_loss = run_epoch(model, arcface, train_loader, opt_s1, sch_s1, True,  DEVICE)
    va_loss = run_epoch(model, arcface, val_loader,   opt_s1, sch_s1, False, DEVICE)
    history['train_loss'].append(tr_loss)
    history['val_loss'].append(va_loss)
    print(f'  S1 ep {ep:02d}/{EPOCHS_S1} | '
          f'loss {tr_loss:.4f}/{va_loss:.4f} | '
          f'{time.time()-t0:.0f}s')


# -- Stage 2: full fine-tuning -------------------------------------------------
print('\n[5/8] Stage 2: full fine-tuning...')
model.unfreeze_backbone()
trainable_all = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'  Trainable params: {trainable_all:,} + {total_arc:,} (ArcFace)')

param_groups = [
    {'params': model.backbone.parameters(), 'lr': LR_BACKBONE},
    {'params': model.proj.parameters(),     'lr': LR_HEAD},
    {'params': arcface.parameters(),        'lr': LR_HEAD},
]
opt_s2 = AdamW(param_groups, weight_decay=WEIGHT_DECAY)
sch_s2 = OneCycleLR(opt_s2,
                    max_lr=[LR_BACKBONE, LR_HEAD, LR_HEAD],
                    steps_per_epoch=len(train_loader),
                    epochs=EPOCHS_S2)

best_val_map  = 0.0
best_state    = None
val_map_log   = []  # (epoch, val_mAP)

for ep in range(1, EPOCHS_S2 + 1):
    t0 = time.time()
    tr_loss = run_epoch(model, arcface, train_loader, opt_s2, sch_s2, True,  DEVICE)
    va_loss = run_epoch(model, arcface, val_loader,   opt_s2, sch_s2, False, DEVICE)
    history['train_loss'].append(tr_loss)
    history['val_loss'].append(va_loss)

    # Evaluate val mAP every VAL_MAP_INTERVAL epochs
    if ep % VAL_MAP_INTERVAL == 0 or ep == EPOCHS_S2:
        val_results = compute_val_map(model, val_ds, labels[val_idx],
                                      BATCH_SIZE, DEVICE)
        vm = val_results['map']
        val_map_log.append((EPOCHS_S1 + ep, vm))

        marker = ''
        if vm > best_val_map:
            best_val_map = vm
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = '  ** best mAP **'
        print(f'  S2 ep {ep:02d}/{EPOCHS_S2} | '
              f'loss {tr_loss:.4f}/{va_loss:.4f} | '
              f'val_mAP {vm:.4f} | '
              f'{time.time()-t0:.0f}s{marker}')
    else:
        print(f'  S2 ep {ep:02d}/{EPOCHS_S2} | '
              f'loss {tr_loss:.4f}/{va_loss:.4f} | '
              f'{time.time()-t0:.0f}s')

model.load_state_dict(best_state)
print(f'  Loaded best checkpoint (val_mAP={best_val_map:.4f})')


# -- Evaluate on validation fold -----------------------------------------------
print('\n[6/8] Computing final identity-balanced mAP...')
val_results = compute_val_map(model, val_ds, labels[val_idx], BATCH_SIZE, DEVICE)
print_results(val_results, idx_to_label, title=f'{VERSION} - Fold {CV_FOLD} Validation')

# Full train leave-one-out mAP for comparison with prior versions
full_ds = JaguarDataset(
    filenames=train_df['filename'].tolist(),
    img_dir=TRAIN_DIR, labels=labels, transform=tf_val)
full_embs    = extract_embeddings(model, full_ds, batch_size=BATCH_SIZE,
                                   device=DEVICE, l2_normalise=False,
                                   desc='Full train embeddings')
full_results = compute_map(full_embs, labels, identity_balanced=True)
print(f'\n  Full train leave-one-out mAP: {full_results["map"]:.4f}')
print(f'  v01: 0.2781  |  v02: 0.5258  |  v03: {full_results["map"]:.4f}')

save_benchmark(
    OUT_DIR / 'benchmarks_v03.csv',
    {
        'version':        VERSION,
        'backbone':       BACKBONE,
        'loss':           f'ArcFace (m={ARC_MARGIN}, s={ARC_SCALE})',
        'embedding_dim':  EMBEDDING_DIM,
        'img_size':       IMG_SIZE,
        'augmentation':   'none (v05 adds augmentation)',
        'cv_map_val':     round(val_results['map'],  5),
        'cv_map_full':    round(full_results['map'], 5),
        'best_identity':  idx_to_label[max(full_results['per_identity_ap'],
                                           key=full_results['per_identity_ap'].get)],
        'worst_identity': idx_to_label[min(full_results['per_identity_ap'],
                                           key=full_results['per_identity_ap'].get)],
        'epochs_s1':      EPOCHS_S1,
        'epochs_s2':      EPOCHS_S2,
        'notes':          'ArcFace, proj 512-dim BN, differential LR, best by val mAP',
    }
)


# -- Visualisations ------------------------------------------------------------
print('\n[7/8] Generating visualisations...')

# 7a. Training loss curves + val mAP overlay
ep_range   = list(range(1, EPOCHS_S1 + EPOCHS_S2 + 1))
map_epochs = [x[0] for x in val_map_log]
map_vals   = [x[1] for x in val_map_log]

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(ep_range, history['train_loss'], label='Train', color='steelblue')
axes[0].plot(ep_range, history['val_loss'],   label='Val',   color='coral')
axes[0].axvline(EPOCHS_S1, color='gray', linestyle='--', linewidth=1,
                label=f'Unfreeze at ep {EPOCHS_S1}')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('ArcFace Loss')
axes[0].set_title('Training Loss', fontweight='bold')
axes[0].legend()

ax2 = axes[1]
ax2.plot(map_epochs, map_vals, 'o-', color='steelblue', markersize=6, label='Val mAP (v03)')
ax2.axhline(0.4821, color='coral',  linestyle='--', linewidth=1, label='v02 val mAP 0.4821')
ax2.axhline(0.2781, color='gray',   linestyle=':',  linewidth=1, label='v01 mAP 0.2781')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Identity-balanced mAP')
ax2.set_title('Validation mAP', fontweight='bold')
ax2.set_ylim(0, 1); ax2.legend()

plt.suptitle(f'v03: ArcFace {BACKBONE} — Training Curves', fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(OUT_DIR / f'{VERSION}_training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved -> v03_training_curves.png')

# 7b. Per-identity AP bar chart: v01 vs v02 vs v03
v03_per_id = {idx_to_label[k]: v for k, v in full_results['per_identity_ap'].items()}
v01_per_id = {
    'Katniss':0.5694,'Lua':0.5186,'Bagua':0.5112,'Tomas':0.4515,
    'Kamaikua':0.3999,'Pyte':0.3825,'Benita':0.3686,'Pollyanna':0.3640,
    'Guaraci':0.3479,'Madalena':0.3408,'Pixana':0.3251,'Estella':0.3035,
    'Abril':0.2865,'Apeiara':0.2752,'Ariely':0.2585,'Ti':0.2456,
    'Alira':0.2420,'Akaloi':0.2377,'Solar':0.2270,'Ousado':0.2187,
    'Medrosa':0.2149,'Overa':0.2058,'Kwang':0.1977,'Marcela':0.1883,
    'Saseka':0.1687,'Oxum':0.1655,'Bororo':0.1604,'Bernard':0.1352,
    'Jaju':0.1332,'Ipepo':0.0968,'Patricia':0.0797,
}

ids_sorted = sorted(v03_per_id.keys(), key=lambda x: -v03_per_id[x])
x = np.arange(len(ids_sorted))
w = 0.28

fig, ax = plt.subplots(figsize=(16, 5))
ax.bar(x - w, [v01_per_id.get(i, 0) for i in ids_sorted], w,
       color='lightsteelblue', label='v01 frozen', edgecolor='white')
ax.bar(x,     [v03_per_id[i] for i in ids_sorted], w,
       color='steelblue', label='v03 ArcFace', edgecolor='white')
ax.axhline(full_results['map'], color='navy', linestyle='--', linewidth=1,
           label=f'v03 mAP={full_results["map"]:.4f}')
ax.axhline(0.2781, color='gray', linestyle=':', linewidth=1, label='v01 mAP=0.2781')
ax.set_xticks(x); ax.set_xticklabels(ids_sorted, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Average Precision'); ax.set_ylim(0, 1)
ax.set_title('v01 vs v03 ArcFace: Per-Identity AP', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(OUT_DIR / f'{VERSION}_per_identity_ap.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved -> v03_per_identity_ap.png')

# 7c. t-SNE of ArcFace embeddings
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
ax.set_title(f'v03: t-SNE of ArcFace {BACKBONE} Embeddings (512-dim)',
             fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=6, ncol=2, markerscale=1.5,
          framealpha=0.7, title='Identity')
ax.axis('off')
plt.tight_layout()
fig.savefig(OUT_DIR / f'{VERSION}_tsne.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved -> v03_tsne.png')


# -- Full retrain + submission -------------------------------------------------
print('\n[8/8] Retraining on ALL data for submission...')
full_train_ds = JaguarDataset(
    filenames=train_df['filename'].tolist(),
    img_dir=TRAIN_DIR, labels=labels, transform=tf_train)
full_loader = DataLoader(full_train_ds, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=2, pin_memory=True)

model_final   = ArcFaceModel(BACKBONE, EMBEDDING_DIM).to(DEVICE)
arcface_final = ArcFaceLoss(EMBEDDING_DIM, N_CLASSES,
                             margin=ARC_MARGIN, scale=ARC_SCALE).to(DEVICE)

# Stage 1: head only
model_final.freeze_backbone()
head_params_f = list(filter(lambda p: p.requires_grad, model_final.parameters())) + \
                list(arcface_final.parameters())
opt_f1 = AdamW(head_params_f, lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
sch_f1 = OneCycleLR(opt_f1, max_lr=LR_HEAD,
                    steps_per_epoch=len(full_loader), epochs=EPOCHS_S1)
for ep in range(1, EPOCHS_S1 + 1):
    loss = run_epoch(model_final, arcface_final, full_loader, opt_f1, sch_f1, True, DEVICE)
    print(f'  Full S1 ep {ep}/{EPOCHS_S1} | loss {loss:.4f}')

# Stage 2: full fine-tune
model_final.unfreeze_backbone()
param_groups_f = [
    {'params': model_final.backbone.parameters(), 'lr': LR_BACKBONE},
    {'params': model_final.proj.parameters(),     'lr': LR_HEAD},
    {'params': arcface_final.parameters(),        'lr': LR_HEAD},
]
opt_f2 = AdamW(param_groups_f, weight_decay=WEIGHT_DECAY)
sch_f2 = OneCycleLR(opt_f2,
                    max_lr=[LR_BACKBONE, LR_HEAD, LR_HEAD],
                    steps_per_epoch=len(full_loader),
                    epochs=EPOCHS_S2)
for ep in range(1, EPOCHS_S2 + 1):
    loss = run_epoch(model_final, arcface_final, full_loader, opt_f2, sch_f2, True, DEVICE)
    if ep % 5 == 0:
        print(f'  Full S2 ep {ep}/{EPOCHS_S2} | loss {loss:.4f}')

# Test embeddings + submission
test_filenames = sorted(pd.unique(
    test_df[['query_image', 'gallery_image']].values.ravel()))
test_ds = JaguarDataset(
    filenames=test_filenames, img_dir=TEST_DIR, labels=None, transform=tf_val)
model_final.eval()
test_embs = extract_embeddings(model_final, test_ds, batch_size=BATCH_SIZE,
                                device=DEVICE, l2_normalise=False,
                                desc='Test embeddings')
fname_to_idx = {f: i for i, f in enumerate(test_filenames)}
make_submission(
    test_df=test_df, test_embeddings=test_embs,
    filename_to_idx=fname_to_idx,
    output_path=OUT_DIR / f'submission_{VERSION}.csv')

torch.save(model_final.state_dict(), OUT_DIR / f'model_{VERSION}.pth')
print(f'  Model saved -> model_{VERSION}.pth')

print(f'\n{"="*60}')
print(f'  {VERSION} COMPLETE')
print(f'  Val fold mAP  : {val_results["map"]:.4f}')
print(f'  Full train mAP: {full_results["map"]:.4f}')
print(f'  Best val mAP  : {best_val_map:.4f}')
print(f'  Outputs in    : {OUT_DIR}')
print(f'{"="*60}')
print('\nDownload /kaggle/working/output/ zip and push figures + results to GitHub.')
