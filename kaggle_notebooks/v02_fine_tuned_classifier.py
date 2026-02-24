"""
v02: Fine-Tuned 31-Class Classifier
=====================================
Fine-tune EfficientNet-B0 end-to-end as a 31-class jaguar classifier.
Use the penultimate layer embeddings (1280-dim) for similarity scoring.

Strategy:
  Stage 1 (5 epochs)  - frozen backbone, train head only  (lr_head=1e-3)
  Stage 2 (25 epochs) - full fine-tune with differential LR
                        (lr_backbone=1e-4, lr_head=1e-3)
  Then retrain on ALL data with same recipe for final submission.

Run on Kaggle (T4 or P100 GPU):
  Upload this script as a Kaggle notebook kernel.
  Competition data must be attached as input dataset.
  Outputs land in /kaggle/working/ — download the zip.
"""

import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# -- Clone repo and add src/ to path -----------------------------------------
REPO_URL = 'https://github.com/Smooth-Cactus0/jaguar-re-identification.git'
REPO_DIR = '/kaggle/working/repo'

if not os.path.exists(REPO_DIR):
    os.system(f'git clone {REPO_URL} {REPO_DIR}')
sys.path.insert(0, REPO_DIR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.data      import JaguarDataset, get_transforms, encode_labels, get_identity_splits
from src.models    import EmbeddingModel
from src.inference import extract_embeddings, make_submission
from src.evaluate  import compute_map, print_results, save_benchmark

# -- Paths (Kaggle layout) ----------------------------------------------------
KAGGLE_INPUT = Path('/kaggle/input/jaguar-re-id')
TRAIN_DIR    = KAGGLE_INPUT / 'train' / 'train'
TEST_DIR     = KAGGLE_INPUT / 'test'  / 'test'
OUT_DIR      = Path('/kaggle/working/output')
OUT_DIR.mkdir(exist_ok=True)

# -- Config -------------------------------------------------------------------
BACKBONE    = 'efficientnet_b0'
IMG_SIZE    = 224
BATCH_SIZE  = 64
N_CLASSES   = 31
LR_HEAD     = 1e-3
LR_BACKBONE = 1e-4
EPOCHS_S1   = 5      # Stage 1: head only
EPOCHS_S2   = 25     # Stage 2: full fine-tune
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.1   # label smoothing regularises classification on small dataset
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
VERSION      = 'v02'
CV_FOLD      = 0     # use fold 0 for validation (80% train / 20% val)

print('\n' + '='*60)
print(f'  {VERSION}: Fine-Tuned Classifier')
print(f'  Backbone   : {BACKBONE}')
print(f'  Device     : {DEVICE}')
print(f'  Img size   : {IMG_SIZE}')
print(f'  Epochs     : S1={EPOCHS_S1} + S2={EPOCHS_S2}')
print(f'  Batch size : {BATCH_SIZE}')
print('='*60)

# -- Load data ----------------------------------------------------------------
print('\n[1/8] Loading data...')
train_df = pd.read_csv(KAGGLE_INPUT / 'train.csv')
test_df  = pd.read_csv(KAGGLE_INPUT / 'test.csv')
labels, label_to_idx, idx_to_label = encode_labels(train_df['ground_truth'])
print(f'  Train: {len(train_df)} images, {N_CLASSES} identities')

# -- CV split (GroupKFold by identity, 5 folds, use fold 0) ------------------
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
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)
print(f'  Train: {len(train_ds)} imgs | Val: {len(val_ds)} imgs')


# -- Model + loss + optimiser -------------------------------------------------
print('\n[3/8] Building model...')

class ClassifierModel(nn.Module):
    """EfficientNet backbone + linear head for 31-class classification."""
    def __init__(self, backbone_name, n_classes, pretrained=True):
        super().__init__()
        self.backbone   = EmbeddingModel(backbone_name, pretrained=pretrained)
        self.classifier = nn.Linear(self.backbone.out_dim, n_classes)

    def forward(self, x, return_embedding=False):
        emb = self.backbone(x)
        if return_embedding:
            return nn.functional.normalize(emb, p=2, dim=1)
        return self.classifier(emb)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(True)


model = ClassifierModel(BACKBONE, N_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

total_params = sum(p.numel() for p in model.parameters())
print(f'  Total params: {total_params:,}')


# -- Training utilities -------------------------------------------------------
def run_epoch(model, loader, optimizer, scheduler, is_train, device):
    model.train(is_train)
    total_loss, correct, n = 0.0, 0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            if is_train:
                optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, targets)
            if is_train:
                loss.backward()
                optimizer.step()
                scheduler.step()
            total_loss += loss.item() * len(imgs)
            correct    += (logits.argmax(1) == targets).sum().item()
            n          += len(imgs)
    return total_loss / n, correct / n


# -- Stage 1: head-only training ----------------------------------------------
print('\n[4/8] Stage 1: head-only training...')
model.freeze_backbone()
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'  Trainable params (head only): {trainable:,}')

opt_s1 = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
               lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
sch_s1 = OneCycleLR(opt_s1, max_lr=LR_HEAD,
                    steps_per_epoch=len(train_loader), epochs=EPOCHS_S1)

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
for ep in range(1, EPOCHS_S1 + 1):
    t0 = time.time()
    tr_loss, tr_acc = run_epoch(model, train_loader, opt_s1, sch_s1, True,  DEVICE)
    va_loss, va_acc = run_epoch(model, val_loader,   opt_s1, sch_s1, False, DEVICE)
    history['train_loss'].append(tr_loss); history['val_loss'].append(va_loss)
    history['train_acc'].append(tr_acc);  history['val_acc'].append(va_acc)
    print(f'  S1 ep {ep:02d}/{EPOCHS_S1} | '
          f'loss {tr_loss:.4f}/{va_loss:.4f} | '
          f'acc {tr_acc:.3f}/{va_acc:.3f} | '
          f'{time.time()-t0:.0f}s')


# -- Stage 2: full fine-tuning ------------------------------------------------
print('\n[5/8] Stage 2: full fine-tuning...')
model.unfreeze_backbone()
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'  Trainable params (all): {trainable:,}')

# Differential learning rates: backbone 10x lower than head
param_groups = [
    {'params': model.backbone.parameters(),   'lr': LR_BACKBONE},
    {'params': model.classifier.parameters(), 'lr': LR_HEAD},
]
opt_s2 = AdamW(param_groups, weight_decay=WEIGHT_DECAY)
sch_s2 = OneCycleLR(opt_s2,
                    max_lr=[LR_BACKBONE, LR_HEAD],
                    steps_per_epoch=len(train_loader),
                    epochs=EPOCHS_S2)

best_val_loss = float('inf')
best_state    = None

for ep in range(1, EPOCHS_S2 + 1):
    t0 = time.time()
    tr_loss, tr_acc = run_epoch(model, train_loader, opt_s2, sch_s2, True,  DEVICE)
    va_loss, va_acc = run_epoch(model, val_loader,   opt_s2, sch_s2, False, DEVICE)
    history['train_loss'].append(tr_loss); history['val_loss'].append(va_loss)
    history['train_acc'].append(tr_acc);  history['val_acc'].append(va_acc)

    if va_loss < best_val_loss:
        best_val_loss = va_loss
        best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f'  S2 ep {ep:02d}/{EPOCHS_S2} | '
              f'loss {tr_loss:.4f}/{va_loss:.4f} | '
              f'acc {tr_acc:.3f}/{va_acc:.3f} | '
              f'{time.time()-t0:.0f}s  ** best **')
    else:
        print(f'  S2 ep {ep:02d}/{EPOCHS_S2} | '
              f'loss {tr_loss:.4f}/{va_loss:.4f} | '
              f'acc {tr_acc:.3f}/{va_acc:.3f} | '
              f'{time.time()-t0:.0f}s')

model.load_state_dict(best_state)
print(f'  Loaded best checkpoint (val_loss={best_val_loss:.4f})')


# -- Evaluate: identity-balanced mAP on validation fold ----------------------
print('\n[6/8] Computing identity-balanced mAP on validation fold...')
model.eval()

# Extract embeddings using the backbone (not classifier logits)
val_embs = extract_embeddings(
    lambda x: model(x, return_embedding=True),
    val_ds, batch_size=BATCH_SIZE, device=DEVICE, desc='Val embeddings')

val_results = compute_map(val_embs, labels[val_idx], identity_balanced=True)
print_results(val_results, idx_to_label, title=f'{VERSION} - Fold {CV_FOLD} Validation')

# Also compute on full training set for comparison with v01
full_ds   = JaguarDataset(
    filenames=train_df['filename'].tolist(),
    img_dir=TRAIN_DIR, labels=labels, transform=tf_val)
full_embs = extract_embeddings(
    lambda x: model(x, return_embedding=True),
    full_ds, batch_size=BATCH_SIZE, device=DEVICE, desc='Full train embeddings')
full_results = compute_map(full_embs, labels, identity_balanced=True)
print(f'\n  Full train leave-one-out mAP: {full_results["map"]:.4f}')
print(f'  (v01 was 0.2781 — improvement: '
      f'{full_results["map"] - 0.2781:+.4f})')

save_benchmark(
    OUT_DIR / 'benchmarks_v02.csv',
    {
        'version':        VERSION,
        'backbone':       BACKBONE,
        'loss':           'CrossEntropy (label_smooth=0.1)',
        'embedding_dim':  model.backbone.out_dim,
        'img_size':       IMG_SIZE,
        'augmentation':   'none (v05 adds augmentation)',
        'cv_map_val':     round(val_results['map'], 5),
        'cv_map_full':    round(full_results['map'], 5),
        'best_identity':  idx_to_label[max(full_results['per_identity_ap'],
                                           key=full_results['per_identity_ap'].get)],
        'worst_identity': idx_to_label[min(full_results['per_identity_ap'],
                                           key=full_results['per_identity_ap'].get)],
        'epochs_s1':      EPOCHS_S1,
        'epochs_s2':      EPOCHS_S2,
        'notes':          'Two-stage fine-tune, differential LR',
    }
)


# -- Visualisations -----------------------------------------------------------
print('\n[7/8] Generating visualisations...')

# 7a. Training curves
ep_range = range(1, EPOCHS_S1 + EPOCHS_S2 + 1)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].plot(ep_range, history['train_loss'], label='Train', color='steelblue')
axes[0].plot(ep_range, history['val_loss'],   label='Val',   color='coral')
axes[0].axvline(EPOCHS_S1, color='gray', linestyle='--', linewidth=1,
                label=f'Unfreeze at ep {EPOCHS_S1}')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss', fontweight='bold')
axes[0].legend()

axes[1].plot(ep_range, history['train_acc'], label='Train', color='steelblue')
axes[1].plot(ep_range, history['val_acc'],   label='Val',   color='coral')
axes[1].axvline(EPOCHS_S1, color='gray', linestyle='--', linewidth=1)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].set_title('Classification Accuracy', fontweight='bold')
axes[1].legend()

plt.suptitle(f'v02: Fine-Tuned {BACKBONE} — Training Curves', fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(OUT_DIR / f'{VERSION}_training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved -> v02_training_curves.png')

# 7b. Per-identity AP comparison (v01 vs v02)
# Load v01 results for comparison if available
v02_per_id = {idx_to_label[k]: v for k, v in full_results['per_identity_ap'].items()}
v01_map_ref = {  # from the v01 run results
    'Katniss':0.5694,'Lua':0.5186,'Bagua':0.5112,'Tomas':0.4515,
    'Kamaikua':0.3999,'Pyte':0.3825,'Benita':0.3686,'Pollyanna':0.3640,
    'Guaraci':0.3479,'Madalena':0.3408,'Pixana':0.3251,'Estella':0.3035,
    'Abril':0.2865,'Apeiara':0.2752,'Ariely':0.2585,'Ti':0.2456,
    'Alira':0.2420,'Akaloi':0.2377,'Solar':0.2270,'Ousado':0.2187,
    'Medrosa':0.2149,'Overa':0.2058,'Kwang':0.1977,'Marcela':0.1883,
    'Saseka':0.1687,'Oxum':0.1655,'Bororo':0.1604,'Bernard':0.1352,
    'Jaju':0.1332,'Ipepo':0.0968,'Patricia':0.0797,
}
ids_sorted = sorted(v02_per_id.keys(), key=lambda x: -v02_per_id[x])
x = np.arange(len(ids_sorted))
w = 0.38

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x - w/2, [v01_map_ref.get(i, 0) for i in ids_sorted], w,
       color='lightsteelblue', label='v01 frozen', edgecolor='white')
ax.bar(x + w/2, [v02_per_id[i] for i in ids_sorted], w,
       color='steelblue', label='v02 fine-tuned', edgecolor='white')
ax.axhline(full_results['map'], color='navy', linestyle='--', linewidth=1,
           label=f'v02 mAP={full_results["map"]:.4f}')
ax.axhline(0.2781, color='gray', linestyle=':', linewidth=1,
           label='v01 mAP=0.2781')
ax.set_xticks(x); ax.set_xticklabels(ids_sorted, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Average Precision'); ax.set_ylim(0, 1)
ax.set_title('v01 vs v02: Per-Identity AP Comparison', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(OUT_DIR / f'{VERSION}_per_identity_ap.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved -> v02_per_identity_ap.png')

# 7c. t-SNE of fine-tuned embeddings
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
ax.set_title(f'v02: t-SNE of Fine-Tuned {BACKBONE} Embeddings', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=6, ncol=2, markerscale=1.5,
          framealpha=0.7, title='Identity')
ax.axis('off')
plt.tight_layout()
fig.savefig(OUT_DIR / f'{VERSION}_tsne.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved -> v02_tsne.png')


# -- Full retrain + submission ------------------------------------------------
print('\n[8/8] Retraining on ALL data for submission...')
full_train_ds = JaguarDataset(
    filenames=train_df['filename'].tolist(),
    img_dir=TRAIN_DIR, labels=labels, transform=tf_train)
full_loader = DataLoader(full_train_ds, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=2, pin_memory=True)

model_final = ClassifierModel(BACKBONE, N_CLASSES).to(DEVICE)

# Stage 1
model_final.freeze_backbone()
opt_f1 = AdamW(filter(lambda p: p.requires_grad, model_final.parameters()),
               lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
sch_f1 = OneCycleLR(opt_f1, max_lr=LR_HEAD,
                    steps_per_epoch=len(full_loader), epochs=EPOCHS_S1)
for ep in range(1, EPOCHS_S1 + 1):
    loss, acc = run_epoch(model_final, full_loader, opt_f1, sch_f1, True, DEVICE)
    print(f'  Full S1 ep {ep}/{EPOCHS_S1} | loss {loss:.4f} | acc {acc:.3f}')

# Stage 2
model_final.unfreeze_backbone()
param_groups_f = [
    {'params': model_final.backbone.parameters(),   'lr': LR_BACKBONE},
    {'params': model_final.classifier.parameters(), 'lr': LR_HEAD},
]
opt_f2 = AdamW(param_groups_f, weight_decay=WEIGHT_DECAY)
sch_f2 = OneCycleLR(opt_f2, max_lr=[LR_BACKBONE, LR_HEAD],
                    steps_per_epoch=len(full_loader), epochs=EPOCHS_S2)
for ep in range(1, EPOCHS_S2 + 1):
    loss, acc = run_epoch(model_final, full_loader, opt_f2, sch_f2, True, DEVICE)
    if ep % 5 == 0:
        print(f'  Full S2 ep {ep}/{EPOCHS_S2} | loss {loss:.4f} | acc {acc:.3f}')

# Test embeddings + submission
test_filenames = sorted(pd.unique(
    test_df[['query_image', 'gallery_image']].values.ravel()))
test_ds = JaguarDataset(
    filenames=test_filenames, img_dir=TEST_DIR, labels=None, transform=tf_val)
model_final.eval()
test_embs = extract_embeddings(
    lambda x: model_final(x, return_embedding=True),
    test_ds, batch_size=BATCH_SIZE, device=DEVICE, desc='Test embeddings')
fname_to_idx = {f: i for i, f in enumerate(test_filenames)}
make_submission(
    test_df=test_df, test_embeddings=test_embs,
    filename_to_idx=fname_to_idx,
    output_path=OUT_DIR / f'submission_{VERSION}.csv')

# Save final model weights
torch.save(model_final.state_dict(), OUT_DIR / f'model_{VERSION}.pth')
print(f'  Model saved -> model_{VERSION}.pth')

print(f'\n{"="*60}')
print(f'  {VERSION} COMPLETE')
print(f'  Val fold mAP  : {val_results["map"]:.4f}')
print(f'  Full train mAP: {full_results["map"]:.4f}')
print(f'  Outputs in    : {OUT_DIR}')
print(f'{"="*60}')
print('\nDownload /kaggle/working/output/ zip and push figures + results to GitHub.')
