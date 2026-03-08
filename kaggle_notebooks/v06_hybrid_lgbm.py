"""
v06: Hybrid — Deep Embeddings + Hand-crafted Features → LightGBM
=================================================================
Best backbone from v04: ConvNeXt-Small (LB=0.797).
v05 heavy augmentation regressed slightly (LB=0.795), so this version
reverts to LIGHT augmentation for the deep model.

Key addition over v05:
  Instead of using cosine similarity on raw embeddings, we:

  1. Extract 512-dim L2-normalised embeddings (TTA x5) for all images.
  2. Extract hand-crafted image features (RGB histograms, foreground stats).
  3. Build pairwise feature vectors:
       [cosine_sim, l2_dist, |emb_i - emb_j| (512), hc_cosine, hc_l2, |hc_i - hc_j| (55)]
     = 571 features per pair.
  4. Train a LightGBM binary classifier: same identity (1) vs different (0).
  5. Use LightGBM probability as the final similarity score.

Why this helps:
  - LightGBM can weight embedding dimensions differently (not all 512 dims
    are equally discriminative for jaguar spot patterns).
  - Hand-crafted features (coat color tone, foreground coverage) are
    orthogonal to what the CNN learns via ArcFace.
  - Learning a non-linear similarity threshold can outperform cosine.

Hand-crafted features (55 dims per image):
  - RGB channel histograms: 16 bins × 3 = 48 dims (L1-normalised density)
  - Foreground coverage: fraction of non-white pixels (1 dim)
  - Foreground mean RGB: 3 dims (normalised 0-1)
  - Foreground std RGB:  3 dims (normalised 0-1)
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
import lightgbm as lgb
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
                            IMAGENET_MEAN, IMAGENET_STD, load_rgba_as_rgb,
                            pad_to_square)
from src.models    import EmbeddingModel
from src.losses    import ArcFaceLoss
from src.inference import make_submission
from src.evaluate  import average_precision, save_benchmark

# -- Paths ---------------------------------------------------------------------
KAGGLE_INPUT = Path('/kaggle/input/competitions/jaguar-re-id')
TRAIN_DIR    = KAGGLE_INPUT / 'train' / 'train'
TEST_DIR     = KAGGLE_INPUT / 'test'  / 'test'
OUT_DIR      = Path('/kaggle/working/output')
OUT_DIR.mkdir(exist_ok=True)

# -- Config --------------------------------------------------------------------
BACKBONE       = 'convnext_small'
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
TTA_VIEWS      = 5
NEG_RATIO      = 5      # negatives per positive pair for LightGBM training
DEVICE         = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_AMP        = DEVICE == 'cuda'
VERSION        = 'v06'
CV_FOLD        = 0
HC_DIM         = 55     # hand-crafted feature dimensionality
WHITE_THRESH   = 240    # pixel value above which all channels = background

print('\n' + '='*60)
print(f'  {VERSION}: Hybrid Deep + LightGBM')
print(f'  Backbone   : {BACKBONE}')
print(f'  ArcFace    : m={ARC_MARGIN}, s={ARC_SCALE}')
print(f'  Aug level  : light (reverted from v05 heavy)')
print(f'  TTA views  : {TTA_VIEWS}')
print(f'  LightGBM   : pairwise features ({EMBEDDING_DIM + HC_DIM + 4}-dim)')
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

print(f'  Train fold: {len(train_idx)} images, Val fold: {len(val_idx)} images')
print(f'  Train identities: {len(np.unique(labels[train_idx]))} | '
      f'Val identities: {len(np.unique(labels[val_idx]))}')

# -- Pre-cache images ----------------------------------------------------------
print('\n[2] Pre-caching images...')
all_train_files  = train_df['filename'].tolist()
test_files_flat  = sorted(pd.unique(test_df[['query_image', 'gallery_image']].values.ravel()))
train_cache = build_image_cache(all_train_files, TRAIN_DIR, img_size=IMG_SIZE, verbose=True)
test_cache  = build_image_cache(test_files_flat,  TEST_DIR,  img_size=IMG_SIZE, verbose=True)

# Light augmentation for training (v04 level — heavy aug hurt in v05)
tf_train = get_transforms(mode='train', img_size=IMG_SIZE, augment_level='light', cached=True)
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


# ===========================================================================
# Hand-crafted feature extraction
# ===========================================================================

def extract_hc_features(filenames, img_dir, cache, verbose=True):
    """
    Extract hand-crafted per-image features (55-dim):

      - RGB channel histograms: 16 bins × 3 channels = 48 dims
        L1-normalised so they represent probability densities, not raw counts.
        Captures the jaguar's overall coat tone (spotted jaguars range from
        cream-base to darker melanistic individuals).

      - Foreground coverage: 1 dim
        Fraction of pixels where not all channels exceed WHITE_THRESH (240).
        Captures how much of the frame the jaguar occupies.

      - Foreground mean RGB: 3 dims (normalised to 0-1)
        Average colour of the jaguar's visible coat.

      - Foreground std RGB: 3 dims (normalised to 0-1)
        Colour variability — high std indicates strong contrast between
        spots and base coat; low std = more uniform (melanistic jaguars).

    All features are computed on the cached 256×256 PIL images for speed.
    """
    features = []
    for i, fname in enumerate(filenames):
        if fname in cache:
            img = cache[fname]
        else:
            img = load_rgba_as_rgb(img_dir / fname)
            img = pad_to_square(img)

        arr = np.array(img, dtype=np.uint8)          # (H, W, 3)

        # RGB histograms (density)
        feats = []
        for ch in range(3):
            hist, _ = np.histogram(arr[:, :, ch], bins=16, range=(0, 256))
            hist = hist.astype(np.float32)
            hist /= (hist.sum() + 1e-8)
            feats.append(hist)

        # Foreground mask: non-white pixels
        fg_mask = ~(
            (arr[:, :, 0] > WHITE_THRESH) &
            (arr[:, :, 1] > WHITE_THRESH) &
            (arr[:, :, 2] > WHITE_THRESH)
        )
        coverage = np.float32(fg_mask.mean())
        feats.append([coverage])

        if fg_mask.any():
            fg_pix = arr[fg_mask].astype(np.float32)   # (N_fg, 3)
            fg_mean = fg_pix.mean(axis=0) / 255.0       # (3,)
            fg_std  = fg_pix.std(axis=0) / 255.0        # (3,)
        else:
            fg_mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            fg_std  = np.zeros(3, dtype=np.float32)

        feats.append(fg_mean)
        feats.append(fg_std)

        features.append(np.concatenate(feats))

        if verbose and (i + 1) % 500 == 0:
            print(f'    HC features: {i+1}/{len(filenames)}')

    return np.array(features, dtype=np.float32)   # (N, 55)


# ===========================================================================
# Pairwise feature construction
# ===========================================================================

def build_pair_features_batch(embeddings, hc_features, idx_i, idx_j):
    """
    Vectorised pairwise feature construction.

    For pairs (i, j) with indices idx_i, idx_j:

      cosine_sim  (1):  dot product of L2-normalised embeddings
      l2_dist     (1):  Euclidean distance of embeddings
      abs_diff_emb (512): element-wise absolute difference of embeddings
                          — most informative feature for LightGBM to learn
                          which embedding dimensions are most discriminative
      hc_cosine   (1):  cosine similarity of L1-normalised colour histograms
      hc_l2       (1):  L2 distance of hand-crafted features
      hc_abs_diff (55): element-wise absolute difference of HC features

    Total: 571 features per pair.

    Args:
        embeddings:  (N, 512) float32, L2-normalised
        hc_features: (N, 55) float32
        idx_i, idx_j: integer arrays of pair indices

    Returns:
        (len(idx_i), 571) float32 numpy array
    """
    idx_i = np.asarray(idx_i, dtype=np.int64)
    idx_j = np.asarray(idx_j, dtype=np.int64)

    e_i = embeddings[idx_i]       # (M, 512)
    e_j = embeddings[idx_j]       # (M, 512)
    h_i = hc_features[idx_i]      # (M, 55)
    h_j = hc_features[idx_j]      # (M, 55)

    cosine      = (e_i * e_j).sum(axis=1, keepdims=True)             # (M, 1)
    l2_emb      = np.linalg.norm(e_i - e_j, axis=1, keepdims=True)   # (M, 1)
    abs_diff_emb = np.abs(e_i - e_j)                                  # (M, 512)

    h_i_n = h_i / (np.linalg.norm(h_i, axis=1, keepdims=True) + 1e-8)
    h_j_n = h_j / (np.linalg.norm(h_j, axis=1, keepdims=True) + 1e-8)
    hc_cosine   = (h_i_n * h_j_n).sum(axis=1, keepdims=True)         # (M, 1)
    hc_l2       = np.linalg.norm(h_i - h_j, axis=1, keepdims=True)   # (M, 1)
    hc_abs_diff = np.abs(h_i - h_j)                                   # (M, 55)

    return np.concatenate(
        [cosine, l2_emb, abs_diff_emb, hc_cosine, hc_l2, hc_abs_diff],
        axis=1, dtype=np.float32
    )   # (M, 571)


def sample_train_pairs(labels, subset_idx=None, neg_ratio=5, seed=42):
    """
    Build positive + sampled negative pairs for LightGBM training.

    Args:
        labels:     (N,) integer identity labels for the full train set
        subset_idx: if given, restrict to images with these indices
                    (e.g., train fold only for unbiased CV)
        neg_ratio:  number of negatives per positive

    Returns:
        (idx_i, idx_j, y) arrays of pair indices and binary labels
    """
    rng = np.random.default_rng(seed)

    if subset_idx is not None:
        idxs = np.asarray(subset_idx)
        lbl  = labels[idxs]
    else:
        idxs = np.arange(len(labels))
        lbl  = labels

    # All positive pairs (same identity) within subset
    pos_pairs = []
    for unique_lbl in np.unique(lbl):
        members = idxs[lbl == unique_lbl]
        for ii in range(len(members)):
            for jj in range(ii + 1, len(members)):
                pos_pairs.append((members[ii], members[jj]))

    pos_i = np.array([p[0] for p in pos_pairs], dtype=np.int64)
    pos_j = np.array([p[1] for p in pos_pairs], dtype=np.int64)
    n_pos = len(pos_i)
    n_neg = n_pos * neg_ratio
    print(f'  Positive pairs: {n_pos:,} | Negative target: {n_neg:,}')

    # Sample negatives (different identity)
    N = len(idxs)
    neg_i_list, neg_j_list = [], []
    while len(neg_i_list) < n_neg:
        batch_a = rng.integers(0, N, size=n_neg * 3)
        batch_b = rng.integers(0, N, size=n_neg * 3)
        real_a  = idxs[batch_a]
        real_b  = idxs[batch_b]
        mask    = (labels[real_a] != labels[real_b]) & (real_a != real_b)
        neg_i_list.extend(real_a[mask].tolist())
        neg_j_list.extend(real_b[mask].tolist())

    neg_i = np.array(neg_i_list[:n_neg], dtype=np.int64)
    neg_j = np.array(neg_j_list[:n_neg], dtype=np.int64)

    all_i = np.concatenate([pos_i, neg_i])
    all_j = np.concatenate([pos_j, neg_j])
    all_y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

    # Shuffle
    perm   = rng.permutation(len(all_i))
    return all_i[perm], all_j[perm], all_y[perm]


# ===========================================================================
# mAP from LightGBM similarity matrix
# ===========================================================================

def compute_map_from_lgbm(lgbm_model, embeddings, hc_features, labels,
                           idx_subset, batch_size=4096):
    """
    Compute identity-balanced mAP using LightGBM similarity scores.

    For each image in idx_subset as query, rank all other subset images by
    LightGBM predicted probability (same identity). Compute AP per query,
    then macro-average per identity.

    Args:
        lgbm_model:  trained LightGBM Booster
        embeddings:  (N_all, 512)
        hc_features: (N_all, 55)
        labels:      (N_all,) integer identity labels
        idx_subset:  indices of images to evaluate on
        batch_size:  chunk size for pair feature computation

    Returns:
        dict with 'map', 'per_identity_ap', 'per_image_ap', 'sim_matrix'
    """
    sub_emb = embeddings[idx_subset]       # (M, 512)
    sub_hc  = hc_features[idx_subset]      # (M, 55)
    sub_lbl = labels[idx_subset]           # (M,)
    M = len(idx_subset)

    # Compute all pairwise LightGBM scores (excluding diagonal)
    # We use the full NxN matrix and zero the diagonal
    print(f'  Computing LightGBM similarity matrix ({M}×{M} = {M*M:,} pairs)...')
    sim_matrix = np.zeros((M, M), dtype=np.float32)

    # Build all off-diagonal pairs in batches
    pair_i_all = []
    pair_j_all = []
    for i in range(M):
        for j in range(M):
            if i != j:
                pair_i_all.append(i)
                pair_j_all.append(j)
    pair_i_all = np.array(pair_i_all, dtype=np.int64)
    pair_j_all = np.array(pair_j_all, dtype=np.int64)

    scores_all = np.zeros(len(pair_i_all), dtype=np.float32)
    for start in range(0, len(pair_i_all), batch_size):
        end = start + batch_size
        pi  = pair_i_all[start:end]
        pj  = pair_j_all[start:end]
        X_batch = build_pair_features_batch(sub_emb, sub_hc, pi, pj)
        scores_all[start:end] = lgbm_model.predict(X_batch)

    for k, (i, j) in enumerate(zip(pair_i_all, pair_j_all)):
        sim_matrix[i, j] = scores_all[k]

    # Compute mAP
    per_image_ap = np.zeros(M)
    for i in range(M):
        gallery_mask    = np.ones(M, dtype=bool)
        gallery_mask[i] = False
        per_image_ap[i] = average_precision(
            sim_matrix[i][gallery_mask],
            sub_lbl[gallery_mask],
            sub_lbl[i]
        )

    unique_labels   = np.unique(sub_lbl)
    per_identity_ap = {}
    for lbl in unique_labels:
        mask = sub_lbl == lbl
        per_identity_ap[int(lbl)] = float(per_image_ap[mask].mean())

    return {
        'map':              float(np.mean(list(per_identity_ap.values()))),
        'per_identity_ap':  per_identity_ap,
        'per_image_ap':     per_image_ap,
        'sim_matrix':       sim_matrix,
    }


# ===========================================================================
# Deep model
# ===========================================================================

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


def get_tta_transforms(img_size: int):
    norm = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    pad  = img_size + 32
    off  = 32
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
    tta_tfs  = get_tta_transforms(img_size)
    model.eval()
    all_embs = []
    for tf in tta_tfs:
        ds     = JaguarDataset(filenames=filenames, img_dir=img_dir,
                               labels=None, transform=tf, image_cache=cache)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
        view_embs = []
        with torch.no_grad():
            for imgs in loader:
                imgs = imgs.to(device)
                with autocast(enabled=USE_AMP):
                    view_embs.append(model(imgs).cpu().numpy())
        all_embs.append(np.concatenate(view_embs, axis=0))
    stacked = np.stack(all_embs, axis=0)       # (V, N, D)
    avg     = stacked.mean(axis=0)             # (N, D)
    norms   = np.linalg.norm(avg, axis=1, keepdims=True).clip(min=1e-8)
    return avg / norms


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


# ===========================================================================
# [3] Build model and train (CV fold)
# ===========================================================================
print('\n[3] Building model (CV fold)...')
model   = ArcFaceModel(BACKBONE, EMBEDDING_DIM).to(DEVICE)
arcface = ArcFaceLoss(EMBEDDING_DIM, N_CLASSES, ARC_MARGIN, ARC_SCALE).to(DEVICE)
scaler  = GradScaler(enabled=USE_AMP)
print(f'  Backbone dim: {model.backbone.out_dim} | '
      f'Total params: {sum(p.numel() for p in model.parameters()):,}')

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
        val_embs = []
        model.eval()
        with torch.no_grad():
            for imgs, _ in val_loader:
                with autocast(enabled=USE_AMP):
                    val_embs.append(model(imgs.to(DEVICE)).cpu().numpy())
        val_embs    = np.concatenate(val_embs, axis=0)
        val_cosine  = float((val_embs @ val_embs.T).mean())  # proxy
        # Quick mAP from cosine
        from src.evaluate import compute_map
        res_quick = compute_map(val_embs, labels[val_idx])
        vm = res_quick['map']
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


# ===========================================================================
# [6] Extract embeddings and hand-crafted features (CV model)
# ===========================================================================
print('\n[6] Extracting embeddings (TTA x{}) for all train images...'.format(TTA_VIEWS))
t0 = time.time()
train_embs = extract_embeddings_tta(
    model, all_train_files, TRAIN_DIR, train_cache, IMG_SIZE, BATCH_SIZE, DEVICE)
print(f'  Done in {time.time()-t0:.0f}s | shape: {train_embs.shape}')

print('\n[7] Extracting hand-crafted features for all train images...')
t0 = time.time()
train_hc = extract_hc_features(all_train_files, TRAIN_DIR, train_cache, verbose=True)
print(f'  Done in {time.time()-t0:.0f}s | shape: {train_hc.shape}')
assert train_hc.shape[1] == HC_DIM, f'Expected {HC_DIM} HC dims, got {train_hc.shape[1]}'

# Cosine-only baseline for comparison
from src.evaluate import compute_map
res_cosine_val = compute_map(train_embs[val_idx], labels[val_idx])
print(f'\n  Cosine-only val mAP (baseline for LightGBM comparison): {res_cosine_val["map"]:.4f}')


# ===========================================================================
# [8] Build pairwise training dataset and train LightGBM
# ===========================================================================
print('\n[8] Building LightGBM training pairs (from train fold only)...')
t0 = time.time()
pair_i, pair_j, pair_y = sample_train_pairs(
    labels, subset_idx=train_idx, neg_ratio=NEG_RATIO, seed=42)
print(f'  Total pairs: {len(pair_y):,} | '
      f'pos: {int(pair_y.sum()):,} | neg: {int((1-pair_y).sum()):,}')

print('  Building feature matrix...')
X_train = build_pair_features_batch(train_embs, train_hc, pair_i, pair_j)
y_train = pair_y.astype(np.float32)
print(f'  X_train shape: {X_train.shape} | '
      f'RAM: ~{X_train.nbytes / 1e6:.0f} MB | {time.time()-t0:.0f}s')

# Build val pairs for LightGBM early stopping
# Use all pairwise val fold combinations (not sampled — exact mAP evaluation)
print('  Building val pairs for early stopping...')
val_pair_i, val_pair_j, val_pair_y = sample_train_pairs(
    labels, subset_idx=val_idx, neg_ratio=NEG_RATIO, seed=99)
X_val = build_pair_features_batch(train_embs, train_hc, val_pair_i, val_pair_j)
y_val = val_pair_y.astype(np.float32)
print(f'  X_val shape: {X_val.shape}')

print('\n[9] Training LightGBM...')
t0 = time.time()

lgbm_params = {
    'objective':        'binary',
    'metric':           'auc',          # monitor AUC; mAP computed separately
    'learning_rate':    0.05,
    'num_leaves':       127,
    'min_child_samples': 20,
    'feature_fraction': 0.5,            # random feature subsampling per tree
    'bagging_fraction': 0.8,            # row subsampling
    'bagging_freq':     5,
    'reg_alpha':        0.1,
    'reg_lambda':       0.1,
    'verbosity':        -1,
    'n_jobs':           -1,
    'seed':             42,
}

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val   = lgb.Dataset(X_val,   label=y_val,   reference=lgb_train)

callbacks = [
    lgb.early_stopping(stopping_rounds=30, verbose=True),
    lgb.log_evaluation(period=50),
]

lgbm_model = lgb.train(
    lgbm_params,
    lgb_train,
    num_boost_round=500,
    valid_sets=[lgb_val],
    callbacks=callbacks,
)
print(f'  LightGBM trained in {time.time()-t0:.0f}s | '
      f'best iteration: {lgbm_model.best_iteration}')


# ===========================================================================
# [10] Evaluate: compare cosine-only vs LightGBM mAP on val fold
# ===========================================================================
print(f'\n[10] Evaluating LightGBM on val fold ({len(val_idx)} images)...')
res_lgbm_val = compute_map_from_lgbm(
    lgbm_model, train_embs, train_hc, labels, val_idx, batch_size=4096)

print(f'\n  {"Method":<25} {"Val mAP":>8}')
print(f'  {"-"*35}')
print(f'  {"Cosine-only baseline":<25} {res_cosine_val["map"]:>8.4f}')
print(f'  {"LightGBM (hybrid)":<25} {res_lgbm_val["map"]:>8.4f}')
lgbm_delta = res_lgbm_val['map'] - res_cosine_val['map']
print(f'  {"LightGBM delta":<25} {lgbm_delta:>+8.4f}')

# Per-identity AP comparison (best/worst)
from src.evaluate import print_results
print('\n  LightGBM per-identity results:')
print(f'  {"Identity":<20} {"Cosine AP":>10} {"LightGBM AP":>12} {"Delta":>8}')
print(f'  {"-"*52}')
for lbl in sorted(res_lgbm_val['per_identity_ap'].keys(),
                  key=lambda k: -res_lgbm_val['per_identity_ap'][k]):
    name     = idx_to_label[lbl]
    cos_ap   = res_cosine_val['per_identity_ap'].get(lbl, 0.0)
    lgbm_ap  = res_lgbm_val['per_identity_ap'][lbl]
    delta    = lgbm_ap - cos_ap
    print(f'  {name:<20} {cos_ap:>10.4f} {lgbm_ap:>12.4f} {delta:>+8.4f}')

best_lbl  = max(res_lgbm_val['per_identity_ap'], key=res_lgbm_val['per_identity_ap'].get)
worst_lbl = min(res_lgbm_val['per_identity_ap'], key=res_lgbm_val['per_identity_ap'].get)
print(f'\n  Best  : {idx_to_label[best_lbl]}  ({res_lgbm_val["per_identity_ap"][best_lbl]:.4f})')
print(f'  Worst : {idx_to_label[worst_lbl]} ({res_lgbm_val["per_identity_ap"][worst_lbl]:.4f})')

# Full train mAP for reference
full_loader_eval = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=2, pin_memory=True)
full_embs_cv = []
model.eval()
with torch.no_grad():
    for imgs, _ in full_loader_eval:
        with autocast(enabled=USE_AMP):
            full_embs_cv.append(model(imgs.to(DEVICE)).cpu().numpy())
full_embs_cv = np.concatenate(full_embs_cv, axis=0)
res_full     = compute_map(full_embs_cv, labels)
print(f'\n  Full train mAP (cosine, CV model): {res_full["map"]:.4f}')

save_benchmark(
    OUT_DIR / 'benchmarks_v06.csv',
    {
        'version':           VERSION,
        'backbone':          BACKBONE,
        'loss':              f'ArcFace (m={ARC_MARGIN}, s={ARC_SCALE}) + LightGBM',
        'embedding_dim':     EMBEDDING_DIM,
        'img_size':          IMG_SIZE,
        'augmentation':      'light',
        'cv_map_cosine':     round(res_cosine_val['map'], 5),
        'cv_map_lgbm':       round(res_lgbm_val['map'], 5),
        'cv_map_full':       round(res_full['map'], 5),
        'best_identity':     idx_to_label[best_lbl],
        'worst_identity':    idx_to_label[worst_lbl],
        'lgbm_delta':        round(lgbm_delta, 5),
        'lgbm_best_iter':    lgbm_model.best_iteration,
        'epochs_s1':         EPOCHS_S1,
        'epochs_s2':         EPOCHS_S2,
        'notes':             f'Light aug + TTA x{TTA_VIEWS} + LightGBM (571 pairwise features)',
    }
)


# ===========================================================================
# [11] Visualisations
# ===========================================================================
print('\n[11] Generating visualisations...')

# Feature importance (top-30 embedding dimensions + HC features)
feat_names = (
    ['cosine_sim', 'l2_dist'] +
    [f'emb_diff_{i}' for i in range(EMBEDDING_DIM)] +
    ['hc_cosine', 'hc_l2'] +
    [f'hc_diff_{i}' for i in range(HC_DIM)]
)
importances = lgbm_model.feature_importance(importance_type='gain')
top_idx     = np.argsort(importances)[::-1][:30]
top_names   = [feat_names[i] for i in top_idx]
top_vals    = importances[top_idx]

fig, ax = plt.subplots(figsize=(12, 5))
colors = ['steelblue' if 'emb' in n else
          ('coral' if 'hc' in n else 'mediumseagreen')
          for n in top_names]
ax.bar(range(30), top_vals / top_vals.sum(), color=colors, edgecolor='white')
ax.set_xticks(range(30))
ax.set_xticklabels(top_names, rotation=60, ha='right', fontsize=7)
ax.set_ylabel('Relative Gain')
ax.set_title('v06: Top-30 LightGBM Feature Importances', fontsize=12, fontweight='bold')
# Legend
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color='steelblue', label='Embedding diff dims'),
    Patch(color='coral',     label='HC colour/texture features'),
    Patch(color='mediumseagreen', label='Scalar features'),
], fontsize=8, loc='upper right')
plt.tight_layout()
fig.savefig(OUT_DIR / f'{VERSION}_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved -> v06_feature_importance.png')

# Per-identity AP: cosine vs LightGBM
lgbm_per_id  = {idx_to_label[k]: v for k, v in res_lgbm_val['per_identity_ap'].items()}
cos_per_id   = {idx_to_label[k]: v for k, v in res_cosine_val['per_identity_ap'].items()}
ids_sorted   = sorted(lgbm_per_id.keys(), key=lambda x: -lgbm_per_id[x])
x = np.arange(len(ids_sorted))
w = 0.38

fig, ax = plt.subplots(figsize=(16, 5))
ax.bar(x - w/2, [cos_per_id.get(i, 0) for i in ids_sorted],  w,
       color='lightsteelblue', label=f'Cosine ({res_cosine_val["map"]:.4f})', edgecolor='white')
ax.bar(x + w/2, [lgbm_per_id[i] for i in ids_sorted],        w,
       color='steelblue',      label=f'LightGBM ({res_lgbm_val["map"]:.4f})', edgecolor='white')
ax.axhline(res_lgbm_val['map'],  color='navy', linestyle='--', linewidth=1)
ax.axhline(res_cosine_val['map'], color='gray', linestyle=':', linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(ids_sorted, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Average Precision')
ax.set_ylim(0, 1)
ax.set_title('v06: Per-Identity AP — Cosine vs LightGBM', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(OUT_DIR / f'{VERSION}_per_identity_ap.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved -> v06_per_identity_ap.png')

# t-SNE of embeddings
print('  Computing t-SNE...')
from sklearn.manifold import TSNE
import seaborn as sns

tsne   = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, n_jobs=-1)
embs2d = tsne.fit_transform(train_embs)
palette = sns.color_palette('tab20', N_CLASSES) + \
          sns.color_palette('tab20b', max(0, N_CLASSES - 20))
fig, ax = plt.subplots(figsize=(12, 10))
for lbl in range(N_CLASSES):
    mask = labels == lbl
    ax.scatter(embs2d[mask, 0], embs2d[mask, 1], color=palette[lbl],
               label=idx_to_label[lbl], s=20, alpha=0.7, edgecolors='none')
ax.set_title(f'v06: t-SNE — ConvNeXt-Small + Light Aug', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=6, ncol=2, markerscale=1.5,
          framealpha=0.7, title='Identity')
ax.axis('off')
plt.tight_layout()
fig.savefig(OUT_DIR / f'{VERSION}_tsne.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved -> v06_tsne.png')


# ===========================================================================
# [12] Full retrain + final LightGBM + submission
# ===========================================================================
print('\n[12] Retraining on ALL data for submission...')

full_loader = DataLoader(full_train_ds, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=2, pin_memory=True)

model_final   = ArcFaceModel(BACKBONE, EMBEDDING_DIM).to(DEVICE)
arcface_final = ArcFaceLoss(EMBEDDING_DIM, N_CLASSES, ARC_MARGIN, ARC_SCALE).to(DEVICE)
scaler_final  = GradScaler(enabled=USE_AMP)

# Stage 1
model_final.freeze_backbone()
head_f  = list(filter(lambda p: p.requires_grad, model_final.parameters())) + \
          list(arcface_final.parameters())
opt_f1  = AdamW(head_f, lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
sch_f1  = OneCycleLR(opt_f1, max_lr=LR_HEAD,
                     steps_per_epoch=len(full_loader), epochs=EPOCHS_S1)
for ep in range(1, EPOCHS_S1 + 1):
    loss = run_epoch(model_final, arcface_final, full_loader,
                     opt_f1, sch_f1, True, DEVICE, scaler_final)
    print(f'  Full S1 ep {ep}/{EPOCHS_S1} | loss {loss:.4f}')

# Stage 2
model_final.unfreeze_backbone()
opt_f2  = AdamW([
    {'params': model_final.backbone.parameters(), 'lr': LR_BACKBONE},
    {'params': model_final.proj.parameters(),     'lr': LR_HEAD},
    {'params': arcface_final.parameters(),        'lr': LR_HEAD},
], weight_decay=WEIGHT_DECAY)
sch_f2  = OneCycleLR(opt_f2, max_lr=[LR_BACKBONE, LR_HEAD, LR_HEAD],
                     steps_per_epoch=len(full_loader), epochs=EPOCHS_S2)
for ep in range(1, EPOCHS_S2 + 1):
    loss = run_epoch(model_final, arcface_final, full_loader,
                     opt_f2, sch_f2, True, DEVICE, scaler_final)
    if ep % 5 == 0:
        print(f'  Full S2 ep {ep}/{EPOCHS_S2} | loss {loss:.4f}')

# Extract final embeddings
print(f'  Extracting final train embeddings (TTA x{TTA_VIEWS})...')
final_train_embs = extract_embeddings_tta(
    model_final, all_train_files, TRAIN_DIR, train_cache, IMG_SIZE, BATCH_SIZE, DEVICE)

print(f'  Extracting test embeddings (TTA x{TTA_VIEWS})...')
final_test_embs  = extract_embeddings_tta(
    model_final, test_files_flat, TEST_DIR, test_cache, IMG_SIZE, BATCH_SIZE, DEVICE)

# Extract HC features for test images
print('  Extracting hand-crafted features for test images...')
test_hc = extract_hc_features(test_files_flat, TEST_DIR, test_cache, verbose=True)

# Final train HC features (re-use train_hc — same images, ordering unchanged)
# Note: train_hc was computed on cached images above; those images are independent
# of model weights, so we can reuse them for the final LightGBM.

# Retrain LightGBM on ALL train pairs (all 31 identities)
print('\n  Training final LightGBM on all train pairs...')
pair_i_all, pair_j_all, pair_y_all = sample_train_pairs(
    labels, subset_idx=None, neg_ratio=NEG_RATIO, seed=42)
print(f'  Total pairs: {len(pair_y_all):,}')

X_full_train = build_pair_features_batch(
    final_train_embs, train_hc, pair_i_all, pair_j_all)
y_full_train = pair_y_all.astype(np.float32)

# Use best_iteration from CV to avoid refitting (prevents overfitting)
lgbm_final_params = {**lgbm_params}
lgbm_dataset_all  = lgb.Dataset(X_full_train, label=y_full_train)
lgbm_final = lgb.train(
    lgbm_final_params,
    lgbm_dataset_all,
    num_boost_round=lgbm_model.best_iteration,  # same depth as CV model
    callbacks=[lgb.log_evaluation(period=50)],
)
print(f'  Final LightGBM: {lgbm_model.best_iteration} iterations')

# Generate submission: predict LightGBM similarity for each test pair
print('\n  Generating submission with LightGBM...')
fname_to_idx_test = {f: i for i, f in enumerate(test_files_flat)}

# Build test pair features in chunks
n_test_pairs  = len(test_df)
chunk_size    = 8192
similarity    = np.zeros(n_test_pairs, dtype=np.float32)

for start in range(0, n_test_pairs, chunk_size):
    end    = min(start + chunk_size, n_test_pairs)
    q_idxs = test_df['query_image'].iloc[start:end].map(fname_to_idx_test).values
    g_idxs = test_df['gallery_image'].iloc[start:end].map(fname_to_idx_test).values
    X_chunk = build_pair_features_batch(
        final_test_embs, test_hc, q_idxs, g_idxs)
    similarity[start:end] = lgbm_final.predict(X_chunk)
    if (start // chunk_size) % 20 == 0:
        print(f'    {end}/{n_test_pairs} test pairs scored')

submission = test_df.copy()
submission['score'] = similarity
# Keep only the columns expected by the competition
output_cols = [c for c in ['id', 'query_image', 'gallery_image', 'score']
               if c in submission.columns]
submission[output_cols].to_csv(OUT_DIR / f'submission_{VERSION}.csv', index=False)
print(f'  Submission saved ({len(submission)} rows)')

torch.save(model_final.state_dict(), OUT_DIR / f'model_{VERSION}.pth')

# Also generate a cosine-based submission (backup, since cosine > LightGBM on val)
# make_submission uses cosine similarity and shifts to [0, 1]
make_submission(
    test_df=test_df, test_embeddings=final_test_embs,
    filename_to_idx=fname_to_idx_test,
    output_path=OUT_DIR / f'submission_{VERSION}_cosine.csv')
print('  Cosine submission also saved for comparison.')


# ===========================================================================
# Summary
# ===========================================================================
print(f'\n{"="*60}')
print(f'  {VERSION} COMPLETE')
print(f'  {"Method":<30} {"Val mAP":>8}')
print(f'  {"-"*40}')
print(f'  {"Cosine-only (v06 deep model)":<30} {res_cosine_val["map"]:>8.4f}')
print(f'  {"LightGBM (hybrid)":<30} {res_lgbm_val["map"]:>8.4f}')
print(f'  {"Delta":<30} {lgbm_delta:>+8.4f}')
print(f'  Full train mAP          : {res_full["map"]:.4f}')
print(f'  LightGBM best iteration : {lgbm_model.best_iteration}')
print(f'  Outputs in              : {OUT_DIR}')
print(f'{"="*60}')
print('\nDownload /kaggle/working/output/ zip and push figures + results to GitHub.')
