"""
v07d: Pseudo-Labeling — Ensemble-based self-training for all 3 models
=====================================================================
Phase 2 of the v07 ensemble pipeline.

Strategy:
  1. Load pre-trained weights + embeddings from v07a, v07b, v07f
  2. Average all 3 embeddings → compute cosine sim to class centroids
     (all models share 1024-dim space → clean average, no concat noise)
  3. Assign pseudo-labels where all 3 models agree AND confidence > 0.90
  4. Fine-tune each model for 5 epochs at lower LR on train + pseudo data
  5. Save updated weights + embeddings for the ensemble notebook (v07e)

Models:
  - v07a: EVA-02 Large (seed=42, 20 epochs)
  - v07b: EVA-02 Large (seed=123, 20 epochs)
  - v07f: DINOv2-Large-reg4 (self-supervised, 15 epochs)
  All produce 1024-dim embeddings → averaged for centroid matching.

Inputs (uploaded as Kaggle datasets from v07a/b/f outputs):
  - model_v07a.pth + embeddings + filenames
  - model_v07b.pth + embeddings + filenames
  - model_v07f.pth + embeddings + filenames

Outputs to /kaggle/working/output/:
  - model_v07a_pl.pth, model_v07b_pl.pth, model_v07f_pl.pth
  - embeddings_v07{a,b,f}_pl_{train,test}.npy
  - filenames_v07{a,b,f}_pl_{train,test}.json
  - pseudo_labels.csv (for analysis)
  - submission_v07d.csv (standalone from best single model after PL)
"""

# ── Install / imports ────────────────────────────────────────────────────
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-qU', 'timm'])

import os, math, random, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from tqdm.auto import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm

# ── Config ───────────────────────────────────────────────────────────────

VERSION = 'v07d'
SEED = 42

NUM_CLASSES = 31
PL_THRESHOLD = 0.90     # confidence threshold for pseudo-labeling
PL_MAX_ADD = 500         # safety cap

# Fine-tuning config (same for all 3 models)
PL_EPOCHS = 5
PL_LR = 1e-5            # lower LR for fine-tuning on pseudo data
WEIGHT_DECAY = 1e-3
EMA_DECAY = 0.999

ARCFACE_S = 30.0
ARCFACE_M = 0.5

USE_TTA = True

NORM_MEAN = [0.481, 0.457, 0.408]
NORM_STD = [0.268, 0.261, 0.275]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE_TYPE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Paths ────────────────────────────────────────────────────────────────

KAGGLE_INPUT = Path('/kaggle/input/competitions/jaguar-re-id')
TRAIN_DIR = KAGGLE_INPUT / 'train' / 'train'
TEST_DIR = KAGGLE_INPUT / 'test' / 'test'
OUT_DIR = Path('/kaggle/working/output')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Auto-detect base model output directories ────────────────────────────
# When you add a notebook's output as input on Kaggle, its files appear at:
#   /kaggle/input/<notebook-url-slug>/output/
# The slug is shown in the notebook URL, e.g. "v07a-eva02-seed42" → slug.
# We search /kaggle/input/ for the marker files so you don't need to hardcode.

def find_model_dir(version: str) -> Path:
    """Search /kaggle/input/ recursively for the directory containing model_{version}.pth."""
    search_root = Path('/kaggle/input')
    marker = f'model_{version}.pth'
    matches = list(search_root.rglob(marker))
    if matches:
        return matches[0].parent
    raise FileNotFoundError(
        f'Could not find {marker} under {search_root}. '
        f'Make sure the v07{version[-1]} notebook output is attached as an input dataset.'
    )

MODEL_A_DIR = find_model_dir('v07a')
MODEL_B_DIR = find_model_dir('v07b')
MODEL_F_DIR = find_model_dir('v07f')
print(f'v07a → {MODEL_A_DIR}')
print(f'v07b → {MODEL_B_DIR}')
print(f'v07f → {MODEL_F_DIR}')

# Model configs for reconstruction
# All 3 models output 1024-dim embeddings → clean averaging in ensemble
MODEL_CONFIGS = {
    'v07a': {
        'model_name': 'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',
        'img_size': 448, 'batch_size': 4, 'grad_accum': 4,
        'model_class': 'eva', 'input_dir': MODEL_A_DIR,
        'norm_mean': [0.481, 0.457, 0.408], 'norm_std': [0.268, 0.261, 0.275],
    },
    'v07b': {
        'model_name': 'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',
        'img_size': 448, 'batch_size': 4, 'grad_accum': 4,
        'model_class': 'eva', 'input_dir': MODEL_B_DIR,
        'norm_mean': [0.481, 0.457, 0.408], 'norm_std': [0.268, 0.261, 0.275],
    },
    'v07f': {
        'model_name': 'vit_large_patch14_reg4_dinov2.lvd142m',
        'img_size': 448, 'batch_size': 4, 'grad_accum': 4,
        'model_class': 'dinov2', 'input_dir': MODEL_F_DIR,
        # DINOv2 was trained with standard ImageNet normalization
        'norm_mean': [0.485, 0.456, 0.406], 'norm_std': [0.229, 0.224, 0.225],
    },
}

# ── Reproducibility ──────────────────────────────────────────────────────

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(SEED)

# ── Dataset ──────────────────────────────────────────────────────────────

class JaguarDataset(Dataset):
    """Dataset that supports images from different directories (train + test)."""
    def __init__(self, df, default_img_dir, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.default_img_dir = Path(default_img_dir)
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['filename']
        # Use per-row directory if available (for mixed train+pseudo data)
        if 'dir' in row and pd.notna(row['dir']):
            img_path = Path(row['dir']) / img_name
        else:
            img_path = self.default_img_dir / img_name
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (448, 448))
        if self.transform:
            img = self.transform(img)
        if self.is_test:
            return img, img_name
        return img, torch.tensor(row['label'], dtype=torch.long)


def get_transforms(img_size, mode='train', norm_mean=None, norm_std=None):
    """Build transforms for a given image size and mode.
    norm_mean/norm_std default to EVA jaguar stats; pass ImageNet stats for DINOv2."""
    mean = norm_mean if norm_mean is not None else NORM_MEAN
    std = norm_std if norm_std is not None else NORM_STD
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=15, translate=(0.15, 0.15),
                                    scale=(0.85, 1.15)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.25),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ── Model definitions ───────────────────────────────────────────────────

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)


class ArcFaceLayer(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label=None):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if label is None:
            return cosine
        phi = cosine - self.m
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s


class EVAReIDModel(nn.Module):
    def __init__(self, model_name, num_classes=31):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.feat_dim = self.backbone.num_features
        self.gem = GeM()
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.head = ArcFaceLayer(self.feat_dim, num_classes, s=ARCFACE_S, m=ARCFACE_M)

    def forward(self, x, label=None):
        features = self.backbone.forward_features(x)
        if features.dim() == 3:
            B, N, C = features.shape
            H = W = int(math.sqrt(N))
            if H * W != N:
                features = features[:, -H * W:, :]
            features = features.permute(0, 2, 1).reshape(B, C, H, W)
        emb = self.gem(features).flatten(1)
        emb = self.bn(emb)
        if label is not None:
            return self.head(emb, label)
        return emb


class DINOv2ReIDModel(nn.Module):
    """DINOv2-Large with register tokens.

    Strips CLS (index 0) and 4 register tokens (indices 1-4) before
    GeM pooling. At 448px with patch14: 32×32=1024 clean patch tokens."""

    NUM_REGISTER_TOKENS = 4

    def __init__(self, model_name, num_classes=31):
        super().__init__()
        # img_size=448 MUST match the resolution used during v07f training.
        # Without it, timm defaults to DINOv2's native 518px → 1369 pos_embed tokens,
        # which is incompatible with the 448px checkpoint (1024 tokens).
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=448)
        self.feat_dim = self.backbone.num_features
        self.gem = GeM()
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.head = ArcFaceLayer(self.feat_dim, num_classes, s=ARCFACE_S, m=ARCFACE_M)
        self.patch_size = 14
        self.grid_size = 448 // self.patch_size  # 32

    def forward(self, x, label=None):
        features = self.backbone.forward_features(x)
        if features.dim() == 3:
            B, total_tokens, C = features.shape
            num_prefix = 1 + self.NUM_REGISTER_TOKENS
            patch_tokens = features[:, num_prefix:, :]
            H = W = self.grid_size
            if patch_tokens.shape[1] != H * W:
                patch_tokens = patch_tokens[:, -H * W:, :]
            features = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)
        emb = self.gem(features).flatten(1)
        emb = self.bn(emb)
        if label is not None:
            return self.head(emb, label)
        return emb


def build_model(model_class, model_name, num_classes=31):
    """Factory: build model by class type."""
    if model_class == 'eva':
        return EVAReIDModel(model_name, num_classes)
    elif model_class == 'dinov2':
        return DINOv2ReIDModel(model_name, num_classes)
    else:
        raise ValueError(f'Unknown model class: {model_class}')


# ── EMA ──────────────────────────────────────────────────────────────────

class EMAModel:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)
        for ema_b, model_b in zip(self.shadow.buffers(), model.buffers()):
            ema_b.data.copy_(model_b.data)

    def state_dict(self):
        return self.shadow.state_dict()


# ── Pseudo-label generation ─────────────────────────────────────────────

def compute_class_centroids(train_emb, train_labels, num_classes=31):
    """Compute L2-normalised centroid for each class from training embeddings."""
    centroids = np.zeros((num_classes, train_emb.shape[1]), dtype=np.float32)
    for c in range(num_classes):
        mask = train_labels == c
        if mask.sum() > 0:
            centroids[c] = train_emb[mask].mean(axis=0)
    # L2 normalise
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    centroids = centroids / norms
    return centroids


def generate_pseudo_labels(ensemble_test_emb, ensemble_train_emb, train_labels,
                           test_filenames, threshold=0.90, max_add=500):
    """Generate pseudo-labels for test images using ensemble embeddings.

    Computes class centroids from training embeddings, then assigns each test
    image the class whose centroid has highest cosine similarity, if above threshold.
    """
    centroids = compute_class_centroids(ensemble_train_emb, train_labels)

    # Cosine similarity: test_emb @ centroids.T → (N_test, N_classes)
    sims = ensemble_test_emb @ centroids.T

    max_sims = sims.max(axis=1)
    pred_classes = sims.argmax(axis=1)

    pl_data = []
    for i in range(len(max_sims)):
        if max_sims[i] > threshold and len(pl_data) < max_add:
            pl_data.append({
                'filename': test_filenames[i],
                'label': int(pred_classes[i]),
                'confidence': float(max_sims[i]),
                'dir': str(TEST_DIR),
            })

    print(f'  Pseudo-labels: {len(pl_data)}/{len(test_filenames)} test images '
          f'above threshold {threshold}')
    if pl_data:
        confs = [d['confidence'] for d in pl_data]
        print(f'  Confidence range: {min(confs):.4f} – {max(confs):.4f}')

    return pd.DataFrame(pl_data)


# ── Training ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scaler, ema, grad_accum):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for i, (imgs, labels) in enumerate(tqdm(loader, leave=False, desc='PL Train')):
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        with torch.amp.autocast(DEVICE_TYPE):
            logits = model(imgs, labels)
            loss = criterion(logits, labels)
            loss = loss / grad_accum
        scaler.scale(loss).backward()
        if (i + 1) % grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema.update(model)
        total_loss += loss.item() * grad_accum
    if (i + 1) % grad_accum != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        ema.update(model)
    return total_loss / len(loader)


# ── Inference ────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(model, loader, use_tta=True):
    model.eval()
    all_feats, all_names = [], []
    for imgs, fnames in tqdm(loader, desc='Extracting'):
        imgs = imgs.to(DEVICE)
        with torch.amp.autocast(DEVICE_TYPE):
            f1 = model(imgs)
            if use_tta:
                f2 = model(torch.flip(imgs, [3]))
                f1 = (f1 + f2) / 2
        all_feats.append(F.normalize(f1, dim=1).float().cpu())
        all_names.extend(fnames)
    return torch.cat(all_feats, dim=0).numpy(), all_names


# ── Post-processing (for standalone submission) ──────────────────────────

def query_expansion(emb, top_k=3):
    sims = emb @ emb.T
    indices = np.argsort(-sims, axis=1)[:, :top_k]
    new_emb = np.zeros_like(emb)
    for i in range(len(emb)):
        new_emb[i] = np.mean(emb[indices[i]], axis=0)
    return new_emb / np.linalg.norm(new_emb, axis=1, keepdims=True)


def k_reciprocal_rerank(prob, k1=15, k2=6, lambda_value=0.4):
    q_g_dist = 1 - prob
    original_dist = q_g_dist.copy()
    initial_rank = np.argsort(original_dist, axis=1)
    nn_k1 = []
    for i in range(prob.shape[0]):
        forward_k1 = initial_rank[i, :k1 + 1]
        backward_k1 = initial_rank[forward_k1, :k1 + 1]
        fi = np.where(backward_k1 == i)[0]
        nn_k1.append(forward_k1[fi])
    jaccard_dist = np.zeros_like(original_dist)
    for i in range(prob.shape[0]):
        ind_non_zero = np.where(original_dist[i, :] < 0.6)[0]
        ind_images = [
            inv for inv in ind_non_zero
            if len(np.intersect1d(nn_k1[i], nn_k1[inv])) > 0
        ]
        for j in ind_images:
            intersection = len(np.intersect1d(nn_k1[i], nn_k1[j]))
            union = len(np.union1d(nn_k1[i], nn_k1[j]))
            jaccard_dist[i, j] = 1 - intersection / union
    return 1 - (jaccard_dist * lambda_value + original_dist * (1 - lambda_value))


# ══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t0 = time.time()

    train_df = pd.read_csv(KAGGLE_INPUT / 'train.csv')
    test_df = pd.read_csv(KAGGLE_INPUT / 'test.csv')

    # Encode labels
    unique_ids = sorted(train_df['ground_truth'].unique())
    label_map = {name: i for i, name in enumerate(unique_ids)}
    idx_to_label = {i: name for name, i in label_map.items()}
    train_df['label'] = train_df['ground_truth'].map(label_map)
    train_df['dir'] = str(TRAIN_DIR)

    print(f'Train: {len(train_df)} images, {len(unique_ids)} classes')
    print(f'Test pairs: {len(test_df)}')

    # ── Step 1: Load pre-computed embeddings from base models ────────────
    print('\n' + '='*60)
    print('STEP 1: Loading base model embeddings')
    print('='*60)

    all_test_embs = {}
    all_train_embs = {}
    test_filenames = None

    for version, cfg in MODEL_CONFIGS.items():
        input_dir = cfg['input_dir']
        print(f'\n  Loading {version} from {input_dir}...')

        test_emb = np.load(input_dir / f'embeddings_{version}_test.npy')
        train_emb = np.load(input_dir / f'embeddings_{version}_train.npy')
        with open(input_dir / f'filenames_{version}_test.json') as f:
            test_names = json.load(f)
        with open(input_dir / f'filenames_{version}_train.json') as f:
            train_names = json.load(f)

        all_test_embs[version] = test_emb
        all_train_embs[version] = train_emb
        print(f'    Test:  {test_emb.shape}')
        print(f'    Train: {train_emb.shape}')

        if test_filenames is None:
            test_filenames = test_names
            train_filenames = train_names

    # ── Step 2: Ensemble embeddings for pseudo-labeling ──────────────────
    print('\n' + '='*60)
    print('STEP 2: Ensembling embeddings for pseudo-label generation')
    print('='*60)

    # All 3 models share 1024-dim space → average all three for strongest signal
    def l2_norm(e):
        return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)

    ens_test = l2_norm(
        all_test_embs['v07a'] + all_test_embs['v07b'] + all_test_embs['v07f'])
    ens_train = l2_norm(
        all_train_embs['v07a'] + all_train_embs['v07b'] + all_train_embs['v07f'])
    print(f'  3-model average: test {ens_test.shape}, train {ens_train.shape}')

    # Per-model predictions for cross-validation
    per_model_test = {v: all_test_embs[v] for v in MODEL_CONFIGS}
    per_model_train = {v: all_train_embs[v] for v in MODEL_CONFIGS}

    # Get train labels in embedding order
    train_label_map = {fn: label_map[train_df.loc[train_df['filename'] == fn,
                       'ground_truth'].values[0]]
                       for fn in train_filenames}
    train_labels = np.array([train_label_map[fn] for fn in train_filenames])

    # ── Step 3: Generate pseudo-labels ───────────────────────────────────
    print('\n' + '='*60)
    print('STEP 3: Generating pseudo-labels (3-way agreement)')
    print('='*60)

    # Generate per-model pseudo-labels for cross-validation
    per_model_pls = {}
    for version in MODEL_CONFIGS:
        pl_df_v = generate_pseudo_labels(
            per_model_test[version], per_model_train[version],
            train_labels, test_filenames,
            threshold=PL_THRESHOLD, max_add=PL_MAX_ADD)
        per_model_pls[version] = dict(zip(pl_df_v['filename'], pl_df_v['label'])) \
            if len(pl_df_v) > 0 else {}
        print(f'  {version}: {len(per_model_pls[version])} above threshold')

    # Also run on 3-model average for highest-confidence candidates
    pl_df_ens = generate_pseudo_labels(
        ens_test, ens_train, train_labels, test_filenames,
        threshold=PL_THRESHOLD, max_add=PL_MAX_ADD)
    ens_pls = dict(zip(pl_df_ens['filename'], pl_df_ens['label'])) \
        if len(pl_df_ens) > 0 else {}

    # Keep only pseudo-labels where ALL 3 individual models agree
    pls_a, pls_b, pls_f = (per_model_pls['v07a'],
                            per_model_pls['v07b'],
                            per_model_pls['v07f'])
    agreed = []
    disagreed = 0
    for fn in ens_pls:
        label_a = pls_a.get(fn)
        label_b = pls_b.get(fn)
        label_f = pls_f.get(fn)
        if label_a is not None and label_b is not None and label_f is not None:
            if label_a == label_b == label_f:
                row = pl_df_ens[pl_df_ens['filename'] == fn].iloc[0].to_dict()
                agreed.append(row)
            else:
                disagreed += 1

    pl_df = pd.DataFrame(agreed) if agreed else pd.DataFrame()
    print(f'\n  3-way agreement: {len(agreed)} agreed, {disagreed} disagreed')

    # Save pseudo-labels for analysis
    if len(pl_df) > 0:
        pl_df['identity'] = pl_df['label'].map(idx_to_label)
        pl_df.to_csv(OUT_DIR / 'pseudo_labels.csv', index=False)
        print(f'  Saved pseudo_labels.csv ({len(pl_df)} entries)')

        # Distribution
        print(f'\n  Pseudo-label distribution:')
        for label_idx in sorted(pl_df['label'].unique()):
            count = (pl_df['label'] == label_idx).sum()
            name = idx_to_label[label_idx]
            print(f'    {name}: {count}')

    # ── Step 4: Fine-tune each model on train + pseudo-labels ────────────
    print('\n' + '='*60)
    print('STEP 4: Fine-tuning models with pseudo-labels')
    print('='*60)

    if len(pl_df) == 0:
        print('  No pseudo-labels generated! Skipping fine-tuning.')
        print('  Copying base model embeddings as-is.')
        for version in MODEL_CONFIGS:
            cfg = MODEL_CONFIGS[version]
            input_dir = cfg['input_dir']
            # Just copy embeddings with _pl suffix
            for suffix in ['test', 'train']:
                src = np.load(input_dir / f'embeddings_{version}_{suffix}.npy')
                np.save(OUT_DIR / f'embeddings_{version}_pl_{suffix}.npy', src)
            for suffix in ['test', 'train']:
                with open(input_dir / f'filenames_{version}_{suffix}.json') as f:
                    data = json.load(f)
                with open(OUT_DIR / f'filenames_{version}_pl_{suffix}.json', 'w') as fw:
                    json.dump(data, fw)
    else:
        # Prepare combined dataset
        combined_base = train_df[['filename', 'label', 'dir']].copy()
        pl_combined = pl_df[['filename', 'label', 'dir']].copy()
        combined_df = pd.concat([combined_base, pl_combined], ignore_index=True)
        print(f'  Combined dataset: {len(combined_df)} images '
              f'({len(combined_base)} train + {len(pl_combined)} pseudo)')

        unique_test_fnames = sorted(
            set(test_df['query_image']) | set(test_df['gallery_image']))

        for version, cfg in MODEL_CONFIGS.items():
            print(f'\n  {"─"*50}')
            print(f'  Fine-tuning {version}: {cfg["model_name"]}')
            print(f'  {"─"*50}')

            img_size = cfg['img_size']
            batch_size = cfg['batch_size']
            grad_accum = cfg['grad_accum']

            # Build model and load base weights
            model = build_model(cfg['model_class'], cfg['model_name']).to(DEVICE)
            if hasattr(model.backbone, 'set_grad_checkpointing'):
                model.backbone.set_grad_checkpointing(True)

            base_weights = torch.load(
                cfg['input_dir'] / f'model_{version}.pth',
                map_location=DEVICE)
            model.load_state_dict(base_weights)
            print(f'    Loaded base weights from {cfg["input_dir"]}')

            # Create dataloaders (use per-model normalization)
            norm_mean = cfg.get('norm_mean')
            norm_std = cfg.get('norm_std')
            train_transform = get_transforms(img_size, mode='train', norm_mean=norm_mean, norm_std=norm_std)
            test_transform = get_transforms(img_size, mode='test', norm_mean=norm_mean, norm_std=norm_std)

            train_dataset = JaguarDataset(combined_df, TRAIN_DIR, train_transform)
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=2, pin_memory=True, drop_last=True)

            test_dataset = JaguarDataset(
                pd.DataFrame({'filename': unique_test_fnames}),
                TEST_DIR, test_transform, is_test=True)
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False,
                num_workers=2, pin_memory=True)

            # Optimizer + EMA
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=PL_LR, weight_decay=WEIGHT_DECAY)
            scaler = torch.amp.GradScaler(DEVICE_TYPE)
            ema = EMAModel(model, decay=EMA_DECAY)
            criterion = nn.CrossEntropyLoss()

            # Fine-tune
            for epoch in range(PL_EPOCHS):
                loss = train_epoch(
                    model, train_loader, optimizer, criterion,
                    scaler, ema, grad_accum)
                print(f'    Epoch {epoch+1}/{PL_EPOCHS} | Loss: {loss:.4f}')

            # Save PL model weights
            model_path = OUT_DIR / f'model_{version}_pl.pth'
            torch.save(ema.state_dict(), model_path)
            print(f'    Saved → {model_path}')

            # Extract updated embeddings using EMA model
            ema_model = ema.shadow

            test_emb, test_names = extract_features(
                ema_model, test_loader, use_tta=USE_TTA)

            # Train embeddings
            train_emb_dataset = JaguarDataset(
                pd.DataFrame({'filename': train_df['filename'].tolist()}),
                TRAIN_DIR, test_transform, is_test=True)
            train_emb_loader = DataLoader(
                train_emb_dataset, batch_size=batch_size, shuffle=False,
                num_workers=2, pin_memory=True)
            train_emb, train_names = extract_features(
                ema_model, train_emb_loader, use_tta=USE_TTA)

            # Save
            np.save(OUT_DIR / f'embeddings_{version}_pl_test.npy', test_emb)
            np.save(OUT_DIR / f'embeddings_{version}_pl_train.npy', train_emb)
            with open(OUT_DIR / f'filenames_{version}_pl_test.json', 'w') as f:
                json.dump(test_names, f)
            with open(OUT_DIR / f'filenames_{version}_pl_train.json', 'w') as f:
                json.dump(train_names, f)
            print(f'    Embeddings: test {test_emb.shape}, train {train_emb.shape}')

            # Free GPU memory before next model
            del model, ema, optimizer, scaler
            torch.cuda.empty_cache()

    # ── Step 5: Generate standalone submission from best PL model ────────
    print('\n' + '='*60)
    print('STEP 5: Generating standalone submission (EVA-A PL)')
    print('='*60)

    # Use the v07a PL model as the standalone submission
    test_emb = np.load(OUT_DIR / 'embeddings_v07a_pl_test.npy')
    with open(OUT_DIR / 'filenames_v07a_pl_test.json') as f:
        test_names = json.load(f)
    img_map = {n: i for i, n in enumerate(test_names)}

    # QE + rerank
    emb = query_expansion(test_emb, top_k=3)
    sim_matrix = emb @ emb.T
    sim_matrix = k_reciprocal_rerank(sim_matrix)

    preds = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc='Mapping'):
        s = sim_matrix[img_map[row['query_image']], img_map[row['gallery_image']]]
        preds.append(max(0.0, min(1.0, float(s))))

    sub = pd.DataFrame({'row_id': test_df['row_id'], 'similarity': preds})
    sub.to_csv(OUT_DIR / f'submission_{VERSION}.csv', index=False)
    sub.to_csv(Path('/kaggle/working/submission.csv'), index=False)
    print(f'  Score range: {np.min(preds):.4f} – {np.max(preds):.4f}')
    print(f'  Score mean:  {np.mean(preds):.4f}')

    total_time = time.time() - t0
    print(f'\n{"="*60}')
    print(f'{VERSION} complete! Total: {total_time/60:.1f} min')
    print(f'{"="*60}')
