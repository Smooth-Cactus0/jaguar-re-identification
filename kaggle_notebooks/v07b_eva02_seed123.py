"""
v07b: EVA-02 Large @ 448px — Model B (seed=123, different augmentation)
=======================================================================
Phase 1 of the v07 ensemble pipeline. Same architecture as v07a but with:
  - seed=123 (different weight init, data order)
  - Milder affine (10°, 0.1 translate, 0.9-1.1 scale)
  - Stronger color jitter (0.3, 0.3)
  - Added RandomAdjustSharpness (p=0.3)
These differences decorrelate error patterns for ensemble diversity.

Outputs saved to /kaggle/working/output/:
  - model_v07b.pth, embeddings_v07b_{train,test}.npy, filenames_v07b_{train,test}.json
  - submission_v07b.csv (standalone)
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

VERSION = 'v07b'
SEED = 123

MODEL_NAME = 'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k'
IMG_SIZE = 448
NUM_CLASSES = 31

NUM_EPOCHS = 20
BATCH_SIZE = 4
GRAD_ACCUM = 4
LR = 2e-5
WEIGHT_DECAY = 1e-3
EMA_DECAY = 0.999

ARCFACE_S = 30.0
ARCFACE_M = 0.5

USE_TTA = True
USE_QE = True
QE_TOP_K = 3
USE_RERANK = True
RERANK_K1 = 20
RERANK_K2 = 6
RERANK_LAMBDA = 0.3

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
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.is_test = is_test
        if not is_test:
            unique_ids = sorted(df['ground_truth'].unique())
            self.label_map = {name: i for i, name in enumerate(unique_ids)}
            self.df['label'] = self.df['ground_truth'].map(self.label_map)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['filename']
        img_path = self.img_dir / img_name
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
        if self.transform:
            img = self.transform(img)
        if self.is_test:
            return img, img_name
        return img, torch.tensor(row['label'], dtype=torch.long)

# ── Transforms (Model B: milder affine, stronger color, + sharpness) ────

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
    transforms.RandomErasing(p=0.25),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

# ── Model ────────────────────────────────────────────────────────────────

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
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
        self.feat_dim = self.backbone.num_features
        self.gem = GeM()
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.head = ArcFaceLayer(self.feat_dim, NUM_CLASSES, s=ARCFACE_S, m=ARCFACE_M)

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


# ── Training ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scaler, ema):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for i, (imgs, labels) in enumerate(tqdm(loader, leave=False, desc='Train')):
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        with torch.amp.autocast(DEVICE_TYPE):
            logits = model(imgs, labels)
            loss = criterion(logits, labels)
            loss = loss / GRAD_ACCUM
        scaler.scale(loss).backward()
        if (i + 1) % GRAD_ACCUM == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema.update(model)
        total_loss += loss.item() * GRAD_ACCUM
    if (i + 1) % GRAD_ACCUM != 0:
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


@torch.no_grad()
def extract_train_features(model, filenames, img_dir, transform, batch_size=4):
    model.eval()
    df = pd.DataFrame({'filename': filenames})
    dataset = JaguarDataset(df, img_dir, transform, is_test=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    return extract_features(model, loader, use_tta=USE_TTA)


# ── Post-processing ─────────────────────────────────────────────────────

def query_expansion(emb, top_k=3):
    print(f'  Applying Query Expansion (top_k={top_k})...')
    sims = emb @ emb.T
    indices = np.argsort(-sims, axis=1)[:, :top_k]
    new_emb = np.zeros_like(emb)
    for i in range(len(emb)):
        new_emb[i] = np.mean(emb[indices[i]], axis=0)
    return new_emb / np.linalg.norm(new_emb, axis=1, keepdims=True)


def k_reciprocal_rerank(prob, k1=20, k2=6, lambda_value=0.3):
    print(f'  Applying k-reciprocal reranking (k1={k1}, k2={k2}, λ={lambda_value})...')
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


def generate_submission(test_df, emb, img_map, output_path):
    if USE_QE:
        emb = query_expansion(emb, top_k=QE_TOP_K)
    sim_matrix = emb @ emb.T
    if USE_RERANK:
        sim_matrix = k_reciprocal_rerank(
            sim_matrix, k1=RERANK_K1, k2=RERANK_K2, lambda_value=RERANK_LAMBDA)
    preds = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc='Mapping'):
        s = sim_matrix[img_map[row['query_image']], img_map[row['gallery_image']]]
        preds.append(max(0.0, min(1.0, float(s))))
    sub = pd.DataFrame({'row_id': test_df['row_id'], 'similarity': preds})
    sub.to_csv(output_path, index=False)
    print(f'  Submission saved → {output_path}')
    print(f'  Score range: {np.min(preds):.4f} – {np.max(preds):.4f}')
    print(f'  Score mean:  {np.mean(preds):.4f}')
    return sub


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t0 = time.time()

    train_df = pd.read_csv(KAGGLE_INPUT / 'train.csv')
    test_df = pd.read_csv(KAGGLE_INPUT / 'test.csv')
    print(f'Train: {len(train_df)} images, {train_df["ground_truth"].nunique()} classes')

    train_dataset = JaguarDataset(train_df, TRAIN_DIR, train_transform)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True)

    unique_test = sorted(set(test_df['query_image']) | set(test_df['gallery_image']))
    test_dataset = JaguarDataset(
        pd.DataFrame({'filename': unique_test}), TEST_DIR, test_transform, is_test=True)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True)

    model = EVAReIDModel().to(DEVICE)
    if hasattr(model.backbone, 'set_grad_checkpointing'):
        model.backbone.set_grad_checkpointing(True)
        print('Gradient checkpointing: ENABLED')

    print(f'Model: {MODEL_NAME} | Seed: {SEED}')
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = torch.amp.GradScaler(DEVICE_TYPE)
    ema = EMAModel(model, decay=EMA_DECAY)
    criterion = nn.CrossEntropyLoss()

    print(f'\n{"="*60}')
    print(f'Training {VERSION}: {MODEL_NAME} @ {IMG_SIZE}px (seed={SEED})')
    print(f'{"="*60}\n')

    for epoch in range(NUM_EPOCHS):
        epoch_t0 = time.time()
        loss = train_epoch(model, train_loader, optimizer, criterion, scaler, ema)
        scheduler.step()
        print(f'Epoch {epoch+1:2d}/{NUM_EPOCHS} | '
              f'Loss: {loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | '
              f'Time: {time.time()-epoch_t0:.0f}s')

    print(f'\nTraining complete in {(time.time()-t0)/60:.1f} min')

    model_path = OUT_DIR / f'model_{VERSION}.pth'
    torch.save(ema.state_dict(), model_path)
    print(f'EMA weights saved → {model_path}')

    ema_model = ema.shadow
    print('\nExtracting test embeddings...')
    test_emb, test_names = extract_features(ema_model, test_loader, use_tta=USE_TTA)
    print(f'  Test: {test_emb.shape}')

    print('Extracting train embeddings...')
    train_emb, train_names = extract_train_features(
        ema_model, train_df['filename'].tolist(), TRAIN_DIR, test_transform, BATCH_SIZE)
    print(f'  Train: {train_emb.shape}')

    np.save(OUT_DIR / f'embeddings_{VERSION}_test.npy', test_emb)
    np.save(OUT_DIR / f'embeddings_{VERSION}_train.npy', train_emb)
    with open(OUT_DIR / f'filenames_{VERSION}_test.json', 'w') as f:
        json.dump(test_names, f)
    with open(OUT_DIR / f'filenames_{VERSION}_train.json', 'w') as f:
        json.dump(train_names, f)

    print('\nGenerating standalone submission...')
    img_map = {n: i for i, n in enumerate(test_names)}
    generate_submission(test_df, test_emb, img_map, OUT_DIR / f'submission_{VERSION}.csv')
    generate_submission(test_df, test_emb, img_map, Path('/kaggle/working/submission.csv'))

    print(f'\n{"="*60}')
    print(f'{VERSION} complete! Total: {(time.time()-t0)/60:.1f} min')
    print(f'{"="*60}')
