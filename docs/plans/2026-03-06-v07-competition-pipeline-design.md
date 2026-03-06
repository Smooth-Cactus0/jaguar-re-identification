# v07 Competition Pipeline Design — Targeting 0.97+ LB

**Date**: 2026-03-06
**Current best**: v06 LB 0.799 (ConvNeXt-Small 224px + ArcFace + cosine)
**Reference**: EVA-02 Large 448px + QE + rerank = 0.941; + pseudo-labeling = 0.941
**Target**: 0.97+ (top leaderboard positions)

## Architecture Overview

```
Phase 1: Train 3 diverse base models          (~2h GPU total)
  NB-A: EVA-02 Large @ 448px, seed=42         (~40 min)
  NB-B: EVA-02 Large @ 448px, seed=123        (~40 min)
  NB-C: ConvNeXt-Large @ 384px                (~30 min)
  Each saves: model weights + test embeddings + train embeddings

Phase 2: Pseudo-labeling round                 (~1.5h GPU total)
  NB-PL: Load all 3 base models
    -> Ensemble embeddings -> pseudo-label test (thresh=0.90)
    -> Retrain each model 5 epochs on train+pseudo
    -> Save updated weights + embeddings

Phase 3: Ensemble + post-processing            (~5 min, CPU-only)
  NB-ENS: Load all embeddings
    -> Concatenate + L2-normalise (3072-dim or 2048+1024)
    -> Database-side Augmentation (weighted QE)
    -> K-reciprocal reranking (tuned params)
    -> Generate submission
```

**Total GPU**: ~3.5h of 30h budget

## Phase 1: Base Model Training

### Model A & B — EVA-02 Large @ 448px

**Why EVA-02 Large**: 300M params, pretrained on ImageNet-22k with masked image
modeling (MIM). The 0.941 reference proves it works on this exact dataset.
patch14 at 448px = 32x32 = 1024 tokens, giving fine spatial resolution for
jaguar spot patterns.

```python
Config:
  model_name: "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
  img_size: 448
  embedding_dim: 1024 (native, no projection)
  batch_size: 4
  grad_accum: 4  (effective batch = 16)
  epochs: 10
  lr: 2e-5 (flat, AdamW)
  weight_decay: 1e-3
  scheduler: CosineAnnealingLR
  loss: ArcFace (m=0.5, s=30) via CrossEntropyLoss on logits
  pooling: GeM (learnable p=3)
  TTA: horizontal flip (average embeddings)
  gradient_checkpointing: True (saves ~40% VRAM)
```

**Model A vs B differences** (for ensemble diversity):
- Model A: seed=42, aug = RandomAffine(15, 0.15, 0.85-1.15) + ColorJitter(0.2, 0.2)
- Model B: seed=123, aug = RandomAffine(10, 0.1, 0.9-1.1) + ColorJitter(0.3, 0.3) + Sharpen(p=0.3)

### Model C — ConvNeXt-Large @ 384px

**Why ConvNeXt-Large**: CNN inductive bias (locality, translation equivariance)
is complementary to ViT's global attention. Different error patterns = better
ensemble. ConvNeXt-Small already proved effective in v04 (LB 0.797).

```python
Config:
  model_name: "convnext_large.fb_in22k_ft_in1k_384"
  img_size: 384
  embedding_dim: 1536 (native)
  batch_size: 8
  grad_accum: 2  (effective batch = 16)
  epochs: 15 (CNNs need more epochs to converge)
  lr: 5e-5 backbone, 1e-3 head (differential LR — proven in v04)
  pooling: GeM (learnable p=3)
  TTA: horizontal flip
```

### Shared Architecture for All Models

```python
class ReIDModel(nn.Module):
    backbone      -> timm (num_classes=0)
    gem           -> GeM(p=3, learnable)
    bn            -> BatchNorm1d(feat_dim)
    arcface_head  -> ArcFaceLayer(feat_dim, 31, s=30, m=0.5)

    forward(x, label=None):
        features = backbone.forward_features(x)
        # ViT: reshape (B, N, C) -> (B, C, H, W) for GeM
        # CNN: already (B, C, H, W)
        emb = bn(gem(features).flatten(1))
        if label: return arcface_head(emb, label)  # training
        return F.normalize(emb)                     # inference

class GeM(nn.Module):
    # Generalised Mean Pooling with learnable exponent p
    # p=1 = average pooling, p=inf = max pooling
    # p=3 (init) upweights high-activation regions (distinctive spots)
```

### EMA (Exponential Moving Average)

All models use EMA with decay=0.999:
```python
# After each optimizer step:
for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
    ema_p.data.mul_(decay).add_(model_p.data, alpha=1-decay)
# Use ema_model for inference
```

### Training Augmentation

```python
transforms.Compose([
    Resize((img_size, img_size)),
    RandomHorizontalFlip(p=0.5),
    RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.85, 1.15)),
    ColorJitter(brightness=0.2, contrast=0.2),
    ToTensor(),
    Normalize([0.481, 0.457, 0.408], [0.268, 0.261, 0.275]),
    RandomErasing(p=0.25),
])
```

## Phase 2: Pseudo-Labeling

### Strategy

1. Extract test embeddings from all 3 base models
2. Ensemble: average L2-normalised embeddings (or concat + renorm)
3. Compute cosine similarity to ArcFace class centroids
4. Assign pseudo-labels where max confidence > 0.90
5. Merge train (1895) + pseudo-labeled test (~350+) images
6. Fine-tune each model for 5 epochs at lower LR (1e-5)
7. Save updated embeddings

### Key Details

- The base EVA model already assigns 361/371 test images above 0.90 confidence
- With ensemble of 3 models, expect ~365+ to pass threshold
- Fine-tune with SAME augmentation as Phase 1 (no special treatment for pseudo)
- Only 5 epochs to avoid overfitting to potentially wrong pseudo-labels

## Phase 3: Ensemble + Post-Processing

### Embedding Fusion

```python
# Option A: Concatenation (preserves all information)
emb_fused = np.concatenate([emb_A, emb_B, emb_C], axis=1)  # (371, 3584)
emb_fused = emb_fused / np.linalg.norm(emb_fused, axis=1, keepdims=True)

# Option B: Average (simpler, works if embeddings are aligned)
emb_fused = (emb_A + emb_B + emb_C) / 3
emb_fused = emb_fused / np.linalg.norm(emb_fused, axis=1, keepdims=True)
```

Concatenation is generally better for diverse architectures (different feature
spaces). Average is better for same-architecture different-seed models.

**Recommended**: Average Models A+B (same arch), then concatenate with Model C:
```python
emb_eva = (emb_A + emb_B) / 2
emb_eva = emb_eva / np.linalg.norm(emb_eva, axis=1, keepdims=True)
emb_fused = np.concatenate([emb_eva, emb_C], axis=1)  # (371, 2560)
emb_fused = emb_fused / np.linalg.norm(emb_fused, axis=1, keepdims=True)
```

### Database-side Augmentation (DBA / Weighted QE)

```python
def dba_query_expansion(emb, top_k=5):
    """Weighted QE: each neighbour contributes proportionally to similarity."""
    sims = emb @ emb.T
    indices = np.argsort(-sims, axis=1)[:, :top_k]
    new_emb = np.zeros_like(emb)
    for i in range(len(emb)):
        weights = sims[i, indices[i]]
        weights = weights / weights.sum()
        new_emb[i] = (emb[indices[i]] * weights[:, None]).sum(axis=0)
    return new_emb / np.linalg.norm(new_emb, axis=1, keepdims=True)
```

### K-Reciprocal Reranking

Same algorithm as reference notebook (Zhong et al. 2017) but with tuned
parameters for our small 371-image gallery:

```
k1=15 (down from 20 — smaller gallery needs smaller k)
k2=6  (keep default)
lambda_value=0.4 (up from 0.3 — trust Jaccard more with small gallery)
```

### Submission Generation

```python
sim_matrix = emb @ emb.T  # after QE
sim_matrix = k_reciprocal_rerank(sim_matrix, k1=15, k2=6, lambda_value=0.4)
# Map to test pairs
for row in test_df:
    similarity = sim_matrix[q_idx, g_idx]
    similarity = max(0.0, min(1.0, similarity))
```

## Notebook Structure

```
kaggle_notebooks/
  v07a_eva02_seed42.py      # Model A: EVA-02 Large, seed 42
  v07b_eva02_seed123.py     # Model B: EVA-02 Large, seed 123, different aug
  v07c_convnext_large.py    # Model C: ConvNeXt-Large @ 384
  v07d_pseudo_labeling.py   # Load 3 models, PL, retrain, save embeddings
  v07e_ensemble_submit.py   # Fuse embeddings, QE, rerank, submit
```

Each training notebook (A/B/C) saves to /kaggle/working/output/:
- model_v07X.pth (weights for PL notebook)
- embeddings_v07X_train.npy (1895, D)
- embeddings_v07X_test.npy (371, D)
- filenames_train.json, filenames_test.json

The PL notebook loads these as Kaggle datasets (uploaded from previous runs).

## Expected Score Progression

| Stage | Technique | Expected LB |
|-------|-----------|-------------|
| Single EVA-02 + cosine | Baseline large model | ~0.92-0.93 |
| + GeM + flip TTA | Better pooling + augmented inference | ~0.93-0.94 |
| + QE + rerank | Post-processing | ~0.94-0.95 |
| + 2nd EVA seed | Same-arch diversity | ~0.95-0.96 |
| + ConvNeXt-Large | Cross-arch diversity | ~0.96-0.97 |
| + Pseudo-labeling | Semi-supervised | ~0.97+ |
| + Tuned rerank params | Final optimisation | ~0.97+ |

## Fallback Improvements (if 0.97 not reached)

- Add 4th model: SwinV2-Large @ 384px
- Multi-round pseudo-labeling (2-3 iterations)
- SubCenter ArcFace (k=3)
- Progressive resolution training (336 -> 448)
- Gallery diffusion (random walk on similarity graph)
- Cluster-based hard assignment (31 clusters, binary similarity)
- Dynamic per-class ArcFace margins
