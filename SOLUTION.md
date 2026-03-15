# Solution Writeup — 4th Place, Jaguar Re-ID (0.939 private LB)

## Quick summary

**4th place / 348 teams** — private LB 0.939.

The core idea: train two EVA-02 Large models with different random seeds + one DINOv2-Large model, apply pseudo-labeling to expand the training set with test images, then ensemble all three with query expansion and k-reciprocal reranking at inference. No tricks, no external data — just careful choices at each step.

All notebook versions (v01 through v07g) are public on the competition page.

---

## The journey

I approached this as a systematic progression from dumb-but-interpretable to powerful-but-principled. Every step was benchmarked before moving on.

**v01 — Frozen EfficientNet-B0 (LB ~0.28)**: The floor. ImageNet features + cosine similarity, no training. Useful mainly to confirm there is signal in pretrained features — jaguars do look different enough that off-the-shelf networks can partially distinguish them.

**v02 — Fine-tuned 31-class classifier (LB 0.414)**: Standard image classification. The embedding space isn't explicitly trained for retrieval, but fine-tuning on the actual 31 identities helped a lot. +73% over the frozen baseline.

**v03 — ArcFace metric learning (LB 0.609)**: This was the first real jump. ArcFace directly optimises the embedding space for re-identification by adding a margin penalty on the angle between an embedding and its class centre. Switching from CrossEntropy to ArcFace: +47%.

**v04 — Backbone search (LB 0.797)**: Compared tf_efficientnetv2_s, convnext_small, vit_small. ConvNeXt won cleanly. Added image caching + AMP to fit 3 backbones in under 25 minutes total.

**v05 — Heavy augmentation + TTA (LB 0.795)**: This one hurt. Heavy aug (random rotation, aggressive colour jitter, random erasing) + 5-view test-time augmentation. Regression on LB. With only 1,895 training images, strong augmentation over-regularises — the model becomes too robust to transformations it shouldn't be robust to.

**v06 — Hybrid LightGBM (LB 0.799)**: Built 571-dim pairwise features (cosine, L2, |emb_i − emb_j|, colour histogram diffs) and trained LightGBM on top of ArcFace embeddings. The LightGBM val mAP (0.491) was below raw cosine (0.510). ArcFace already encodes the right similarity — adding a learned layer on top adds noise.

**v07a/b — EVA-02 Large (LB 0.918 / 0.915)**: This was the big leap. Moved from ConvNeXt-Small to EVA-02 Large (300M parameters, patch14 at 448px). The private LB went from 0.799 to 0.918 — a +15% jump in a single step. Two seeds (42 and 123) to enable ensemble diversity later.

**v07f — DINOv2-Large-reg4 (LB 0.886)**: Added DINOv2 as a third model. The reasoning: EVA-02 is supervised (trained on ImageNet-21k labels), DINOv2 is self-supervised (DINO+iBOT on 142M curated images). Different pretraining paradigms should produce genuinely different feature representations, making them useful ensemble members even when all three are the same architecture class.

**v07d — Pseudo-labeling (LB 0.935)**: Used the three trained models to generate pseudo-labels for test images: only images where all three models agreed (confidence > 0.90) were added to training. Then fine-tuned EVA-02a for 5 more epochs at lr=1e-5. ~100-150 pseudo-labels added.

**v07e — Final ensemble (LB 0.939)**: Simple 3-way average of EVA-A + EVA-B + DINOv2 embeddings (all 1024-dim), followed by query expansion and k-reciprocal reranking. This is the selected submission.

**v07g — Round-2 pseudo-labeling (LB 0.937)**: Tried a second round of pseudo-labeling on the already-fine-tuned model with a more aggressive strategy (single-model confidence using ArcFace head weights, ~300+ labels). Small regression. Stacking two PL rounds adds noise rather than signal — the first round already captured most of the easy assignments.

---

## The best notebook in detail (v07e)

The final pipeline has five distinct stages. Here's what each does and why.

### 1. Backbone: EVA-02 Large @ 448px

EVA-02 is a ViT-based model (patch size 14, 448×448 input → 32×32 = 1024 patch tokens) pretrained with masked image modelling on ImageNet-21k then fine-tuned on ImageNet-1k. The key property for re-ID is the resolution: at 448px with 14px patches, each token covers a 14×14px region of the image. Jaguar spot patterns have detail at roughly that scale, so the model's attention can resolve individual spots rather than blurred regions.

ConvNeXt at 224px with its effective receptive fields was already working well, but EVA-02 at 448px sees nearly 4× more pixel area per decision. That's the main reason for the +15% LB jump.

Two seeds (42 and 123) are trained separately. They converge to slightly different regions of the embedding space — enough to reduce ensemble variance when averaged later.

### 2. Embedding head: GeM + BN + ArcFace

**GeM pooling** (Generalised Mean Pooling, learnable p≈3): takes the 1024 patch tokens from the ViT, reshapes them into a 32×32 spatial grid, and pools with a learnable power mean. When p=1 it's average pooling; as p increases it increasingly emphasises the highest-activating (most discriminative) spatial locations. For re-ID this matters — you want the spot pattern regions to dominate the embedding, not the background.

**Batch normalisation** on the embedding before ArcFace stabilises training and makes the embedding norm uniform across samples, which is important for cosine similarity to be meaningful.

**ArcFace** (s=30, m=0.5): the angular margin loss. For each batch, it adds a penalty of 0.5 radians (≈29°) to the angle between each embedding and its correct class centre. This forces the model to place embeddings closer to their class centre than any other centre — directly optimising the geometry used at inference.

### 3. Training: EMA + gradient checkpointing

**EMA** (exponential moving average, decay=0.999): maintains a shadow copy of the model weights that is a running average over the last ~1000 gradient steps. The shadow model is used at inference. This smooths out late-training oscillations and consistently produces better retrieval performance than the final raw checkpoint.

**Gradient checkpointing**: at 448px with a 300M parameter ViT and batch size 4, we're near the P100's VRAM limit. Gradient checkpointing recomputes activations during the backward pass rather than storing them, trading ~30% compute overhead for ~40% VRAM savings. Without it, training would require batch size 1-2 which hurts convergence.

**20 epochs matters**: v07a at 10 epochs scored 0.849. At 20 epochs it scored 0.918. The +0.069 gap is because EVA-02's large capacity takes time to converge on a 1,895-image dataset — it's learning slowly but surely.

### 4. Pseudo-labeling

After training all three models, we use them to label the 371 test images. Only images where all three models predicted the same identity with confidence > 0.90 were accepted as pseudo-labels. This 3-way agreement filter is conservative (~100-150 samples) but high precision — we're adding mostly correct labels.

The fine-tuning step (5 epochs at lr=1e-5 = 10× smaller than the original lr) is deliberately gentle. The goal is to reinforce known patterns with the new test-domain samples, not to overfit to potentially noisy pseudo-labels.

### 5. Ensemble + post-processing

**3-way average**: simple arithmetic mean of the L2-normalised embeddings from EVA-A, EVA-B, and DINOv2 (all 1024-dim). Then re-normalise. This is the right fusion strategy for same-dimension embeddings: it's equivalent to taking the centroid in the embedding space. We tried concatenation (EVA 1024d + ConvNeXt 1536d) earlier and it regressed results — concatenation puts architecturally different features in the same vector without any calibration, the distance metric becomes meaningless.

**Query Expansion** (top-3 DBA): for each image, replace its embedding with the average of itself and its 3 nearest neighbours in the gallery. This uses the assumption that neighbours in embedding space are likely the same identity — a reasonable assumption for well-trained ArcFace models.

**K-reciprocal reranking** (k1=20, λ=0.3): the Zhong et al. 2017 reranking algorithm. The idea: if image A is in image B's top-k neighbourhood AND B is in A's top-k neighbourhood, they are "k-reciprocal nearest neighbours" — very likely the same identity. It computes a Jaccard similarity over these reciprocal sets and blends it with the original cosine similarity (λ controls the blend). This consistently boosted scores by 0.01-0.02 throughout the competition.

---

## What didn't work

- **Heavy augmentation**: over-regularises at 1,895 images. Jaguars are already diverse enough.
- **LightGBM on pairwise features**: ArcFace geometry is already optimal for cosine; adding a nonlinear layer on top hurts.
- **Cross-architecture concat**: mixing 1024-dim and 1536-dim embeddings. The Euclidean distances become dominated by the larger-norm component.
- **Round-2 pseudo-labeling**: stacking PL rounds. Second round adds noise the first round already filtered.

---

## Project

All training and inference code is on GitHub: [Smooth-Cactus0/jaguar-re-identification](https://github.com/Smooth-Cactus0/jaguar-re-identification). The repo includes all notebook versions (v01–v07g), EDA notebooks, reusable `src/` modules, and benchmark results.

Most of the notebooks in this project were built with the help of **Claude Sonnet 4.6** — useful for rapidly prototyping and iterating on training loops, post-processing pipelines, and debugging Kaggle-specific issues (path resolution, VRAM management, etc.).

— Alexy Louis
