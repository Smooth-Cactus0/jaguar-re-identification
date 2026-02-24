# CLAUDE.md — Jaguar Re-Identification Competition

## Competition Overview

**Kaggle**: [Jaguar Re-ID](https://www.kaggle.com/competitions/jaguar-re-id)
**Task**: Closed-set all-vs-all re-identification — predict pairwise similarity scores between test images of jaguars.
**Metric**: Identity-balanced Mean Average Precision (mAP) — macro-averaged across 31 jaguar identities.
**Author**: Alexy Louis

## Dataset

- **Train**: 1,895 images across 31 jaguars (13–183 images per individual). Images in `train/train/`.
- **Test**: 371 images, 137,270 pairwise comparisons. Images in `test/test/`.
- **All 31 test identities appear in the training set** (closed-set).
- Images are pre-cropped jaguar cutouts (background removed via SAM3).
- Significant class imbalance; metric compensates via identity-balanced averaging.

## Compute Resources

- **Primary training**: Kaggle notebooks (P100/T4, 16GB VRAM)
- **Local**: GTX 1050 (4GB VRAM) — used for EDA, prototyping, visualization, git management
- **Python**: `C:/Users/alexy/AppData/Local/Programs/Python/Python311/python.exe`

## Project Structure

```
jaguar-re-id/
├── CLAUDE.md                    # This file
├── README.md                    # GitHub-facing project description
├── train.csv                    # Training labels
├── test.csv                     # Test pairs
├── sample_submission.csv        # Submission template
├── train/train/                 # Training images (1,895 PNGs)
├── test/test/                   # Test images (371 PNGs)
├── notebooks/                   # Local notebooks (EDA, visualization, analysis)
├── kaggle_notebooks/            # Kaggle GPU notebooks (training, inference)
├── src/                         # Reusable Python modules
│   ├── data.py                  # Dataset classes, dataloaders, augmentation
│   ├── models.py                # Model definitions (backbones, heads)
│   ├── losses.py                # Loss functions (ArcFace, CosFace, triplet)
│   ├── train.py                 # Training loop, validation
│   ├── evaluate.py              # mAP computation, per-identity metrics
│   ├── inference.py             # Embedding extraction, similarity scoring
│   ├── utils.py                 # General utilities
│   └── visualize.py             # t-SNE, confusion plots, embedding viz
├── configs/                     # Training configs (YAML or Python dicts)
├── figures/                     # Saved plots and visualizations for GitHub
├── results/                     # Benchmark CSVs, model comparison tables
├── submissions/                 # Generated submission files
└── models/                      # Saved model weights (gitignored)
```

## Workflow

- **Local notebooks** for EDA, visualization, and analysis — run locally, save figures, push to GitHub.
- **Kaggle notebooks** for heavy GPU training — minimal visuals, focused on compute.
- **src/ modules** are shared between local and Kaggle. On Kaggle, clone the repo and add to sys.path.
- All advances are benchmarked (CV mAP), visualized, and documented before pushing.

## Model Progression Roadmap

Each version is benchmarked and documented. Order matters — each builds on learnings from the previous.

### v01 — Simple Baseline (no training)
- Pretrained ResNet/EfficientNet as frozen feature extractor
- Cosine similarity on extracted embeddings
- Establishes performance floor
- **Goal**: Understand embedding quality of pretrained features on jaguar spot patterns

### v02 — Fine-tuned Classifier
- Train as 31-class classification (CrossEntropy loss)
- Use penultimate layer embeddings for similarity scoring
- First trained baseline
- **Goal**: See how much fine-tuning helps over frozen features

### v03 — Metric Learning
- ArcFace / CosFace loss head
- Directly optimizes embedding space for re-ID
- **Goal**: Significant jump — metric learning is the core of re-ID

### v04 — Backbone Exploration
- Compare EfficientNetV2, ConvNeXt, ViT
- Same ArcFace head, compare embedding quality
- **Goal**: Find best backbone for jaguar spot patterns

### v05 — Augmentation & TTA
- Training augmentation: flips, color jitter, random erasing, cutout
- Test-Time Augmentation: multi-crop, flip averaging on embeddings
- **Goal**: Robustness to viewpoint, lighting, partial occlusion

### v06 — Metadata + Hybrid Features
- Extract metadata/engineered features from images (color histograms, texture, spot density)
- Combine with deep embeddings
- Feed into XGBoost / LightGBM for final similarity scoring
- **Goal**: Capture information that CNNs might miss

### v07 — Ensemble & Final Submission
- Blend best models via rank averaging or learned weights
- Optimize similarity score calibration
- Per-identity analysis to find and fix weak spots
- **Goal**: Maximum mAP, polished final submission

## Benchmarking Standards

Every model version must record:
- **CV mAP** (identity-balanced, matching competition metric)
- **Per-identity AP** breakdown (to spot weak jaguars)
- **Training time** and **inference time**
- **Embedding dimensionality**

Results are tracked in `results/benchmarks.csv` with columns:
```
version, backbone, loss, embedding_dim, augmentation, cv_map, best_identity_ap, worst_identity_ap, train_time_min, notes
```

## Visualization Standards

Each major version produces:
- **t-SNE / UMAP** of embeddings colored by identity
- **Per-identity AP bar chart**
- **Sample retrieval visualization** (query → top-5 matches)
- Figures saved to `figures/` with descriptive names (e.g., `v03_arcface_tsne.png`)

## Git Conventions

- Commit messages: `v01: simple baseline with frozen EfficientNet` format
- Branch: `main` for this project
- `.gitignore`: model weights, raw image data, `__pycache__`, `.ipynb_checkpoints`
- Push figures and results to GitHub — they tell the story

## Key Technical Notes

- **Closed-set**: All 31 test identities are in training. This means classification-based approaches are viable, but metric learning generalizes better.
- **Identity-balanced mAP**: Each of the 31 jaguars contributes equally regardless of image count. Models must perform well on rare jaguars (Bernard: 13 imgs, Ipepo: 13 imgs).
- **Near-duplicates**: Training set may contain near-duplicates from burst photography. Use careful validation splits (group by identity, not random).
- **Viewpoint variation**: Left flank ≠ right flank. Spot patterns are asymmetric. Horizontal flip augmentation needs careful consideration.
- **Image paths**: Train images at `train/train/train_XXXX.png`, test at `test/test/test_XXXX.png` (double-nested from Kaggle extraction).

## Dependencies

Core: `torch`, `torchvision`, `timm`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
Metric learning: `pytorch-metric-learning` (optional, can implement ArcFace manually)
Boosting: `xgboost`, `lightgbm`
Visualization: `umap-learn`, `plotly`
