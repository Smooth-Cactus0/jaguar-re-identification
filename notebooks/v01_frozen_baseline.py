"""
v01: Frozen Pretrained Baseline
================================
No training. Use a pretrained EfficientNet-B0 (ImageNet weights) as a frozen
feature extractor and measure how well off-the-shelf ImageNet features
distinguish individual jaguars via cosine similarity.

This establishes the performance floor and validates our evaluation pipeline.

Run from the notebooks/ directory:
    python v01_frozen_baseline.py
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
import torch
import warnings
warnings.filterwarnings('ignore')

from src.data      import JaguarDataset, get_transforms, encode_labels
from src.models    import get_model
from src.inference import extract_embeddings, make_submission
from src.evaluate  import compute_map, print_results, save_benchmark

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).parent.parent
TRAIN_DIR  = ROOT / 'train' / 'train'
TEST_DIR   = ROOT / 'test'  / 'test'
FIGURES    = ROOT / 'figures'
RESULTS    = ROOT / 'results'
SUBS       = ROOT / 'submissions'
RESULTS.mkdir(exist_ok=True)
SUBS.mkdir(exist_ok=True)

BACKBONE    = 'efficientnet_b0'
IMG_SIZE    = 224
BATCH_SIZE  = 32          # conservative for CPU / 4GB VRAM
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
VERSION     = 'v01'

print('\n' + '='*60)
print(f'  {VERSION}: Frozen Pretrained Baseline')
print(f'  Backbone : {BACKBONE}')
print(f'  Device   : {DEVICE}')
print(f'  Img size : {IMG_SIZE}')
print('='*60)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print('\n[1/6] Loading data...')
train_df = pd.read_csv(ROOT / 'train.csv')
test_df  = pd.read_csv(ROOT / 'test.csv')

labels, label_to_idx, idx_to_label = encode_labels(train_df['ground_truth'])
n_classes = len(label_to_idx)
print(f'  Train: {len(train_df)} images, {n_classes} identities')
print(f'  Test:  {len(test_df)} pairs')

# ---------------------------------------------------------------------------
# 2. Build model (frozen)
# ---------------------------------------------------------------------------
print('\n[2/6] Building model...')
model = get_model(BACKBONE, pretrained=True, freeze=True)
model.eval()

# ---------------------------------------------------------------------------
# 3. Extract train embeddings
# ---------------------------------------------------------------------------
print('\n[3/6] Extracting train embeddings...')
tf = get_transforms(mode='val', img_size=IMG_SIZE)
train_ds = JaguarDataset(
    filenames=train_df['filename'].tolist(),
    img_dir=TRAIN_DIR,
    labels=labels,
    transform=tf,
)
train_embs = extract_embeddings(model, train_ds, batch_size=BATCH_SIZE,
                                device=DEVICE, desc='Train embeddings')
print(f'  Shape: {train_embs.shape}  |  dtype: {train_embs.dtype}')
print(f'  L2 norm check (should be ~1.0): {np.linalg.norm(train_embs[0]):.4f}')

# ---------------------------------------------------------------------------
# 4. Evaluate: identity-balanced mAP on full training set (leave-one-out)
# ---------------------------------------------------------------------------
print('\n[4/6] Computing identity-balanced mAP (leave-one-out on train)...')
results = compute_map(train_embs, labels, identity_balanced=True)
print_results(results, idx_to_label, title=f'{VERSION} - Frozen {BACKBONE}')

# Save to benchmarks CSV
save_benchmark(
    RESULTS / 'benchmarks.csv',
    {
        'version':         VERSION,
        'backbone':        BACKBONE,
        'loss':            'none (frozen)',
        'embedding_dim':   model.out_dim,
        'img_size':        IMG_SIZE,
        'augmentation':    'none',
        'cv_map':          round(results['map'], 5),
        'best_identity':   idx_to_label[max(results['per_identity_ap'], key=results['per_identity_ap'].get)],
        'best_ap':         round(max(results['per_identity_ap'].values()), 5),
        'worst_identity':  idx_to_label[min(results['per_identity_ap'], key=results['per_identity_ap'].get)],
        'worst_ap':        round(min(results['per_identity_ap'].values()), 5),
        'notes':           'ImageNet pretrained, no fine-tuning',
    }
)

# ---------------------------------------------------------------------------
# 5. Visualise results
# ---------------------------------------------------------------------------
print('\n[5/6] Generating visualisations...')

# 5a. Per-identity AP bar chart
per_id_ap = {idx_to_label[k]: v for k, v in results['per_identity_ap'].items()}
per_id_df = pd.Series(per_id_ap).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(14, 5))
colors = ['steelblue' if v > results['map'] else 'tomato' for v in per_id_df.values]
bars = ax.bar(per_id_df.index, per_id_df.values, color=colors, edgecolor='white')
ax.axhline(results['map'], color='black', linestyle='--', linewidth=1.5,
           label=f'mAP = {results["map"]:.4f}')
ax.set_xlabel('Jaguar Identity', fontsize=11)
ax.set_ylabel('Average Precision', fontsize=11)
ax.set_title(f'v01: Per-Identity AP (Frozen {BACKBONE} + Cosine Similarity)',
             fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.set_ylim(0, 1)
ax.legend(fontsize=10)
plt.tight_layout()
fig.savefig(FIGURES / f'{VERSION}_per_identity_ap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved -> figures/{VERSION}_per_identity_ap.png')

# 5b. t-SNE of train embeddings
print('  Computing t-SNE (may take 30-60s)...')
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, n_jobs=-1)
embs_2d = tsne.fit_transform(train_embs)

fig, ax = plt.subplots(figsize=(12, 10))
palette = sns.color_palette('tab20', n_classes) + sns.color_palette('tab20b', max(0, n_classes-20))
for lbl in range(n_classes):
    mask = labels == lbl
    ax.scatter(embs_2d[mask, 0], embs_2d[mask, 1],
               color=palette[lbl], label=idx_to_label[lbl],
               s=20, alpha=0.7, edgecolors='none')
ax.set_title(f'v01: t-SNE of Frozen {BACKBONE} Embeddings (train set, {n_classes} identities)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('t-SNE dim 1'); ax.set_ylabel('t-SNE dim 2')
ax.legend(loc='upper right', fontsize=6, ncol=2, markerscale=1.5,
          framealpha=0.7, title='Identity')
ax.axis('off')
plt.tight_layout()
fig.savefig(FIGURES / f'{VERSION}_tsne.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved -> figures/{VERSION}_tsne.png')

# 5c. Sample retrieval visualisation
# For 4 query images (one per each 'difficulty' quartile), show top-5 matches
def show_retrievals(query_indices, train_embs, train_df, labels, n_show=5):
    sim = train_embs @ train_embs.T
    n_q  = len(query_indices)
    fig, axes = plt.subplots(n_q, n_show + 1,
                             figsize=((n_show + 1) * 1.8, n_q * 2.2))
    fig.suptitle(f'v01: Sample Retrievals (Query | Top-{n_show} by cosine similarity)',
                 fontsize=11, fontweight='bold')

    from src.data import load_rgba_as_rgb, pad_to_square
    for row, qi in enumerate(query_indices):
        # Query
        q_fname = train_df['filename'].iloc[qi]
        q_img   = pad_to_square(load_rgba_as_rgb(TRAIN_DIR / q_fname))
        q_label = idx_to_label[labels[qi]]
        axes[row, 0].imshow(q_img.resize((128, 128)))
        axes[row, 0].set_title(f'QUERY\n{q_label}', fontsize=7, fontweight='bold',
                               color='navy')
        axes[row, 0].axis('off')
        axes[row, 0].patch.set_edgecolor('navy')
        axes[row, 0].patch.set_linewidth(3)

        # Top-N gallery (excluding self)
        scores = sim[qi].copy()
        scores[qi] = -999
        top_idx = np.argsort(-scores)[:n_show]
        for col, gi in enumerate(top_idx):
            g_fname = train_df['filename'].iloc[gi]
            g_img   = pad_to_square(load_rgba_as_rgb(TRAIN_DIR / g_fname))
            g_label = idx_to_label[labels[gi]]
            match   = (labels[gi] == labels[qi])
            color   = 'green' if match else 'red'
            axes[row, col+1].imshow(g_img.resize((128, 128)))
            axes[row, col+1].set_title(f'{g_label}\nsim={scores[gi]:.2f}',
                                        fontsize=7, color=color)
            axes[row, col+1].axis('off')

    plt.tight_layout()
    return fig

# Pick 6 representative queries: 2 common, 2 medium, 2 rare identities
from src.data import encode_labels as _enc
id_counts = train_df['ground_truth'].value_counts()
common_ids = id_counts.index[:2].tolist()
medium_ids = id_counts.index[len(id_counts)//2 : len(id_counts)//2 + 2].tolist()
rare_ids   = id_counts.index[-2:].tolist()
query_ids  = common_ids + medium_ids + rare_ids

query_indices = []
for identity in query_ids:
    idxs = train_df[train_df.ground_truth == identity].index.tolist()
    query_indices.append(idxs[len(idxs)//2])   # pick a middle image

retrieval_fig = show_retrievals(query_indices, train_embs, train_df, labels)
retrieval_fig.savefig(FIGURES / f'{VERSION}_sample_retrievals.png',
                       dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved -> figures/{VERSION}_sample_retrievals.png')

# ---------------------------------------------------------------------------
# 6. Generate submission
# ---------------------------------------------------------------------------
print('\n[6/6] Generating test submission...')
test_filenames = sorted(pd.unique(
    test_df[['query_image', 'gallery_image']].values.ravel()
))
test_ds = JaguarDataset(
    filenames=test_filenames,
    img_dir=TEST_DIR,
    labels=None,
    transform=tf,
)
test_embs = extract_embeddings(model, test_ds, batch_size=BATCH_SIZE,
                               device=DEVICE, desc='Test embeddings')
fname_to_idx = {f: i for i, f in enumerate(test_filenames)}
make_submission(
    test_df=test_df,
    test_embeddings=test_embs,
    filename_to_idx=fname_to_idx,
    output_path=SUBS / f'submission_{VERSION}.csv',
)

print(f'\n{"="*60}')
print(f'  {VERSION} COMPLETE')
print(f'  CV mAP: {results["map"]:.4f}')
print(f'  Submission: submissions/submission_{VERSION}.csv')
print(f'{"="*60}')
