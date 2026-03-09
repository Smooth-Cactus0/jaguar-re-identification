"""
v07e: Ensemble + Post-Processing → Final Submission
====================================================
Phase 3 of the v07 pipeline. CPU-only, ~5 minutes.

Strategy:
  1. Load PL embeddings from v07d (or base from v07a/b/f if no PL)
  2. Average all 3 embeddings (v07a + v07b + v07f → all 1024-dim)
     — EVA-02 x2 (supervised) + DINOv2-reg4 (self-supervised)
     — Same dim → clean average, no concat noise
  3. L2-normalise the averaged embeddings
  4. Apply DBA (Database-side Augmentation / weighted QE)
  5. Apply k-reciprocal reranking with tuned parameters
  6. Generate submission

Also generates ablation submissions (no rerank, no QE) for comparison.

Inputs (uploaded as Kaggle dataset from v07d output):
  - embeddings_v07{a,b,f}_pl_test.npy
  - filenames_v07{a,b,f}_pl_test.json

Outputs:
  - submission.csv (final, to /kaggle/working/)
  - submission_v07e_full.csv (3-avg + DBA + rerank)
  - submission_v07e_qe_only.csv (3-avg + DBA, no rerank)
  - submission_v07e_cosine_only.csv (3-avg, no post-processing)
"""

import json, time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

# ── Config ───────────────────────────────────────────────────────────────

VERSION = 'v07e'

# DBA (weighted Query Expansion) parameters
DBA_TOP_K = 5
DBA_ITERATIONS = 1       # can increase to 2 for stronger smoothing

# K-reciprocal reranking parameters (tuned for small 371-image gallery)
RERANK_K1 = 15           # neighbourhood size (default 20, reduced for small gallery)
RERANK_K2 = 6            # secondary neighbourhood
RERANK_LAMBDA = 0.4      # Jaccard weight (default 0.3, increased for small gallery)

# ── Paths ────────────────────────────────────────────────────────────────

KAGGLE_INPUT = Path('/kaggle/input/competitions/jaguar-re-id')
OUT_DIR = Path('/kaggle/working/output')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Auto-detect input directories ────────────────────────────────────────
# Searches /kaggle/input/ for embedding files — works whether you attached
# notebook outputs directly or uploaded them as named datasets.

def find_embedding_dir(version: str, pl: bool = True) -> Path | None:
    """Find the directory containing embeddings for a given version."""
    search_root = Path('/kaggle/input')
    suffix = 'pl_test' if pl else 'test'
    marker = f'embeddings_{version}_{suffix}.npy'
    matches = list(search_root.rglob(marker))
    if matches:
        return matches[0].parent
    return None

# Try PL embeddings first (from v07d), fall back to base embeddings (from v07a/b/f)
_pl_dir = find_embedding_dir('v07a', pl=True)
PL_DIR = _pl_dir if _pl_dir is not None else Path('/kaggle/input/v07d-output')

BASE_DIRS = {}
for _v in ['v07a', 'v07b', 'v07f']:
    _d = find_embedding_dir(_v, pl=False)
    BASE_DIRS[_v] = _d if _d is not None else Path(f'/kaggle/input/{_v}-output')

print(f'PL embeddings dir:   {PL_DIR}  (exists: {PL_DIR.exists()})')
for _v, _d in BASE_DIRS.items():
    print(f'Base {_v} dir: {_d}  (exists: {_d.exists()})')

# ── Helper functions ─────────────────────────────────────────────────────

def load_embeddings(version, suffix='test', use_pl=True):
    """Load embeddings, trying PL first, then base."""
    if use_pl:
        pl_path = PL_DIR / f'embeddings_{version}_pl_{suffix}.npy'
        if pl_path.exists():
            emb = np.load(pl_path)
            print(f'  {version} {suffix}: loaded PL embeddings {emb.shape}')
            return emb

    base_path = BASE_DIRS[version] / f'embeddings_{version}_{suffix}.npy'
    if base_path.exists():
        emb = np.load(base_path)
        print(f'  {version} {suffix}: loaded BASE embeddings {emb.shape}')
        return emb

    raise FileNotFoundError(f'No embeddings found for {version} {suffix}')


def load_filenames(version, suffix='test', use_pl=True):
    """Load filename list, trying PL first, then base."""
    if use_pl:
        pl_path = PL_DIR / f'filenames_{version}_pl_{suffix}.json'
        if pl_path.exists():
            with open(pl_path) as f:
                return json.load(f)

    base_path = BASE_DIRS[version] / f'filenames_{version}_{suffix}.json'
    if base_path.exists():
        with open(base_path) as f:
            return json.load(f)

    raise FileNotFoundError(f'No filenames found for {version} {suffix}')


def l2_normalise(emb):
    """L2-normalise along the feature dimension."""
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return emb / norms


# ── Embedding fusion ─────────────────────────────────────────────────────

def fuse_embeddings(emb_a, emb_b, emb_f):
    """Fuse embeddings: average all 3 models (all 1024-dim).

    EVA-02 A (supervised, seed=42) + EVA-02 B (supervised, seed=123)
    + DINOv2-Large-reg4 (self-supervised) — all 1024-dim ViT-L/14.
    Clean average: no cross-arch concat noise.
    Result: 1024-dim fused embedding.
    """
    emb_fused = (emb_a + emb_b + emb_f) / 3
    emb_fused = l2_normalise(emb_fused)
    print(f'  3-way average: {emb_a.shape} → fused: {emb_fused.shape}')
    return emb_fused


# ── DBA (Database-side Augmentation / Weighted Query Expansion) ──────────

def dba_query_expansion(emb, top_k=5, iterations=1):
    """Weighted QE: replace each embedding with similarity-weighted mean
    of its top-k neighbours. Unlike simple QE which uses uniform weights,
    DBA weights neighbours by their cosine similarity — closer neighbours
    have more influence.

    Can be applied iteratively (2+ rounds) for stronger smoothing, though
    1 round is usually sufficient."""
    for it in range(iterations):
        sims = emb @ emb.T
        indices = np.argsort(-sims, axis=1)[:, :top_k]
        new_emb = np.zeros_like(emb)
        for i in range(len(emb)):
            weights = sims[i, indices[i]]
            weights = weights / weights.sum()
            new_emb[i] = (emb[indices[i]] * weights[:, None]).sum(axis=0)
        emb = l2_normalise(new_emb)
        if iterations > 1:
            print(f'    DBA iteration {it+1}/{iterations} complete')
    return emb


# ── K-Reciprocal Reranking ──────────────────────────────────────────────

def k_reciprocal_rerank(prob, k1=20, k2=6, lambda_value=0.3):
    """K-reciprocal reranking (Zhong et al., CVPR 2017).

    Refines cosine similarity using Jaccard distance between
    k-reciprocal nearest neighbour sets. Key insight: mutual
    nearest neighbours (A in B's top-k AND B in A's top-k) are
    strong identity matches.

    For our small 371-image gallery:
    - k1=15 (reduced from 20 — fewer images need smaller neighbourhood)
    - lambda=0.4 (increased — trust Jaccard structure more)
    """
    print(f'  Reranking (k1={k1}, k2={k2}, λ={lambda_value})...')
    q_g_dist = 1 - prob
    original_dist = q_g_dist.copy()
    initial_rank = np.argsort(original_dist, axis=1)

    # Build k-reciprocal nearest neighbour sets
    nn_k1 = []
    for i in range(prob.shape[0]):
        forward_k1 = initial_rank[i, :k1 + 1]
        backward_k1 = initial_rank[forward_k1, :k1 + 1]
        fi = np.where(backward_k1 == i)[0]
        nn_k1.append(forward_k1[fi])

    # Compute Jaccard distance
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

    # Weighted combination of Jaccard and original distance
    final_sim = 1 - (jaccard_dist * lambda_value +
                     original_dist * (1 - lambda_value))
    return final_sim


# ── Submission generation ────────────────────────────────────────────────

def make_submission(test_df, sim_matrix, img_map, output_path):
    """Map NxN similarity matrix to submission CSV."""
    preds = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc='Mapping'):
        q_idx = img_map[row['query_image']]
        g_idx = img_map[row['gallery_image']]
        s = sim_matrix[q_idx, g_idx]
        preds.append(max(0.0, min(1.0, float(s))))

    sub = pd.DataFrame({'row_id': test_df['row_id'], 'similarity': preds})
    sub.to_csv(output_path, index=False)
    print(f'  Saved → {output_path}')
    print(f'  Range: {np.min(preds):.4f} – {np.max(preds):.4f} | '
          f'Mean: {np.mean(preds):.4f}')
    return sub


# ══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t0 = time.time()

    test_df = pd.read_csv(KAGGLE_INPUT / 'test.csv')
    print(f'Test pairs: {len(test_df)}')

    # ── Load embeddings ──────────────────────────────────────────────────
    print('\n' + '='*60)
    print('Loading embeddings')
    print('='*60)

    emb_a = load_embeddings('v07a')
    emb_b = load_embeddings('v07b')
    emb_f = load_embeddings('v07f')
    test_names = load_filenames('v07a')

    img_map = {n: i for i, n in enumerate(test_names)}

    # ── Fuse embeddings ──────────────────────────────────────────────────
    print('\n' + '='*60)
    print('Fusing embeddings')
    print('='*60)

    emb_fused = fuse_embeddings(emb_a, emb_b, emb_f)

    # ── Ablation 1: Raw cosine (no post-processing) ─────────────────────
    print('\n' + '='*60)
    print('Submission 1: Cosine only (no post-processing)')
    print('='*60)

    sim_cosine = emb_fused @ emb_fused.T
    make_submission(test_df, sim_cosine, img_map,
                    OUT_DIR / f'submission_{VERSION}_cosine_only.csv')

    # ── Ablation 2: DBA only (no reranking) ──────────────────────────────
    print('\n' + '='*60)
    print(f'Submission 2: DBA (top_k={DBA_TOP_K}) only')
    print('='*60)

    emb_dba = dba_query_expansion(emb_fused, top_k=DBA_TOP_K,
                                   iterations=DBA_ITERATIONS)
    sim_dba = emb_dba @ emb_dba.T
    make_submission(test_df, sim_dba, img_map,
                    OUT_DIR / f'submission_{VERSION}_qe_only.csv')

    # ── Final: DBA + K-reciprocal reranking ──────────────────────────────
    print('\n' + '='*60)
    print(f'Submission 3: DBA + Reranking (FINAL)')
    print('='*60)

    sim_final = k_reciprocal_rerank(sim_dba, k1=RERANK_K1, k2=RERANK_K2,
                                     lambda_value=RERANK_LAMBDA)
    make_submission(test_df, sim_final, img_map,
                    OUT_DIR / f'submission_{VERSION}_full.csv')

    # Save as the main submission
    make_submission(test_df, sim_final, img_map,
                    Path('/kaggle/working/submission.csv'))

    # ── Also try individual model submissions for comparison ─────────────
    print('\n' + '='*60)
    print('Individual model submissions (for analysis)')
    print('='*60)

    for version, emb in [('v07a', emb_a), ('v07b', emb_b), ('v07f', emb_f)]:
        emb_qe = dba_query_expansion(emb, top_k=3, iterations=1)
        sim = emb_qe @ emb_qe.T
        sim = k_reciprocal_rerank(sim, k1=RERANK_K1, k2=RERANK_K2,
                                   lambda_value=RERANK_LAMBDA)
        make_submission(test_df, sim, img_map,
                        OUT_DIR / f'submission_{version}_individual.csv')

    # ── Summary ──────────────────────────────────────────────────────────
    total_time = time.time() - t0
    print(f'\n{"="*60}')
    print(f'{VERSION} complete! Total: {total_time:.0f}s')
    print(f'{"="*60}')
    print(f'\nSubmissions generated:')
    print(f'  1. submission_{VERSION}_cosine_only.csv  (raw cosine)')
    print(f'  2. submission_{VERSION}_qe_only.csv      (DBA, no rerank)')
    print(f'  3. submission_{VERSION}_full.csv          (DBA + rerank) ← MAIN')
    print(f'  + 3 individual model submissions')
    print(f'\nThe main submission.csv uses the full DBA + reranking pipeline.')
