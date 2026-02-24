"""
src/evaluate.py — Identity-balanced mAP evaluation (matches competition metric).

The competition uses identity-balanced mAP: compute AP for every query image,
then macro-average first within each identity, then across identities. This
gives each of the 31 jaguars equal weight regardless of image count.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Core AP computation
# ---------------------------------------------------------------------------

def average_precision(scores: np.ndarray, labels: np.ndarray,
                      query_label: int) -> float:
    """
    Compute AP for a single query.

    Args:
        scores:       1-D array of similarity scores for all gallery items
        labels:       1-D array of integer labels for gallery items
        query_label:  the true label of the query

    Returns:
        AP score in [0, 1]
    """
    # Sort gallery by descending score
    order    = np.argsort(-scores)
    gt       = (labels[order] == query_label).astype(int)
    n_pos    = gt.sum()
    if n_pos == 0:
        return 0.0

    # Precision at each positive hit
    cum_hits = np.cumsum(gt)
    ranks    = np.arange(1, len(gt) + 1)
    precisions = cum_hits[gt == 1] / ranks[gt == 1]
    return float(precisions.mean())


# ---------------------------------------------------------------------------
# Full evaluation on embeddings
# ---------------------------------------------------------------------------

def compute_map(embeddings: np.ndarray, labels: np.ndarray,
                identity_balanced: bool = True) -> Dict:
    """
    Compute mAP from a matrix of L2-normalised embeddings.

    For each image as a query, the remaining images form the gallery.
    Uses cosine similarity (dot product of L2-normalised vectors).

    Args:
        embeddings:         (N, D) float32 array, L2-normalised
        labels:             (N,) integer labels
        identity_balanced:  if True, macro-average per-identity APs (competition metric)
                            if False, micro-average over all queries

    Returns:
        dict with keys: 'map', 'per_identity_ap', 'per_image_ap'
    """
    N = len(embeddings)
    # Full pairwise cosine similarity matrix
    sim_matrix = embeddings @ embeddings.T      # (N, N), values in [-1, 1]

    per_image_ap = np.zeros(N)
    for i in range(N):
        # Exclude self from gallery
        gallery_mask    = np.ones(N, dtype=bool)
        gallery_mask[i] = False
        scores = sim_matrix[i][gallery_mask]
        gal_labels = labels[gallery_mask]
        per_image_ap[i] = average_precision(scores, gal_labels, labels[i])

    # Per-identity AP: average over all images of that identity
    unique_labels = np.unique(labels)
    per_identity_ap = {}
    for lbl in unique_labels:
        mask = labels == lbl
        per_identity_ap[int(lbl)] = float(per_image_ap[mask].mean())

    if identity_balanced:
        map_score = float(np.mean(list(per_identity_ap.values())))
    else:
        map_score = float(per_image_ap.mean())

    return {
        'map':              map_score,
        'per_identity_ap':  per_identity_ap,
        'per_image_ap':     per_image_ap,
    }


# ---------------------------------------------------------------------------
# Pretty-print results
# ---------------------------------------------------------------------------

def print_results(results: Dict, idx_to_label: Dict = None,
                  title: str = 'Evaluation Results') -> None:
    """Print a summary table of mAP results."""
    print(f'\n{"="*50}')
    print(f'  {title}')
    print(f'{"="*50}')
    print(f'  Identity-balanced mAP : {results["map"]:.4f}')
    print()

    per_id = results['per_identity_ap']
    rows = []
    for lbl, ap in sorted(per_id.items(), key=lambda x: -x[1]):
        name = idx_to_label[lbl] if idx_to_label else str(lbl)
        rows.append({'identity': name, 'AP': ap})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format='{:.4f}'.format))
    print(f'\n  Best  : {df.iloc[0]["identity"]} ({df.iloc[0]["AP"]:.4f})')
    print(f'  Worst : {df.iloc[-1]["identity"]} ({df.iloc[-1]["AP"]:.4f})')
    print(f'{"="*50}\n')


# ---------------------------------------------------------------------------
# Save to benchmarks CSV
# ---------------------------------------------------------------------------

def save_benchmark(results_csv: 'Path', row: Dict) -> None:
    """Append a result row to the benchmarks CSV."""
    import pandas as pd
    from pathlib import Path
    path = Path(results_csv)
    df_new = pd.DataFrame([row])
    if path.exists():
        df = pd.read_csv(path)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(path, index=False)
    print(f'  Benchmark saved -> {path.name}')
