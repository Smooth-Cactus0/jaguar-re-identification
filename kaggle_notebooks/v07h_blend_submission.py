"""
v07h: Submission blending — v07e (LB 0.939) + winning notebook (LB 0.941)
==========================================================================
CPU-only, runs in seconds. No GPU needed.

Inputs (attach as notebook datasets):
  1. v07e output   → contains submission_v07e.csv  or  submission.csv
  2. winning_nb output → contains submission.csv  (LB 0.941)

Strategy: weighted average of similarity scores.
  blend = alpha * sim_winner + (1 - alpha) * sim_v07e
  alpha = 0.5  (equal weight — can tune)

Why blending works even with similar architectures:
  - Both use EVA-02 Large, but trained with different seeds / PL strategies
  - Different similarity matrices → averaging smooths ranking errors
  - Empirical rule: blending two LB-0.94 submissions often yields 0.942–0.945
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

# ── Config ────────────────────────────────────────────────────────────────

VERSION = 'v07h'

# Blend weights. alpha → winner, (1-alpha) → v07e
# 0.5 = equal blend; try 0.4–0.6 if you have local CV
ALPHA = 0.5

OUT_DIR = Path('/kaggle/working')

# ── File discovery ────────────────────────────────────────────────────────

def find_file(pattern: str, required=True) -> Path | None:
    """rglob search under /kaggle/input for a filename pattern."""
    matches = list(Path('/kaggle/input').rglob(pattern))
    if matches:
        print(f'  Found {pattern}: {matches[0]}')
        return matches[0]
    if required:
        raise FileNotFoundError(
            f'Could not find {pattern!r} under /kaggle/input.\n'
            f'Attach the relevant notebook output as an input dataset.'
        )
    return None

# ── Load submissions ──────────────────────────────────────────────────────

print('='*60)
print('Loading submissions...')
print('='*60)

# v07e saves to /kaggle/working/output/submission_v07e_full.csv
# Winning notebook saves to /kaggle/working/submission.csv
# Attach them as separate input datasets to avoid rglob collisions.

print('\nIMPORTANT: Attach exactly two datasets:')
print('  Dataset 1: v07e notebook output   (contains submission_v07e_full.csv)')
print('  Dataset 2: winning notebook output (contains submission.csv)')
print()

sub_v07e_path   = find_file('submission_v07e_full.csv')
sub_winner_path = find_file('submission.csv')

sub_v07e   = pd.read_csv(sub_v07e_path).sort_values('row_id').reset_index(drop=True)
sub_winner = pd.read_csv(sub_winner_path).sort_values('row_id').reset_index(drop=True)

assert (sub_v07e['row_id'].values == sub_winner['row_id'].values).all(), \
    'row_id mismatch between submissions!'

print(f'\nv07e    sim: {sub_v07e["similarity"].mean():.4f} ± {sub_v07e["similarity"].std():.4f}')
print(f'Winner  sim: {sub_winner["similarity"].mean():.4f} ± {sub_winner["similarity"].std():.4f}')

# ── Blend ─────────────────────────────────────────────────────────────────

print(f'\nBlending: {ALPHA:.2f} × winner + {1-ALPHA:.2f} × v07e')

blended = ALPHA * sub_winner['similarity'].values + (1 - ALPHA) * sub_v07e['similarity'].values

# Also try rank-based blending as an alternative
rank_winner = sub_winner['similarity'].rank(pct=True).values
rank_v07e   = sub_v07e['similarity'].rank(pct=True).values
rank_blend  = ALPHA * rank_winner + (1 - ALPHA) * rank_v07e

print(f'\nScore blend  sim: {blended.mean():.4f} ± {blended.std():.4f}')
print(f'Rank  blend  sim: {rank_blend.mean():.4f} ± {rank_blend.std():.4f}')

# ── Save both options ─────────────────────────────────────────────────────

sub_score = pd.DataFrame({'row_id': sub_v07e['row_id'], 'similarity': blended})
sub_rank  = pd.DataFrame({'row_id': sub_v07e['row_id'], 'similarity': rank_blend})

sub_score.to_csv(OUT_DIR / f'submission_{VERSION}_score.csv', index=False)
sub_rank.to_csv( OUT_DIR / f'submission_{VERSION}_rank.csv',  index=False)

# Default submission.csv = score blend
sub_score.to_csv(OUT_DIR / 'submission.csv', index=False)

print(f'\nSaved:')
print(f'  submission_{VERSION}_score.csv  (score average  — submit this first)')
print(f'  submission_{VERSION}_rank.csv   (rank average   — submit if score blend fails)')
print(f'  submission.csv                  (= score blend, for quick submit)')
print(f'\n{VERSION} blend complete.')
