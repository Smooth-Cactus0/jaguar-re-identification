"""
v03 Local Analysis
==================
Run after downloading Kaggle outputs for v03.

Copy the downloaded files to the right places first:
  - figures/v03_*.png         (from output zip)
  - results/benchmarks_v03.csv
  - submissions/submission_v03.csv

Then run:
    python v03_analysis.py

Produces a progression chart (v01 -> v02 -> v03) and updates benchmarks.csv.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT    = Path(__file__).parent.parent
FIGURES = ROOT / 'figures'
RESULTS = ROOT / 'results'


# ---------------------------------------------------------------------------
# 1. Load and merge benchmark results
# ---------------------------------------------------------------------------
print('[1/3] Loading benchmark results...')

bench_prev  = pd.read_csv(RESULTS / 'benchmarks.csv')
bench_v03_path = RESULTS / 'benchmarks_v03.csv'

if not bench_v03_path.exists():
    print('  benchmarks_v03.csv not found.')
    print('  Run the Kaggle notebook first, download output/, copy here.')
    sys.exit(1)

bench_v03 = pd.read_csv(bench_v03_path)

# Normalise cv_map column name
bench_v03 = bench_v03.rename(columns={'cv_map_val': 'cv_map'})
for col in bench_prev.columns:
    if col not in bench_v03.columns:
        bench_v03[col] = ''

bench_all = pd.concat([bench_prev, bench_v03[bench_prev.columns]], ignore_index=True)
bench_all.to_csv(RESULTS / 'benchmarks.csv', index=False)
print(f'  benchmarks.csv updated ({len(bench_all)} rows)')
print(bench_all[['version', 'backbone', 'loss', 'cv_map']].to_string(index=False))


# ---------------------------------------------------------------------------
# 2. Verify v03 figures exist
# ---------------------------------------------------------------------------
print('\n[2/3] Checking v03 figures...')
expected = ['v03_training_curves.png', 'v03_per_identity_ap.png', 'v03_tsne.png']
for f in expected:
    p = FIGURES / f
    status = 'OK' if p.exists() else 'MISSING - copy from Kaggle output'
    print(f'  {f}: {status}')


# ---------------------------------------------------------------------------
# 3. Print progression summary
# ---------------------------------------------------------------------------
print('\n[3/3] Progression summary')

rows = bench_all.set_index('version')['cv_map']
for ver in ['v01', 'v02', 'v03']:
    if ver in rows.index:
        val = float(rows[ver])
        print(f'  {ver}: {val:.4f}')

if 'v03' in rows.index and 'v02' in rows.index:
    delta_v2_v3 = float(rows['v03']) - float(rows['v02'])
    delta_v1_v3 = float(rows['v03']) - float(rows['v01'])
    print(f'\n  v02 -> v03 delta: {delta_v2_v3:+.4f}  ({delta_v2_v3/float(rows["v02"])*100:+.1f}%)')
    print(f'  v01 -> v03 delta: {delta_v1_v3:+.4f}  ({delta_v1_v3/float(rows["v01"])*100:+.1f}%)')
    if delta_v2_v3 > 0.05:
        print('\n  ArcFace delivers a meaningful improvement over classifier fine-tuning.')
    elif delta_v2_v3 > 0:
        print('\n  Small improvement — consider tuning ArcFace margin/scale for v04.')
    else:
        print('\n  ArcFace underperformed — check for LR issues or try CosFace variant.')

print('\nNext: commit all v03 artifacts and push to GitHub.')
