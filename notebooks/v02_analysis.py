"""
v02 Local Analysis
==================
Run after downloading Kaggle outputs for v02.

Copy the downloaded files to the right places first:
  - figures/v02_*.png         (from output zip)
  - results/benchmarks_v02.csv
  - submissions/submission_v02.csv

Then run:
    python v02_analysis.py

Produces a combined v01 vs v02 comparison figure and updates benchmarks.csv.
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

bench_v01 = pd.read_csv(RESULTS / 'benchmarks.csv')
bench_v02_path = RESULTS / 'benchmarks_v02.csv'

if not bench_v02_path.exists():
    print('  benchmarks_v02.csv not found.')
    print('  Run the Kaggle notebook first, download output/, copy here.')
    sys.exit(1)

bench_v02 = pd.read_csv(bench_v02_path)

# Append v02 row to main benchmarks.csv
bench_all = pd.concat([bench_v01, bench_v02], ignore_index=True)
bench_all.to_csv(RESULTS / 'benchmarks.csv', index=False)
print(f'  benchmarks.csv updated ({len(bench_all)} rows)')
print(bench_all[['version','backbone','cv_map']].to_string(index=False))

# ---------------------------------------------------------------------------
# 2. Verify v02 figures exist
# ---------------------------------------------------------------------------
print('\n[2/3] Checking v02 figures...')
expected = ['v02_training_curves.png', 'v02_per_identity_ap.png', 'v02_tsne.png']
for f in expected:
    p = FIGURES / f
    status = 'OK' if p.exists() else 'MISSING - copy from Kaggle output'
    print(f'  {f}: {status}')

# ---------------------------------------------------------------------------
# 3. Print summary comparison
# ---------------------------------------------------------------------------
print('\n[3/3] Summary')
v01_map = bench_v01[bench_v01.version == 'v01']['cv_map'].values
v02_map = bench_v02['cv_map_full'].values if 'cv_map_full' in bench_v02.columns \
          else bench_v02['cv_map'].values

if len(v01_map) and len(v02_map):
    delta = float(v02_map[0]) - float(v01_map[0])
    print(f'  v01 mAP: {float(v01_map[0]):.4f}')
    print(f'  v02 mAP: {float(v02_map[0]):.4f}')
    print(f'  Delta  : {delta:+.4f}  ({delta/float(v01_map[0])*100:+.1f}%)')
    print()
    if delta > 0.1:
        print('  Strong improvement from fine-tuning.')
    elif delta > 0.05:
        print('  Moderate improvement from fine-tuning.')
    else:
        print('  Small improvement — check for overfitting or try longer training.')

print('\nNext: commit all v02 artifacts and push to GitHub.')
