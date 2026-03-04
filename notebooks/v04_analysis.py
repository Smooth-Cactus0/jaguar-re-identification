"""
v04 Local Analysis
==================
Run after downloading Kaggle outputs for v04.

Copy the downloaded files first:
  - figures/v04_backbone_comparison.png
  - results/benchmarks_v04.csv
  - submissions/submission_v04.csv

Then run:
    python v04_analysis.py

Prints backbone comparison summary and updates benchmarks.csv.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import pandas as pd
from pathlib import Path

ROOT    = Path(__file__).parent.parent
RESULTS = ROOT / 'results'
FIGURES = ROOT / 'figures'

print('[1/3] Loading benchmark results...')

bench_prev     = pd.read_csv(RESULTS / 'benchmarks.csv')
bench_v04_path = RESULTS / 'benchmarks_v04.csv'

if not bench_v04_path.exists():
    print('  benchmarks_v04.csv not found — run the Kaggle notebook first.')
    sys.exit(1)

bench_v04 = pd.read_csv(bench_v04_path).rename(columns={'cv_map_val': 'cv_map'})
for col in bench_prev.columns:
    if col not in bench_v04.columns:
        bench_v04[col] = ''

bench_all = pd.concat([bench_prev, bench_v04[bench_prev.columns]], ignore_index=True)
bench_all.to_csv(RESULTS / 'benchmarks.csv', index=False)
print(f'  Updated benchmarks.csv ({len(bench_all)} rows)')

print('\n[2/3] Backbone comparison:')
print(f'  {"Backbone":<35} {"Val mAP":>8} {"Full mAP":>10}')
print(f'  {"-"*55}')
for _, row in bench_v04.iterrows():
    print(f'  {row["backbone"]:<35} {row["cv_map"]:>8.4f} {row.get("cv_map_full", ""):>10}')

print('\n[3/3] Checking figures...')
for f in ['v04_backbone_comparison.png']:
    p = FIGURES / f
    print(f'  {f}: {"OK" if p.exists() else "MISSING"}')

print('\nNext: commit all v04 artifacts and push to GitHub.')
