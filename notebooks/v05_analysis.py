"""
v05 Local Analysis
==================
Run after downloading Kaggle outputs for v05.

Copy the downloaded files first:
  - figures/v05_per_identity_ap.png
  - figures/v05_tsne.png
  - results/benchmarks_v05.csv
  - submissions/submission_v05.csv

Then run:
    python v05_analysis.py

Prints TTA vs no-TTA comparison, per-identity deltas vs v04, and updates benchmarks.csv.
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
bench_v05_path = RESULTS / 'benchmarks_v05.csv'

if not bench_v05_path.exists():
    print('  benchmarks_v05.csv not found — run the Kaggle notebook first.')
    sys.exit(1)

bench_v05 = pd.read_csv(bench_v05_path)

# Map cv_map_val -> cv_map for unified tracking
bench_v05 = bench_v05.rename(columns={'cv_map_val': 'cv_map'})
for col in bench_prev.columns:
    if col not in bench_v05.columns:
        bench_v05[col] = ''

bench_all = pd.concat([bench_prev, bench_v05[bench_prev.columns]], ignore_index=True)
bench_all.to_csv(RESULTS / 'benchmarks.csv', index=False)
print(f'  Updated benchmarks.csv ({len(bench_all)} rows)')

print('\n[2/3] TTA vs no-TTA comparison:')
row = bench_v05.iloc[0]
no_tta = float(row.get('cv_map_val_noTTA', row['cv_map']))
tta    = float(row['cv_map'])
delta  = tta - no_tta
print(f'  Val mAP (no TTA): {no_tta:.4f}')
print(f'  Val mAP (TTA x5): {tta:.4f}  ({delta:+.4f} from TTA)')

# Compare to best v04
v04_row = bench_prev[bench_prev['version'] == 'v04_convnext']
if not v04_row.empty:
    v04_map = float(v04_row['cv_map'].values[0])
    delta_v4 = tta - v04_map
    print(f'\n  v04 ConvNeXt val mAP: {v04_map:.4f}')
    print(f'  v05 TTA delta vs v04: {delta_v4:+.4f}')
    if delta_v4 > 0:
        print('  v05 improves over v04 — augmentation + TTA helps.')
    else:
        print('  v05 regresses vs v04 — heavy augmentation may be over-regularising.')
        print('  Consider: lighter augmentation, or keep v04 as the base for v06.')

print('\n[3/3] Checking figures...')
for f in ['v05_per_identity_ap.png', 'v05_tsne.png']:
    p = FIGURES / f
    print(f'  {f}: {"OK" if p.exists() else "MISSING"}')

print('\nNext: commit all v05 artifacts and push to GitHub.')
