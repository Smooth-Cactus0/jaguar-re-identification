"""
v06 Local Analysis
==================
Run after downloading Kaggle outputs for v06.

Copy the downloaded files first:
  - figures/v06_feature_importance.png
  - figures/v06_per_identity_ap.png
  - figures/v06_tsne.png
  - results/benchmarks_v06.csv
  - submissions/submission_v06.csv

Then run:
    python v06_analysis.py

Prints cosine-only vs LightGBM comparison, feature importance summary,
and updates benchmarks.csv.
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
bench_v06_path = RESULTS / 'benchmarks_v06.csv'

if not bench_v06_path.exists():
    print('  benchmarks_v06.csv not found — run the Kaggle notebook first.')
    sys.exit(1)

bench_v06 = pd.read_csv(bench_v06_path)

# cv_map_lgbm is the primary metric for v06
bench_v06['cv_map'] = bench_v06['cv_map_lgbm']
for col in bench_prev.columns:
    if col not in bench_v06.columns:
        bench_v06[col] = ''

bench_all = pd.concat([bench_prev, bench_v06[bench_prev.columns]], ignore_index=True)
bench_all.to_csv(RESULTS / 'benchmarks.csv', index=False)
print(f'  Updated benchmarks.csv ({len(bench_all)} rows)')

print('\n[2/3] Cosine vs LightGBM comparison:')
row = bench_v06.iloc[0]
cos_map   = float(row.get('cv_map_cosine', 0))
lgbm_map  = float(row.get('cv_map_lgbm', row['cv_map']))
delta     = lgbm_map - cos_map
print(f'  Cosine-only val mAP : {cos_map:.4f}')
print(f'  LightGBM val mAP    : {lgbm_map:.4f}  ({delta:+.4f})')
print(f'  LightGBM iterations : {row.get("lgbm_best_iter", "?")}')

# Compare to v04 (best cosine baseline)
v04_row = bench_prev[bench_prev['version'] == 'v04_convnext']
if not v04_row.empty:
    v04_map = float(v04_row['cv_map'].values[0])
    print(f'\n  v04 ConvNeXt val mAP (reference): {v04_map:.4f}')
    print(f'  v06 LightGBM delta vs v04:        {lgbm_map - v04_map:+.4f}')

if delta > 0.01:
    print('\n  LightGBM delivers meaningful improvement — hybrid approach works!')
elif delta > 0:
    print('\n  Small improvement — LightGBM adds value, keep for ensemble.')
else:
    print('\n  LightGBM did not outperform cosine — deep features may already be optimal.')
    print('  Consider: more HC features (LBP, GLCM texture), PCA on embeddings, different neg ratio.')

print('\n[3/3] Checking figures...')
for f in ['v06_feature_importance.png', 'v06_per_identity_ap.png', 'v06_tsne.png']:
    p = FIGURES / f
    print(f'  {f}: {"OK" if p.exists() else "MISSING"}')

print('\nNext: commit all v06 artifacts and push to GitHub.')
