"""
Jaguar Re-ID: Exploratory Data Analysis Script
Run from the notebooks/ directory:
    python run_eda.py

Saves all figures to ../figures/
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import imagehash
import warnings
warnings.filterwarnings('ignore')

# -- Paths ------------------------------------------------------------------
ROOT      = Path(__file__).parent.parent
TRAIN_DIR = ROOT / 'train' / 'train'
TEST_DIR  = ROOT / 'test'  / 'test'
FIGURES   = ROOT / 'figures'
FIGURES.mkdir(exist_ok=True)

# -- Plot style -------------------------------------------------------------
plt.rcParams.update({
    'figure.dpi': 120,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})
PALETTE = sns.color_palette('tab20', 31)

# -- Helpers ----------------------------------------------------------------
def load_thumb(path, size=(128, 128)):
    """Load image as RGB thumbnail composited on white background."""
    pil = Image.open(path).convert('RGBA')
    bg  = Image.new('RGBA', size, (255, 255, 255, 255))
    pil.thumbnail(size, Image.LANCZOS)
    offset = ((size[0] - pil.width) // 2, (size[1] - pil.height) // 2)
    bg.paste(pil, offset, mask=pil)
    return bg.convert('RGB')

def save(name):
    path = FIGURES / name
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  Saved -> figures/{name}')
    plt.close()

# ==========================================================================
print('\n' + '='*60)
print('JAGUAR RE-ID: EDA')
print('='*60)

# -- 1. Load data -----------------------------------------------------------
print('\n[1/9] Loading data...')
train_df = pd.read_csv(ROOT / 'train.csv')
test_df  = pd.read_csv(ROOT / 'test.csv')
identity_counts = train_df['ground_truth'].value_counts()
test_images = pd.unique(test_df[['query_image', 'gallery_image']].values.ravel())

print(f'  Train : {len(train_df):,} images, {train_df.ground_truth.nunique()} identities')
print(f'  Test  : {len(test_df):,} pairs, {len(test_images)} unique images')
print(f'  Count range: {identity_counts.min()} - {identity_counts.max()} imgs/identity')

# -- 2. Class distribution --------------------------------------------------
print('\n[2/9] Class distribution...')
fig, ax = plt.subplots(figsize=(14, 5))
colors = [PALETTE[i] for i in range(len(identity_counts))]
bars = ax.bar(identity_counts.index, identity_counts.values,
              color=colors, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, identity_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            str(val), ha='center', va='bottom', fontsize=7, fontweight='bold')
mean_count = identity_counts.mean()
ax.axhline(mean_count, color='crimson', linestyle='--', linewidth=1.2,
           label=f'Mean: {mean_count:.0f}')
ax.set_xlabel('Jaguar Identity', fontsize=11)
ax.set_ylabel('Number of Training Images', fontsize=11)
ax.set_title('Training Set: Images per Jaguar Identity', fontsize=13, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.legend(fontsize=10)
plt.tight_layout()
save('eda_01_class_distribution.png')

# -- 3. Identity gallery (one image per jaguar) -----------------------------
print('\n[3/9] Identity gallery...')
identities = identity_counts.index.tolist()
n_cols, n_ids = 6, len(identities)
n_rows = (n_ids + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.5))
fig.suptitle('One Representative Image per Jaguar Identity', fontsize=13,
             fontweight='bold', y=1.01)
for idx, identity in enumerate(identities):
    row, col = divmod(idx, n_cols)
    ax = axes[row, col]
    sample_file = train_df[train_df.ground_truth == identity]['filename'].iloc[0]
    ax.imshow(load_thumb(TRAIN_DIR / sample_file))
    ax.set_title(f'{identity}\n({identity_counts[identity]} imgs)', fontsize=8,
                 fontweight='bold')
    ax.axis('off')
for idx in range(n_ids, n_rows * n_cols):
    row, col = divmod(idx, n_cols)
    axes[row, col].axis('off')
plt.tight_layout()
save('eda_02_identity_gallery.png')

# -- 4. Intra-identity variability ------------------------------------------
print('\n[4/9] Intra-identity variability (top 10 identities)...')
top10  = identity_counts.head(10).index.tolist()
n_show = 5

fig, axes = plt.subplots(len(top10), n_show, figsize=(n_show * 2.2, len(top10) * 2.4))
fig.suptitle('Intra-Identity Variability: 5 Images per Jaguar (Top 10 by Count)',
             fontsize=12, fontweight='bold', y=1.005)
for row, identity in enumerate(top10):
    files   = train_df[train_df.ground_truth == identity]['filename'].tolist()
    indices = np.linspace(0, len(files)-1, n_show, dtype=int)
    for col, fname in enumerate([files[i] for i in indices]):
        ax = axes[row, col]
        ax.imshow(load_thumb(TRAIN_DIR / fname))
        ax.axis('off')
        if col == 0:
            ax.set_ylabel(identity, fontsize=9, fontweight='bold',
                          rotation=0, labelpad=50, va='center')
plt.tight_layout()
save('eda_03_intra_identity_variability.png')

# -- 5. Image properties ----------------------------------------------------
print('\n[5/9] Image properties (scanning all train images)...')
records = []
for fname in train_df['filename']:
    with Image.open(TRAIN_DIR / fname) as img:
        w, h = img.size
        mode = img.mode
    records.append({'filename': fname, 'width': w, 'height': h,
                    'mode': mode, 'aspect': w/h, 'mpx': w*h/1e6})
props = pd.DataFrame(records).merge(train_df, on='filename')

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(props['width'],  bins=40, color='steelblue',      edgecolor='white', linewidth=0.5)
axes[0].set_xlabel('Width (px)');  axes[0].set_ylabel('Count')
axes[0].set_title('Image Width Distribution')
axes[1].hist(props['height'], bins=40, color='coral',          edgecolor='white', linewidth=0.5)
axes[1].set_xlabel('Height (px)'); axes[1].set_title('Image Height Distribution')
axes[2].hist(props['aspect'], bins=40, color='mediumseagreen', edgecolor='white', linewidth=0.5)
axes[2].axvline(1.0, color='red', linestyle='--', linewidth=1, label='Square')
axes[2].set_xlabel('Aspect Ratio (W/H)'); axes[2].set_title('Aspect Ratio Distribution')
axes[2].legend()
plt.suptitle('Training Image Properties', fontsize=12, fontweight='bold')
plt.tight_layout()
save('eda_04_image_properties.png')

print(f'  Modes: {props["mode"].value_counts().to_dict()}')
print(f'  Width : {props["width"].min()}-{props["width"].max()} px')
print(f'  Height: {props["height"].min()}-{props["height"].max()} px')
print(f'  Aspect: {props["aspect"].min():.2f}-{props["aspect"].max():.2f}')
print(f'  Portrait (H>W): {(props["height"]>props["width"]).sum()} | '
      f'Landscape: {(props["width"]>props["height"]).sum()}')

# -- 6. Near-duplicate detection --------------------------------------------
print('\n[6/9] Near-duplicate detection (perceptual hashing)...')
hashes = {}
for i, fname in enumerate(train_df['filename']):
    if i % 300 == 0:
        print(f'  Hashing {i}/{len(train_df)}...')
    img = Image.open(TRAIN_DIR / fname).convert('RGB')
    hashes[fname] = imagehash.phash(img)

print(f'  Hashing {len(train_df)}/{len(train_df)}... done.')
print('  Computing pairwise distances...')

filenames = list(hashes.keys())
id_map    = train_df.set_index('filename')['ground_truth'].to_dict()
near_dups = []
for i in range(len(filenames)):
    for j in range(i+1, len(filenames)):
        dist = hashes[filenames[i]] - hashes[filenames[j]]
        if dist <= 8:
            near_dups.append({
                'img1': filenames[i], 'img2': filenames[j], 'hamming': dist,
                'id1': id_map[filenames[i]], 'id2': id_map[filenames[j]],
            })

dups_df = pd.DataFrame(near_dups) if near_dups else pd.DataFrame(
    columns=['img1','img2','hamming','id1','id2'])

print(f'  Near-duplicate pairs (hamming <= 8): {len(dups_df)}')
if len(dups_df) > 0:
    same_id = (dups_df['id1'] == dups_df['id2']).sum()
    diff_id = (dups_df['id1'] != dups_df['id2']).sum()
    print(f'  Same identity: {same_id}  |  Different identity: {diff_id}')

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    dup_counts = dups_df[dups_df['id1'] == dups_df['id2']]['id1'].value_counts()
    dup_counts.plot(kind='bar', ax=axes[0], color='slateblue', edgecolor='white')
    axes[0].set_title('Near-Duplicate Pairs per Identity', fontweight='bold')
    axes[0].set_xlabel('Jaguar Identity'); axes[0].set_ylabel('Pairs')
    axes[0].tick_params(axis='x', rotation=45)
    axes[1].hist(dups_df['hamming'], bins=range(0, 10), color='tomato',
                 edgecolor='white', align='left')
    axes[1].set_title('Hamming Distance Distribution', fontweight='bold')
    axes[1].set_xlabel('Hamming Distance (0=identical)'); axes[1].set_ylabel('Count')
    axes[1].set_xticks(range(0, 9))
    plt.suptitle('Near-Duplicate Analysis (pHash threshold = 8)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    save('eda_05_near_duplicates.png')

    ex = dups_df.iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    axes[0].imshow(load_thumb(TRAIN_DIR / ex['img1'], (256, 256)))
    axes[0].set_title(f"{ex['img1']}\n{ex['id1']}", fontsize=9); axes[0].axis('off')
    axes[1].imshow(load_thumb(TRAIN_DIR / ex['img2'], (256, 256)))
    axes[1].set_title(f"{ex['img2']}\n{ex['id2']} | d={ex['hamming']}", fontsize=9)
    axes[1].axis('off')
    plt.suptitle('Example Near-Duplicate Pair', fontweight='bold')
    plt.tight_layout()
    save('eda_05b_near_duplicate_example.png')
else:
    print('  No near-duplicates found - dataset is clean.')

# -- 7+8. Color + Foreground in ONE pass ------------------------------------
# Merge both loops to open each image only once.
# thumbnail() avoids decoding the full PNG resolution before downsample.
print('\n[7-8/9] Color + foreground analysis (single pass, 128px thumbnails)...')
THUMB = (128, 128)
color_records = []
fg_records    = []
for i, (fname, identity) in enumerate(zip(train_df['filename'], train_df['ground_truth'])):
    if i % 400 == 0:
        print(f'  Processing {i}/{len(train_df)}...')
    pil = Image.open(TRAIN_DIR / fname).convert('RGBA')
    pil.thumbnail(THUMB, Image.LANCZOS)
    img = np.array(pil)

    alpha_mask = img[:, :, 3] > 10
    fg_frac    = alpha_mask.mean()
    fg_records.append({'filename': fname, 'identity': identity, 'fg_frac': fg_frac})

    if alpha_mask.sum() >= 100:
        rgb = img[:, :, :3].astype(float)
        r = rgb[:,:,0][alpha_mask]
        g = rgb[:,:,1][alpha_mask]
        b = rgb[:,:,2][alpha_mask]
        brightness = np.sqrt(r**2 + g**2 + b**2).mean()
        color_records.append({'identity': identity,
                              'R': r.mean(), 'G': g.mean(), 'B': b.mean(),
                              'brightness': brightness})

print(f'  Processing {len(train_df)}/{len(train_df)}... done.')

# Color plot
color_df  = pd.DataFrame(color_records)
id_colors = color_df.groupby('identity')[['R','G','B','brightness']].mean()
id_sorted = id_colors.sort_values('brightness')

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
x, w = np.arange(len(id_sorted)), 0.25
axes[0].bar(x-w, id_sorted['R'], w, color='tomato',         label='R', alpha=0.85)
axes[0].bar(x,   id_sorted['G'], w, color='mediumseagreen', label='G', alpha=0.85)
axes[0].bar(x+w, id_sorted['B'], w, color='steelblue',      label='B', alpha=0.85)
axes[0].set_xticks(x)
axes[0].set_xticklabels(id_sorted.index, rotation=45, ha='right', fontsize=8)
axes[0].set_ylabel('Mean pixel value (foreground only)')
axes[0].set_title('Mean RGB per Identity (sorted by brightness)', fontweight='bold')
axes[0].legend()

means = color_df.groupby('identity')['brightness'].mean().sort_values()
stds  = color_df.groupby('identity')['brightness'].std().reindex(means.index)
axes[1].errorbar(range(len(means)), means.values, yerr=stds.values,
                 fmt='o', color='darkorange', ecolor='gray',
                 elinewidth=1, capsize=3, markersize=6)
axes[1].set_xticks(range(len(means)))
axes[1].set_xticklabels(means.index, rotation=45, ha='right', fontsize=8)
axes[1].set_ylabel('Brightness (RMS, foreground pixels)')
axes[1].set_title('Brightness per Identity (mean +/- std)', fontweight='bold')
plt.suptitle('Per-Identity Color Characteristics', fontsize=12, fontweight='bold')
plt.tight_layout()
save('eda_06_color_characteristics.png')
print(f'  Brightness range: {means.min():.1f} - {means.max():.1f}')

# Foreground plot
print('\n[8/9] Foreground coverage plot...')
fg_df    = pd.DataFrame(fg_records)
fg_by_id = fg_df.groupby('identity')['fg_frac'].mean().sort_values()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].hist(fg_df['fg_frac'] * 100, bins=30, color='teal', edgecolor='white')
axes[0].set_xlabel('Foreground Coverage (%)'); axes[0].set_ylabel('Count')
axes[0].set_title('Jaguar Foreground Coverage Distribution', fontweight='bold')
fg_by_id.plot(kind='bar', ax=axes[1], color='teal', edgecolor='white')
axes[1].set_title('Mean Foreground Coverage per Identity', fontweight='bold')
axes[1].set_xlabel('Jaguar Identity'); axes[1].set_ylabel('Mean FG Fraction')
axes[1].tick_params(axis='x', rotation=45)
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
plt.suptitle('Foreground Coverage (Alpha Channel)', fontsize=12, fontweight='bold')
plt.tight_layout()
save('eda_07_foreground_coverage.png')
print(f'  Mean coverage: {fg_df["fg_frac"].mean():.1%}')
print(f'  Low coverage (<30%): {(fg_df["fg_frac"] < 0.3).sum()} images')

# -- 9. Summary -------------------------------------------------------------
print('\n[9/9] Summary')
n_pos_ratio = (identity_counts.mean() - 1) / (len(test_images) - 1)
print()
print('=' * 60)
print('KEY FINDINGS')
print('=' * 60)
print(f'  Imbalance ratio       : {identity_counts.max()/identity_counts.min():.1f}x')
print(f'  Near-duplicates found : {len(dups_df)} pairs')
print(f'  Image mode            : RGBA (alpha = foreground mask)')
print(f'  Aspect ratio range    : {props["aspect"].min():.2f} - {props["aspect"].max():.2f}')
print(f'  Est. positive ratio   : {n_pos_ratio:.1%} of test pairs')
print(f'  Mean FG coverage      : {fg_df["fg_frac"].mean():.1%}')
print()
print('MODELLING IMPLICATIONS')
print('  - Use identity-stratified GroupKFold for CV splits')
print('  - Composite RGBA onto black/white bg before CNN input')
print('  - Pad to square aspect ratio (not stretch) before resizing')
print('  - Track per-identity AP -- rare jaguars (13 imgs) matter equally')
print('  - Flip augmentation: use with care (left/right flanks differ)')
print()
print(f'Figures saved to: {FIGURES}')
print('Done.')
