"""
src/inference.py — Embedding extraction and submission generation.

Decoupled from model definition so any backbone can be dropped in.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_embeddings(model: torch.nn.Module,
                       dataset,
                       batch_size: int = 64,
                       device: str = 'cpu',
                       l2_normalise: bool = True,
                       desc: str = 'Extracting') -> np.ndarray:
    """
    Run model forward pass over a dataset and return embeddings as numpy array.

    Args:
        model:        PyTorch model in eval mode, output is (B, D) embeddings
        dataset:      JaguarDataset (or any Dataset returning images)
        batch_size:   images per forward pass
        device:       'cpu' or 'cuda'
        l2_normalise: normalise to unit sphere for cosine similarity via dot product
        desc:         tqdm progress bar label

    Returns:
        (N, D) float32 numpy array
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=(device == 'cuda'))
    model  = model.to(device).eval()

    all_embs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=True):
            # Dataset may return (img, label) or just img
            if isinstance(batch, (list, tuple)):
                imgs = batch[0]
            else:
                imgs = batch
            imgs = imgs.to(device)
            embs = model(imgs)            # (B, D)
            if l2_normalise:
                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            all_embs.append(embs.cpu().numpy())

    return np.concatenate(all_embs, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Submission generation
# ---------------------------------------------------------------------------

def make_submission(test_df,
                    test_embeddings: np.ndarray,
                    filename_to_idx: dict,
                    output_path,
                    score_transform=None) -> None:
    """
    Generate a Kaggle submission CSV from test embeddings.

    Args:
        test_df:           DataFrame with columns [row_id, query_image, gallery_image]
        test_embeddings:   (N_test, D) L2-normalised embeddings for test images
        filename_to_idx:   maps 'test_XXXX.png' -> row index in test_embeddings
        output_path:       path to write submission CSV
        score_transform:   optional callable to post-process similarity scores
    """
    import pandas as pd

    query_idx   = test_df['query_image'].map(filename_to_idx).values
    gallery_idx = test_df['gallery_image'].map(filename_to_idx).values

    # Batch cosine similarity: dot product of L2-normalised vectors
    q_embs = test_embeddings[query_idx]    # (N_pairs, D)
    g_embs = test_embeddings[gallery_idx]  # (N_pairs, D)
    similarities = (q_embs * g_embs).sum(axis=1)  # (N_pairs,)

    # Shift from [-1, 1] to [0, 1] for submission
    similarities = (similarities + 1.0) / 2.0

    if score_transform is not None:
        similarities = score_transform(similarities)

    submission = pd.DataFrame({
        'row_id':     test_df['row_id'].values,
        'similarity': similarities,
    })
    submission.to_csv(output_path, index=False)
    print(f'  Submission saved -> {output_path}')
    print(f'  Score range: {similarities.min():.4f} - {similarities.max():.4f}')
    print(f'  Score mean:  {similarities.mean():.4f}')
