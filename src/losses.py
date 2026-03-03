"""
src/losses.py — Metric learning loss functions for jaguar re-ID.

Implements ArcFace (Deng et al., 2019) and CosFace (Wang et al., 2018).
Both are angular-margin softmax variants that directly optimise the
embedding space for similarity rather than for classification accuracy.

Usage (v03+):
    arcface = ArcFaceLoss(embedding_dim=512, num_classes=31).to(device)
    # In training loop:
    embs = model(imgs)          # (B, 512), L2-normalised
    loss = arcface(embs, labels)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss for Deep Face Recognition.
    https://arxiv.org/abs/1801.07698

    Adds an angular margin m to the angle between the embedding and its
    true class weight vector, forcing tighter intra-class clustering and
    larger inter-class margins in hyperspherical embedding space.

    The loss is:
        L = -log( exp(s * cos(theta_y + m)) /
                  (exp(s * cos(theta_y + m)) + sum_j exp(s * cos(theta_j))) )

    where theta_y is the angle to the true-class prototype.

    Args:
        embedding_dim:  dimensionality of L2-normalised input embeddings
        num_classes:    number of identities (31 for jaguar re-ID)
        margin:         angular margin in radians (default 0.5 ≈ 28.6°)
        scale:          feature scale factor s (default 30 for small datasets)
    """

    def __init__(self, embedding_dim: int, num_classes: int,
                 margin: float = 0.5, scale: float = 30.0):
        super().__init__()
        self.s = scale
        self.m = margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute trig constants for the margin transform
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th    = math.cos(math.pi - margin)   # cos(pi - m)
        self.mm    = math.sin(math.pi - margin) * margin  # sin(pi-m)*m — lower-bound fallback

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) L2-normalised feature vectors
            labels:     (B,)  integer class labels

        Returns:
            scalar ArcFace cross-entropy loss
        """
        # L2-normalise class weight vectors — ArcFace requires unit-norm prototypes
        w = F.normalize(self.weight, p=2, dim=1)  # (C, D)

        # Cosine similarity: cos(theta_j) for each class j
        cosine = F.linear(embeddings, w)  # (B, C)

        # For the true class: compute cos(theta + m) = cos*cos_m - sin*sin_m
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=1e-6))
        phi  = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)

        # Numerical guard: when theta > pi - m, the margin pushes past pi.
        # Fall back to cos(theta) - mm to maintain a monotonic lower bound.
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encode labels to select the true-class column
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        # Replace true-class logit with margin-adjusted value; scale all logits
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s

        return F.cross_entropy(logits, labels.long())


class CosFaceLoss(nn.Module):
    """
    CosFace: Large Margin Cosine Loss.
    https://arxiv.org/abs/1801.09414

    Subtracts a cosine margin m directly from the true-class logit.
    Simpler than ArcFace (no arccos/arcsin) but similarly effective.

    L = -log( exp(s * (cos(theta_y) - m)) /
              (exp(s * (cos(theta_y) - m)) + sum_j exp(s * cos(theta_j))) )

    Args:
        embedding_dim:  dimensionality of L2-normalised input embeddings
        num_classes:    number of identities
        margin:         cosine margin (default 0.35)
        scale:          feature scale factor s (default 30)
    """

    def __init__(self, embedding_dim: int, num_classes: int,
                 margin: float = 0.35, scale: float = 30.0):
        super().__init__()
        self.s = scale
        self.m = margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) L2-normalised feature vectors
            labels:     (B,)  integer class labels

        Returns:
            scalar CosFace cross-entropy loss
        """
        w      = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, w)                # (B, C)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        logits = (cosine - one_hot * self.m) * self.s   # subtract margin from true class
        return F.cross_entropy(logits, labels.long())
