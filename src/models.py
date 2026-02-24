"""
src/models.py — Backbone factory for all versions.

Uses timm for consistent access to EfficientNet, ConvNeXt, ViT, etc.
The EmbeddingModel wrapper strips the classifier head and returns raw
pooled features — drop-in ready for metric learning heads in v03+.
"""

import torch
import torch.nn as nn
import timm


class EmbeddingModel(nn.Module):
    """
    Pretrained backbone with classifier head replaced by identity.
    Output: (B, embedding_dim) float32 tensor, UNNORMALISED.
    Normalisation is handled in inference.py.

    Args:
        backbone_name:  timm model name (e.g. 'efficientnet_b0')
        embedding_dim:  output dimension; None = use backbone's native dim
        pretrained:     load ImageNet weights
        drop_rate:      dropout before embedding (0 for v01 frozen baseline)
    """

    def __init__(self, backbone_name: str = 'efficientnet_b0',
                 embedding_dim: int = None,
                 pretrained: bool = True,
                 drop_rate: float = 0.0):
        super().__init__()

        # Create backbone with no head (num_classes=0 returns pooled features)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,          # removes classifier, outputs pooled features
            drop_rate=drop_rate,
        )
        native_dim = self.backbone.num_features

        if embedding_dim is not None and embedding_dim != native_dim:
            self.proj = nn.Sequential(
                nn.Linear(native_dim, embedding_dim),
            )
            self.out_dim = embedding_dim
        else:
            self.proj    = nn.Identity()
            self.out_dim = native_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)   # (B, native_dim)
        return self.proj(feat)    # (B, out_dim)

    def freeze_backbone(self):
        """Freeze all backbone parameters (v01 baseline mode)."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        print(f'  Backbone frozen ({sum(p.numel() for p in self.backbone.parameters()):,} params)')

    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning (v02+)."""
        for p in self.backbone.parameters():
            p.requires_grad_(True)
        print(f'  Backbone unfrozen')


def get_model(backbone_name: str = 'efficientnet_b0',
              embedding_dim: int = None,
              pretrained: bool = True,
              freeze: bool = False) -> EmbeddingModel:
    """
    Factory function. Returns ready-to-use EmbeddingModel.

    Common choices by version:
      v01: backbone_name='efficientnet_b0', freeze=True
      v02: backbone_name='efficientnet_b0', freeze=False
      v03: same + ArcFace head added externally
      v04: 'efficientnetv2_s', 'convnext_small', 'vit_small_patch16_224'
    """
    model = EmbeddingModel(backbone_name, embedding_dim, pretrained)
    if freeze:
        model.freeze_backbone()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f'  Model: {backbone_name}  |  '
          f'dim: {model.out_dim}  |  '
          f'trainable: {n_trainable:,} / {n_total:,}')
    return model
