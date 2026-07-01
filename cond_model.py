"""Conditioning encoders for the generation (flow) model.

* ``PCModel2``  – point-cloud encoder (Gaussian Fourier-feature embedder + cross/self
                  attention) producing a 1024-token conditioning sequence
                  (input_proj 262 = GaussianEmbedder.out_dim + 3).
* ``ImgModel``  – frozen DINOv2 (ViT-L/14 + registers) image encoder for image conditioning.
                  Loaded lazily because it downloads weights from torch.hub on first use.

``FourierEmbedder`` / ``MLP_gelu`` / ``ResidualCrossAttentionBlock_gelu`` are reused from
``model.py`` to avoid duplication.
"""
import torch
from torch import nn
from torchvision import transforms as T

from model import ResidualCrossAttentionBlock_gelu


# Image preprocessing for DINOv2 (no resize — renders are already 518x518 = 37*14).
img_transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class GaussianEmbedder(nn.Module):
    """Random Gaussian Fourier features: out_dim = 2*mapping_size + input_dim."""

    def __init__(self, input_dim: int = 3, mapping_size: int = 128, scale: float = 10.0):
        super().__init__()
        self.input_dim = input_dim
        # B is a persistent buffer -> loaded from the checkpoint (random init is overwritten).
        self.register_buffer("B", torch.randn((input_dim, mapping_size)) * scale)
        self.out_dim = mapping_size * 2 + input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = (2.0 * torch.pi * x) @ self.B
        embed = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return torch.cat([x, embed], dim=-1)


class PCModel2(nn.Module):
    """Point-cloud conditioning encoder: pc [B, N, 6] -> tokens [B, 1024, 768]."""

    def __init__(self):
        super().__init__()
        self.embedder = GaussianEmbedder()
        self.input_proj = nn.Linear(self.embedder.out_dim + 3, 768)
        self.cross_attn = ResidualCrossAttentionBlock_gelu(768, 12)
        self_layer = nn.TransformerEncoderLayer(768, 12, batch_first=True, norm_first=True, dropout=0.0)
        self.self_attn = nn.TransformerEncoder(self_layer, num_layers=8, norm=nn.LayerNorm(768))

    def forward(self, pc):
        bs, n_points, _ = pc.shape
        feat = self.input_proj(torch.cat([pc[..., 3:], self.embedder(pc[..., :3])], dim=-1))
        # 1024 random query seeds attend over the full point set (training used random, not FPS).
        idx = torch.randperm(n_points, device=feat.device)[:1024][None]
        query = feat.gather(1, idx.unsqueeze(-1).expand(bs, -1, feat.shape[-1]))
        return self.self_attn(self.cross_attn(query, feat))


class ImgModel(nn.Module):
    """Frozen DINOv2 ViT-L/14 (+registers) image encoder -> [B, num_patches+1, 1024].

    NOTE: instantiating this downloads the DINOv2 weights from torch.hub
    (facebookresearch/dinov2). Only constructed when image conditioning is used.
    """

    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')

    def forward(self, imgs):
        feat = self.model.get_intermediate_layers(imgs, return_class_token=True)[0]
        return torch.cat((feat[0], feat[1][:, None]), dim=1)  # patch tokens + cls token
