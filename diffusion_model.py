"""Conditional generation (rectified-flow) model: FusedModelFlow.

A PlainDiT flow transformer generates a VAE latent set conditioned on a point cloud
(``PCModel2``) or image (``ImgModel`` / DINOv2 features), which the frozen DualBrep VAE
then decodes into SDF + UDF fields. Used for the point-cloud and image generation checkpoints.

Inference (``inference``) integrates the flow ODE from noise to the latent, then decodes.
"""
import math
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import model as model_mod
from model import _inference
from dit import PlainDiT
import cond_model as cond_mod


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(next(self.parameters()).dtype)
        return self.mlp(t_freq)


class MM(nn.Module):
    """Multimodal conditioning: point cloud, image features, or unconditional token."""

    def __init__(self, v_hidden=1024, pc_model_name="PCModel2"):
        super().__init__()
        self.pc_fc = nn.Linear(768, v_hidden)
        self.img_fc = nn.Linear(1024, v_hidden)
        self.pcmodel = getattr(cond_mod, pc_model_name)()
        self.uncond_token = nn.Parameter(torch.rand(1, 1, v_hidden))

    def forward(self, v_data):
        if "pc" in v_data:
            return self.pc_fc(self.pcmodel(v_data["pc"]))          # [B, 1024, hidden]
        if "imgs_feat" in v_data:
            return self.img_fc(v_data["imgs_feat"])                # [B, tokens, hidden]
        bs = v_data["id_aug"].shape[0]
        return self.uncond_token.expand(bs, 256, -1)


class FusedModelFlow(nn.Module):
    def __init__(self, v_args):
        super().__init__()
        self.mm = MM(1024, v_args.get("pc_model_name", "PCModel2"))

        ModelClass = getattr(model_mod, v_args["vae_name"])
        self.vae_model = ModelClass(v_args)
        self.latent_dim = self.vae_model.out_dim
        # Load the (frozen) VAE weights used to train the flow.
        ckpt_path = str(v_args["vae_weights"])
        if ckpt_path.startswith("s3"):
            from cloudpathlib import S3Path
            with tempfile.TemporaryDirectory() as tmp:
                local = Path(tmp) / Path(ckpt_path).name
                S3Path(ckpt_path).download_to(str(local))
                weights = torch.load(str(local), map_location="cpu", weights_only=False)["state_dict"]
        else:
            weights = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        weights = {k.replace("model._orig_mod.", "").replace("model.", ""): v for k, v in weights.items()}
        self.vae_model.load_state_dict(weights, strict=True)
        self.vae_model.eval()
        self.vae_model.requires_grad_(False)

        self.model = PlainDiT(32, 32, 1024, v_args["depth"], v_args["head"], xtrans=True)
        self.time_embed = TimestepEmbedder(1024)
        self.t_block = nn.Sequential(
            nn.Linear(1024, 1024), nn.GELU(),
            nn.Linear(1024, 1024), nn.Linear(1024, 6 * 1024),
        )
        self.global_scale = v_args.get("global_scale", 2.0)
        self._imgmodel = None  # DINOv2, built lazily for image conditioning

    @property
    def imgmodel(self):
        if self._imgmodel is None:
            self._imgmodel = cond_mod.ImgModel().to(next(self.model.parameters()).device).eval()
            self._imgmodel.requires_grad_(False)
        return self._imgmodel

    @torch.no_grad()
    def inference(self, v_data, steps=50, temperature: float = -1):
        # Image conditioning: encode raw images to DINOv2 features first.
        if "imgs" in v_data and "imgs_feat" not in v_data:
            v_data["imgs_feat"] = self.imgmodel(v_data["imgs"])

        if "pc" in v_data:
            bs = len(v_data["pc"])
        elif "imgs_feat" in v_data:
            bs = len(v_data["imgs_feat"])
        else:
            bs = v_data["id_aug"].shape[0]
        cond = self.mm(v_data)
        device, dtype = cond.device, cond.dtype

        gt_face_z = None
        if "sample_points_surfaces" in v_data and "sample_points_edges" in v_data:
            gt_face_z, _ = self.vae_model.encode(v_data, v_test=True)
            gt_face_z = gt_face_z / self.global_scale

        n_tokens = self.latent_dim if self.vae_model.is_fps is False else self.latent_dim * 2
        noise = torch.randn((bs, n_tokens, 32), device=device, dtype=dtype)
        times_int = (torch.linspace(0, 1.0, steps=steps, device=device, dtype=dtype) * 999).long()
        times = times_int / 1000
        distances = (times[1:] - times[:-1]).abs()

        latents = noise
        for t_int, dt in zip(times_int[:-1], distances):
            time_embed = self.t_block(self.time_embed(t_int[None]))
            pred_flow = self.model(latents, cond, time_embed.repeat(bs, 1))
            if temperature > 0:
                pred_uncond = self.model(latents, torch.zeros_like(cond), time_embed.repeat(bs, 1))
                pred_flow = pred_uncond + temperature * (pred_flow - pred_uncond)
            latents = latents - pred_flow * dt

        pred_decoded = self.vae_model.decode(latents * self.global_scale)
        gt_decoded = self.vae_model.decode(gt_face_z * self.global_scale) if gt_face_z is not None else None
        return pred_decoded, gt_decoded
