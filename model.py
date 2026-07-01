"""DualBrep VAE model (inference).

The ``DualVAE`` autoencoder used in Make-A-BRep. It keeps the modules and forward logic needed to encode a
sampled point cloud (surface / edge / voronoi points) into a latent set and decode
SDF + UDF fields, so the released checkpoint loads with zero missing/unexpected keys.

The decoder predicts two channels per query point:
  * channel 0: SDF (truncated, scaled to roughly [-1, 1]; surface at 0)
  * channel 1: UDF to BRep edges (scaled to roughly [0, 1])
Multiply by ``dataset.clip_value`` (0.015) to recover metric distances.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from scipy.spatial.transform import Rotation
from torch_cluster import fps


# --------------------------------------------------------------------------- #
# Embedders / attention blocks
# --------------------------------------------------------------------------- #
class FourierEmbedder(nn.Module):
    def __init__(self, num_freqs: int = 6, logspace: bool = True, input_dim: int = 3,
                 include_input: bool = True, include_pi: bool = True, denominator=1) -> None:
        super().__init__()
        self.denominator = denominator
        if logspace:
            frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        else:
            frequencies = torch.linspace(1.0, 2.0 ** (num_freqs - 1), num_freqs, dtype=torch.float32)
        if include_pi:
            frequencies *= torch.pi
        frequencies /= denominator
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs
        self.out_dim = self.get_dims(input_dim)

    def get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        return input_dim * (self.num_freqs * 2 + temp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
            if self.include_input:
                return torch.cat((x, embed.sin() * self.denominator, embed.cos() * self.denominator), dim=-1)
            return torch.cat((embed.sin() * self.denominator, embed.cos() * self.denominator), dim=-1)
        return x


class MLP_gelu(nn.Module):
    def __init__(self, *, width: int):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4)
        self.c_proj = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class ResidualCrossAttentionBlock_gelu(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ln_1 = nn.LayerNorm(width)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = MLP_gelu(width=width)
        self.ln_3 = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        data = self.ln_2(data)
        x = x + self.attn(self.ln_1(x), data, data, need_weights=False)[0]
        x = x + self.mlp(self.ln_3(x))
        return x


# --------------------------------------------------------------------------- #
# Geometric helpers
# --------------------------------------------------------------------------- #
def rotate(v_points, v_matrix):
    """Apply a (bs, 4, 4) affine to points (and, if present, their normals)."""
    shape = v_points.shape
    with_normal = shape[-1] > 3
    points = torch.cat([v_points[..., :3], torch.ones_like(v_points[..., :1])], dim=-1)
    a_points = (v_matrix @ points[..., :4].permute(0, 2, 1)).permute(0, 2, 1)[..., :3]
    if with_normal:
        normals = v_points[..., 3:6]
        opoints = points[..., :3] + normals
        opoints = torch.cat([opoints, torch.ones_like(opoints[..., :1])], dim=-1)
        a_opoints = (v_matrix @ opoints.permute(0, 2, 1)).permute(0, 2, 1)[..., :3]
        a_normals = a_opoints - a_points
        a_points = torch.cat([a_points, a_normals], dim=-1)
    return a_points.reshape(shape)


def random_matrix(bs, device, dtype):
    q = torch.randn(bs, 4, device=device, dtype=dtype)
    q = q / torch.norm(q, dim=1, keepdim=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w
    row0 = torch.stack([1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)], dim=1)
    row1 = torch.stack([2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)], dim=1)
    row2 = torch.stack([2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)], dim=1)
    return torch.stack([row0, row1, row2], dim=1)


def augment_submitted(v_data, v_matrix, v_test):
    """Normalize SDF/UDF targets and optionally rotate the sample/query points.

    ``is_aug`` controls the transform applied uniformly to the batch:
      0 -> identity (canonical pose, used for plain reconstruction)
      1 -> one of the 24 octahedral rotations selected by ``id_aug``
      2 -> a random rotation
      3 -> mixed (rotation / scale / shift)
    """
    clip_value = v_data["clip_value"][0]
    v_data["query_surface_sdf"] = torch.clamp(v_data["query_surface_sdf"], -clip_value, clip_value) / clip_value
    v_data["query_edge_sdf"] = torch.clamp(v_data["query_edge_sdf"], -clip_value, clip_value) / clip_value
    v_data["query_surface_udf"] = torch.clamp(v_data["query_surface_udf"], 0, clip_value) / clip_value
    v_data["query_edge_udf"] = torch.clamp(v_data["query_edge_udf"], 0, clip_value) / clip_value

    device = v_data["sample_points_surfaces"].device
    dtype = v_data["sample_points_surfaces"].dtype
    bs = v_data["sample_points_surfaces"].shape[0]
    matrix = torch.eye(4, device=device, dtype=dtype)[None, :].repeat([bs, 1, 1])
    if v_data["is_aug"][0] == 1:
        id_aug = v_data["id_aug"] if v_test else np.random.randint(0, len(v_matrix))
        matrix[:, :3, :3] = v_matrix[id_aug]
    elif v_data["is_aug"][0] == 2:
        matrix[:, :3, :3] = random_matrix(bs, device, dtype)
    elif v_data["is_aug"][0] == 3:
        ratio = np.array([0.2, 0.4, 0.2, 0.2])  # unchanged, rotation, scale, shift
        id_aug = np.random.choice(np.arange(4), size=bs, p=ratio)
        num_rot = int(np.sum(id_aug == 1))
        matrix[id_aug == 1, :3, :3] = v_matrix[np.random.randint(0, len(v_matrix), size=num_rot)]
        num_scale = int(np.sum(id_aug == 2))
        scale = (1.05 - 0.8) * torch.rand(num_scale, 3) + 0.8
        matrix[id_aug == 2, :3, :3] = torch.diag_embed(scale).to(device=device, dtype=dtype)
        num_shift = int(np.sum(id_aug == 3))
        matrix[id_aug == 3, :3, 3] = (0.05 + 0.05) * torch.rand(num_shift, 3, device=device, dtype=dtype) - 0.05

    new_data = {}
    new_data["sample_points_surfaces"] = rotate(v_data["sample_points_surfaces"], matrix)
    new_data["sample_points_edges"] = rotate(v_data["sample_points_edges"], matrix)
    new_data["sample_points_voronoi"] = rotate(v_data["sample_points_voronoi"], matrix)
    new_data["query_surface_points"] = rotate(v_data["query_surface_points"], matrix)
    new_data["query_edge_points"] = rotate(v_data["query_edge_points"], matrix)
    new_data["query_surface_sdf"] = v_data["query_surface_sdf"]
    new_data["query_edge_sdf"] = v_data["query_edge_sdf"]
    new_data["query_surface_udf"] = v_data["query_surface_udf"]
    new_data["query_edge_udf"] = v_data["query_edge_udf"]
    return new_data


# --------------------------------------------------------------------------- #
# Grid inference (decode latent -> dense SDF/UDF volume)
# --------------------------------------------------------------------------- #
# The model truncates SDF to +/- clip_value and scales to [-1, 1], so the surface
# sits at value 0. NEAR_SURFACE_BAND is the half-width (in scaled units) of the
# region refined at high resolution by the accelerated mesher below.
NEAR_SURFACE_BAND = 0.010 / 0.015  # ~0.667, generous band around the surface


@torch.no_grad()
def _inference(v_latent, v_query, v_res=256, v_points=None):
    """Dense evaluation of the decoder on a v_res^3 grid (or on v_points)."""
    device, dtype = v_latent.device, v_latent.dtype
    batch_size = 100000

    def eval_pts(pts):
        results = []
        for i in range(0, pts.shape[1], batch_size):
            end = min(i + batch_size, pts.shape[1])
            results.append(v_query(pts[:, i:end].expand(v_latent.shape[0], -1, -1), v_latent))
        return torch.cat(results, dim=1)

    def get_grid_pts(res):
        grid = torch.linspace(-1, 1, res, device=device, dtype=dtype)
        pts = torch.stack(torch.meshgrid(grid, grid, grid, indexing="ij"), dim=-1)
        return pts.reshape(1, -1, 3)

    pts = get_grid_pts(v_res) if v_points is None else \
        torch.from_numpy(v_points).to(device=device, dtype=dtype)[None, :]
    results = eval_pts(pts)
    bs, num_channels = results.shape[0], results.shape[-1]
    if v_points is None:
        results = results.view(bs, v_res, v_res, v_res, num_channels)
    return results


@torch.no_grad()
def _inference_acc(v_latent, v_query, v_res=256, v_points=None, half_band=NEAR_SURFACE_BAND):
    """Accelerated multi-resolution evaluation, centered on the SDF=0 surface.

    Stage 1 evaluates a coarse 128^3 grid. Subsequent stages only re-evaluate the
    near-surface band (|sdf| < half_band) at higher resolution and trilinearly
    upsample everywhere else. Produces an SDF iso=0 surface identical to the dense
    path while evaluating far fewer points. Falls back to the dense path when
    ``v_points`` is given (probing arbitrary points).
    """
    if v_points is not None:
        return _inference(v_latent, v_query, v_res, v_points)
    if v_res < 128:
        raise ValueError("v_res must be at least 128 for multi-stage inference")

    device, dtype = v_latent.device, v_latent.dtype
    bs = v_latent.shape[0]
    batch_size = 100000

    def eval_pts(pts):
        results = []
        for i in range(0, pts.shape[1], batch_size):
            end = min(i + batch_size, pts.shape[1])
            results.append(v_query(pts[:, i:end].expand(bs, -1, -1), v_latent))
        return torch.cat(results, dim=1)

    def get_grid_pts(res):
        grid = torch.linspace(-1, 1, res, device=device, dtype=dtype)
        pts = torch.stack(torch.meshgrid(grid, grid, grid, indexing="ij"), dim=-1)
        return pts.reshape(1, -1, 3).expand(bs, -1, -1)

    def upsample(vol, factor):
        r = vol.shape[2]
        return F.interpolate(vol.float(), size=(r * factor,) * 3,
                             mode='trilinear', align_corners=True).to(vol.dtype)

    def near_surface_mask(sdf_vol, factor):
        near = (sdf_vol.abs() < half_band).to(dtype)
        dilated = F.max_pool3d(near, kernel_size=3, stride=1, padding=1) > 0.5
        out = dilated
        for dim in (2, 3, 4):
            out = out.repeat_interleave(factor, dim=dim)
        return out.view(bs, -1).any(dim=0)

    # Stage 1: coarse 128^3
    res_128 = 128
    results = eval_pts(get_grid_pts(res_128))
    sdf_vol = results[..., 0].view(bs, 1, res_128, res_128, res_128)
    udf_vol = results[..., 1].view(bs, 1, res_128, res_128, res_128)

    # Stage 2: refine near-surface up to min(v_res, 256)
    res_mid = min(v_res, 256)
    factor = res_mid // res_128
    if factor > 1:
        mask_bool = near_surface_mask(sdf_vol, factor)
        results = eval_pts(get_grid_pts(res_mid)[:, mask_bool, :])
        sdf_vol = upsample(sdf_vol, factor)
        udf_vol = upsample(udf_vol, factor)
        sdf_flat, udf_flat = sdf_vol.view(bs, -1), udf_vol.view(bs, -1)
        sdf_flat[:, mask_bool] = results[..., 0]
        udf_flat[:, mask_bool] = results[..., 1]
        sdf_vol = sdf_flat.view(bs, 1, res_mid, res_mid, res_mid)
        udf_vol = udf_flat.view(bs, 1, res_mid, res_mid, res_mid)

    if v_res <= 256:
        return torch.cat([sdf_vol.permute(0, 2, 3, 4, 1), udf_vol.permute(0, 2, 3, 4, 1)], dim=-1)

    # Stage 3: refine near-surface at 512^3
    mask_bool = near_surface_mask(sdf_vol, 2)
    results = eval_pts(get_grid_pts(512)[:, mask_bool, :])
    sdf_vol = upsample(sdf_vol, 2)
    udf_vol = upsample(udf_vol, 2)
    sdf_flat, udf_flat = sdf_vol.view(bs, -1), udf_vol.view(bs, -1)
    sdf_flat[:, mask_bool] = results[..., 0]
    udf_flat[:, mask_bool] = results[..., 1]
    return torch.cat([sdf_flat.view(bs, 512, 512, 512, 1), udf_flat.view(bs, 512, 512, 512, 1)], dim=-1)


# --------------------------------------------------------------------------- #
# DualVAE
# --------------------------------------------------------------------------- #
class DualVAE(nn.Module):
    def __init__(self, v_args):
        super().__init__()
        self.is_deterministic = v_args["is_deterministic"]
        self.tokens = np.array(v_args["tokens"], dtype=np.int32)
        self.train_prob = np.array(v_args["train_prob"])
        self.test_prob = v_args["test_prob"]
        self.out_dim = self.test_prob * 2

        include_pi = bool(v_args.get("include_pi", False))
        self.is_fps = bool(v_args.get("is_fps", False))
        if v_args.get("is_gaussian_embedding", 1) == 2:
            raise NotImplementedError("Released DualVAE uses is_gaussian_embedding=1 (Fourier).")

        width = 768
        embed_dim = 32
        self.embedder = FourierEmbedder(num_freqs=8, include_input=True, input_dim=3, include_pi=include_pi)
        self.input_proj = nn.Linear(self.embedder.out_dim + 3, width)

        self.cross_attn = ResidualCrossAttentionBlock_gelu(width, 12)
        self.cross_attn1 = ResidualCrossAttentionBlock_gelu(width, 12)
        self.cross_attn2 = ResidualCrossAttentionBlock_gelu(width, 12)

        self_layer = nn.TransformerEncoderLayer(width, 12, batch_first=True, norm_first=True)
        self.self_attn = nn.TransformerEncoder(self_layer, num_layers=8, norm=nn.LayerNorm(width))

        self.pre_kl = nn.Linear(width, embed_dim)

        self.post_kl = nn.Linear(embed_dim, width)
        decoder_layer = nn.TransformerEncoderLayer(width, 12, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=16, norm=nn.LayerNorm(width))

        self.query_proj = nn.Linear(self.embedder.out_dim, width)
        self.querier1 = ResidualCrossAttentionBlock_gelu(width, 12)
        self.output_proj1 = nn.Linear(width, 2)

        aug_matrix = Rotation.create_group('O').as_matrix()
        self.register_buffer("aug_matrix", torch.from_numpy(aug_matrix).float())

        self.mask_voronoi_prob = v_args.get("mask_voronoi_prob", 0)
        self.kl_weights = v_args["kl_weights"]

    def encode_kl_embed(self, latents, deterministic: bool = True):
        mean = self.pre_kl(latents)
        logvar, var = 0, 1
        if (not deterministic and self.kl_weights > 0) or (not self.is_deterministic):
            kl_embed = mean + torch.randn_like(mean)
        else:
            kl_embed = mean
        ref_mean, ref_var = 0, 1
        ref_logvar = math.log(ref_var)
        kl_loss = 0.5 * torch.mean(torch.pow(mean - ref_mean, 2) / ref_var + var / ref_var - 1.0 - logvar + ref_logvar)
        return kl_embed, kl_loss if self.kl_weights > 0 else 0

    def augment(self, v_data, v_test):
        return augment_submitted(v_data, self.aug_matrix, v_test)

    def encode(self, v_data, v_test=False, v_keypoints=None):
        surface_points = v_data["sample_points_surfaces"]
        edge_points = v_data["sample_points_edges"]
        voronoi_points = v_data["sample_points_voronoi"]

        device = surface_points.device
        bs = surface_points.shape[0]
        N_points = surface_points.shape[1]
        dim_points = surface_points.shape[2]

        surface_feat = torch.cat([surface_points[..., 3:], self.embedder(surface_points[..., :3])], dim=-1)
        edge_feat = torch.cat([edge_points[..., 3:], self.embedder(edge_points[..., :3])], dim=-1)
        voronoi_feat = torch.cat([voronoi_points[..., 3:], self.embedder(voronoi_points[..., :3])], dim=-1)

        surface_feat = self.input_proj(surface_feat)
        edge_feat = self.input_proj(edge_feat)
        voronoi_feat = self.input_proj(voronoi_feat)

        if v_test:
            ratio_coarse = ratio_sharp = self.test_prob / N_points
        else:
            tokens = self.tokens.astype(np.float32)
            coarse_ratios = tokens / N_points
            ratio_coarse = np.random.choice(coarse_ratios, size=1, p=self.train_prob)[0].item()
            index = np.where(coarse_ratios == ratio_coarse)[0]
            ratio_sharp = (tokens / N_points)[index].item()

        flattened = surface_points.view(bs * N_points, dim_points)
        batch = torch.repeat_interleave(torch.arange(bs, device=device), N_points)
        idx = fps(flattened, batch, ratio=ratio_coarse)
        query_surface = surface_feat.view(bs * N_points, -1)[idx].view(bs, -1, surface_feat.shape[-1])

        flattened = edge_points.view(bs * N_points, dim_points)
        batch = torch.repeat_interleave(torch.arange(bs, device=edge_points.device), N_points)
        idx = fps(flattened, batch, ratio=ratio_sharp)
        query_edge = edge_feat.view(bs * N_points, -1)[idx].view(bs, -1, edge_feat.shape[-1])

        query = torch.cat([query_surface, query_edge], dim=1)

        latents_surface = self.cross_attn(query, surface_feat)
        latents_edge = self.cross_attn1(query, edge_feat)
        latents_voronoi = self.cross_attn2(query, voronoi_feat)
        latents = latents_surface
        mask = np.random.rand(latents.shape[0]) > self.mask_voronoi_prob
        latents[mask] = latents[mask] + latents_voronoi[mask] + latents_edge[mask]
        latents = self.self_attn(latents)
        return self.encode_kl_embed(latents, v_test)

    def decode(self, kl_embed):
        return self.decoder(self.post_kl(kl_embed))

    def query(self, points, v_latent):
        query_feat = self.querier1(self.query_proj(self.embedder(points)), v_latent)
        return self.output_proj1(query_feat)


# --------------------------------------------------------------------------- #
# DualVAE_PC — point-cloud reconstruction VAE
# --------------------------------------------------------------------------- #
class DualVAE_PC(DualVAE):
    """"Submitted recon" VAE for reconstructing a shape from a **raw point cloud**.

    Architecturally identical to ``DualVAE`` (same weights layout), but the
    encoder also accepts surface points alone: when ``sample_points_edges`` /
    ``sample_points_voronoi`` are absent (point-cloud input) the edge/voronoi
    cross-attentions are skipped and the latent is built from the surface stream
    only. This checkpoint was trained with voronoi masking (``mask_voronoi_prob``),
    so surface-only encoding is in-distribution. Used with the ``pc_test`` `.ply`
    point clouds.
    """

    def __init__(self, v_args):
        super().__init__(v_args)
        self.is_noisy = bool(v_args.get("is_noisy", False))
        # decode_embedder is a parameter-free Fourier module present in the original
        # (used only by the training forward); kept for fidelity, contributes no weights.
        self.decode_embedder = FourierEmbedder(
            num_freqs=8, include_input=True, input_dim=3,
            include_pi=bool(v_args.get("include_pi", False)))

    def encode(self, v_data, v_test=False, v_keypoints=None):
        surface_points = v_data["sample_points_surfaces"]
        device = surface_points.device
        bs = surface_points.shape[0]
        N_points = surface_points.shape[1]
        dim_points = surface_points.shape[2]

        surface_feat = self.input_proj(
            torch.cat([surface_points[..., 3:], self.embedder(surface_points[..., :3])], dim=-1))

        # Point-cloud input has surface points only; the dual fields are optional.
        not_test_ae = "sample_points_edges" in v_data
        if not_test_ae:
            edge_points = v_data["sample_points_edges"]
            voronoi_points = v_data["sample_points_voronoi"]
            edge_feat = self.input_proj(
                torch.cat([edge_points[..., 3:], self.embedder(edge_points[..., :3])], dim=-1))
            voronoi_feat = self.input_proj(
                torch.cat([voronoi_points[..., 3:], self.embedder(voronoi_points[..., :3])], dim=-1))

        if v_test:
            # Clamp so FPS never asks for more tokens than there are points (small clouds).
            ratio_coarse = ratio_sharp = min(self.test_prob, N_points) / N_points
        else:
            tokens = self.tokens.astype(np.float32)
            coarse_ratios = tokens / N_points
            ratio_coarse = np.random.choice(coarse_ratios, size=1, p=self.train_prob)[0].item()
            ratio_sharp = (tokens / N_points)[np.where(coarse_ratios == ratio_coarse)[0]].item()

        flattened = surface_points.view(bs * N_points, dim_points)
        batch = torch.repeat_interleave(torch.arange(bs, device=device), N_points)
        idx = fps(flattened, batch, ratio=ratio_coarse)
        query_surface = surface_feat.view(bs * N_points, -1)[idx].view(bs, -1, surface_feat.shape[-1])

        if not_test_ae:
            flattened = edge_points.view(bs * N_points, dim_points)
            batch = torch.repeat_interleave(torch.arange(bs, device=edge_points.device), N_points)
            idx = fps(flattened, batch, ratio=ratio_sharp)
            query_edge = edge_feat.view(bs * N_points, -1)[idx].view(bs, -1, edge_feat.shape[-1])

        mask = np.random.rand(surface_points.shape[0]) > self.mask_voronoi_prob
        query = query_surface
        if not_test_ae:
            query[mask] = query[mask] + query_edge[mask]

        latents = self.cross_attn(query, surface_feat)
        if not_test_ae:
            latents_voronoi = self.cross_attn2(query, voronoi_feat)
            latents_edge = self.cross_attn1(query, edge_feat)
            latents[mask] = latents[mask] + latents_voronoi[mask] + latents_edge[mask]
        latents = self.self_attn(latents)
        return self.encode_kl_embed(latents, v_test)
