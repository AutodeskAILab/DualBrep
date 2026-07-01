"""DualBrep VAE — reconstruction inference.

Encodes each input shape into a latent set, decodes a dense SDF + UDF volume,
extracts meshes, probes per-face fields, and (optionally) segments the surface
into BRep faces. Two input modes are supported via ``dataset.input``:
  * ``implicit``   – precomputed ``.npz`` geometry (surface+edge+voronoi+queries),
                     reconstructed with ``DualVAE``.
  * ``pointcloud`` – raw oriented point clouds / meshes (``.ply`` / ``.obj``),
                     reconstructed with ``DualVAE_PC`` (surface-only encoding).

Per shape, results are written to ``<output_dir>/<prefix>_<id_aug>/``:
    recon_sdf.ply        reconstructed surface (marching cubes on the SDF volume)
    recon_udf.ply         edge/wireframe iso-surface of the UDF field
    sdf_g.npy / udf_g.npy per-face SDF / UDF probed at recon_sdf face centers
    cluster.ply           surface colored by BRep-face label (runtime.compute_clustering)

Usage:
    python ae_reconstruct.py                              # use config.yaml defaults (implicit)
    python ae_reconstruct.py config=config_pc.yaml        # point-cloud reconstruction
    python ae_reconstruct.py checkpoint=/path/to.ckpt dataset.data_root=/path/to/dir \\
                    runtime.output_dir=output runtime.test_res=256
Any ``key=value`` (dotlist) argument overrides the config.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # allow `python ae_reconstruct.py` from anywhere

import numpy as np
import torch
import trimesh
import lightning.pytorch as pl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import model as model_mod
from model import _inference, _inference_acc
from dataset import VAEDataset, PointCloudDataset
from mesh_utils import sdf2mesh, udf2mesh
from clustering import process_item

SUPPORTED_MODELS = {"DualVAE", "DualVAE_PC"}


def load_checkpoint_state(path):
    """Load a Lightning checkpoint and return a bare model state_dict.

    Strips the ``model.`` (and optional ``model._orig_mod.`` torch.compile) prefixes
    so the weights map directly onto the model instance.
    """
    path = str(path)
    if path.startswith("s3://"):
        from cloudpathlib import S3Path
        tmp = tempfile.mkdtemp()
        local = Path(tmp) / Path(path).name
        print(f"Downloading checkpoint {path} ...")
        S3Path(path).download_to(str(local))
        path = str(local)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    state = {}
    for k, v in sd.items():
        if k.startswith("model._orig_mod."):
            k = k[len("model._orig_mod."):]
        elif k.startswith("model."):
            k = k[len("model."):]
        state[k] = v
    return state


class VAEInference(pl.LightningModule):
    """Wraps the VAE and runs the reconstruction pipeline in ``test_step``."""

    def __init__(self, cfg):
        super().__init__()
        ModelClass = getattr(model_mod, cfg.model.name)
        self.model = ModelClass(OmegaConf.to_container(cfg.model, resolve=True))
        self.clip_value = float(cfg.dataset.clip_value)
        self.test_res = int(cfg.runtime.test_res)
        self.acc = bool(cfg.runtime.acc)
        self.udf_threshold = float(cfg.runtime.udf_threshold)
        self.compute_clustering = bool(cfg.runtime.compute_clustering)
        self.out_root = Path(cfg.runtime.output_dir)
        self.out_root.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        model = self.model
        data = batch
        if "query_surface_sdf" in data:          # normalize targets + apply pose
            data = model.augment(data, True)
        kl_embed, _ = model.encode(data, True)
        latents = model.decode(kl_embed)

        infer = _inference_acc if self.acc else _inference
        grid = infer(latents, model.query, self.test_res, None)
        sdf = np.clip(grid[..., 0].float().cpu().numpy(), -1, 1) * self.clip_value
        udf = np.clip(grid[..., 1].float().cpu().numpy(), 0, 1) * self.clip_value

        for i in range(sdf.shape[0]):
            prefix = batch["prefix"][i]
            id_aug = batch["id_aug"][i]
            id_aug = int(id_aug.item()) if torch.is_tensor(id_aug) else int(id_aug)
            item_root = self.out_root / f"{prefix}_{id_aug}"
            item_root.mkdir(parents=True, exist_ok=True)
            try:
                v, f = sdf2mesh(sdf[i])
                if f.shape[0] == 0:
                    print(f"[{prefix}] empty SDF mesh, skipping")
                    continue
                sdf_mesh = trimesh.Trimesh(vertices=v, faces=f)
                sdf_mesh.export(str(item_root / "recon_sdf.ply"))

                vu, fu = udf2mesh(udf[i], self.udf_threshold)
                if fu.shape[0] > 0:
                    trimesh.Trimesh(vertices=vu, faces=fu).export(str(item_root / "recon_udf.ply"))

                # Per-face SDF/UDF probed at the surface-mesh face centers (clustering input).
                vv = np.asarray(sdf_mesh.vertices)
                ff = np.asarray(sdf_mesh.faces)
                centers = vv[ff].mean(axis=1)
                face_res = _inference(latents[i:i + 1], model.query, self.test_res, centers)
                sdf_g = np.clip(face_res[0, ..., 0].float().cpu().numpy(), -1, 1) * self.clip_value
                udf_g = np.clip(face_res[0, ..., 1].float().cpu().numpy(), 0, 1) * self.clip_value
                np.save(str(item_root / "sdf_g"), sdf_g)
                np.save(str(item_root / "udf_g"), udf_g)

                # Per-axis normalization params let clustering denormalize back to input coords.
                if "norm_center" in batch and "norm_scale" in batch:
                    np.savez(str(item_root / "norm_params.npz"),
                             center=batch["norm_center"][i].float().cpu().numpy(),
                             scale=batch["norm_scale"][i].float().cpu().numpy())

                msg = f"[{prefix}] SDF {len(sdf_mesh.faces)}f, UDF {fu.shape[0]}f"
                if self.compute_clustering:
                    ok = process_item(item_root)
                    msg += f", cluster {'OK' if ok else 'FAILED'}"
                print(msg + f" -> {item_root}")
            except Exception as e:
                print(f"[{prefix}] ERROR: {e}")


def load_config():
    here = Path(os.path.dirname(os.path.abspath(__file__)))
    cli = OmegaConf.from_dotlist([a for a in sys.argv[1:] if "=" in a])
    cfg_name = str(cli.pop("config")) if "config" in cli else "config.yaml"
    # Resolve a relative config path against the package dir so it works from any cwd.
    config_path = cfg_name if (os.path.isabs(cfg_name) or os.path.exists(cfg_name)) else str(here / cfg_name)
    cfg = OmegaConf.merge(OmegaConf.load(config_path), cli)
    return cfg


def main():
    cfg = load_config()
    print(OmegaConf.to_yaml(cfg))
    torch.set_float32_matmul_precision("high")

    if cfg.model.name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model.name={cfg.model.name}; expected one of {sorted(SUPPORTED_MODELS)}.")

    # Resolve a relative name_list against the package directory so `python ae_reconstruct.py`
    # works from any working directory (config.yaml is also resolved that way).
    here = Path(os.path.dirname(os.path.abspath(__file__)))
    nl = cfg.dataset.get("name_list")
    if nl not in (None, "", "None") and not str(nl).startswith("s3://"):
        if not os.path.isabs(str(nl)) and not os.path.exists(str(nl)):
            cfg.dataset.name_list = str(here / str(nl))

    module = VAEInference(cfg)
    state = load_checkpoint_state(cfg.checkpoint)
    missing, unexpected = module.model.load_state_dict(state, strict=False)
    if missing:
        print(f"WARNING: {len(missing)} missing keys, e.g. {missing[:5]}")
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected keys, e.g. {unexpected[:5]}")
    if not missing and not unexpected:
        print("Checkpoint loaded (0 missing / 0 unexpected keys).")

    ds_conf = OmegaConf.to_container(cfg.dataset, resolve=True)
    ds_conf["num_samples"] = cfg.runtime.get("num_samples")
    input_type = str(cfg.dataset.get("input", "implicit"))
    if input_type == "pointcloud":
        dataset = PointCloudDataset(ds_conf)
    elif input_type == "implicit":
        dataset = VAEDataset(ds_conf)
    else:
        raise ValueError(f"Unknown dataset.input={input_type}; expected 'implicit' or 'pointcloud'.")
    loader = DataLoader(dataset, batch_size=int(cfg.runtime.batch_size),
                        num_workers=int(cfg.runtime.num_workers), shuffle=False)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=int(cfg.runtime.gpu) if torch.cuda.is_available() else 1,
        precision=str(cfg.runtime.precision),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    trainer.test(module, dataloaders=loader)
    print(f"\nDone. Results in {module.out_root.resolve()}")


if __name__ == "__main__":
    main()
