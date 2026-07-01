"""DualBrep — conditional shape generation (rectified flow).

Loads the ``FusedModelFlow`` generation model, encodes the conditioning input into a
sequence, samples a VAE latent set with the flow ODE, decodes it into SDF + UDF fields,
extracts meshes, and (optionally) segments the surface into BRep faces.
Two conditioning modes (``dataset.input``):

    pointcloud  oriented point cloud (.ply) --PCModel2--------> cond   (config_gen_pc.yaml)
    image       single RGB render (.png/.jpg) --ImgModel/DINOv2-> cond   (config_gen_img.yaml)

           cond --> flow ODE (noise -> latent) --DualVAE.decode--> SDF/UDF
                --> marching cubes --> clustering

Per shape, results go to ``<output_dir>/<prefix>/``:
    cond_pc.ply / cond_img.png   the conditioning input
    recon_sdf.ply         generated surface (marching cubes on the SDF volume)
    recon_udf.ply         generated UDF edge/wireframe iso-surface
    sdf_g.npy / udf_g.npy per-face SDF / UDF probed at recon_sdf face centers
    cluster.ply           surface colored by BRep-face label (runtime.compute_clustering)

Usage:
    python vae_generate.py config=config_gen_pc.yaml      # point cloud -> shape
    python vae_generate.py config=config_gen_img.yaml     # image -> shape
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import trimesh
import lightning.pytorch as pl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_model import FusedModelFlow
from model import _inference, _inference_acc
from dataset import PointCloudDataset, ImageDataset
from mesh_utils import sdf2mesh, udf2mesh
from clustering import process_item


def load_flow_state(path):
    """Load the flow Lightning checkpoint -> bare ``FusedModelFlow`` state_dict."""
    path = str(path)
    if path.startswith("s3://"):
        from cloudpathlib import S3Path
        tmp = tempfile.mkdtemp()
        local = Path(tmp) / Path(path).name
        print(f"Downloading flow checkpoint {path} ...")
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


class FlowGen(pl.LightningModule):
    """Wraps FusedModelFlow and runs point-cloud-conditioned generation in ``test_step``."""

    def __init__(self, cfg):
        super().__init__()
        self.model = FusedModelFlow(OmegaConf.to_container(cfg.model, resolve=True))
        self.clip_value = float(cfg.dataset.clip_value)
        self.test_res = int(cfg.runtime.test_res)
        self.acc = bool(cfg.runtime.acc)
        self.steps = int(cfg.runtime.steps)
        self.temperature = float(cfg.runtime.temperature)
        self.udf_threshold = float(cfg.runtime.udf_threshold)
        self.compute_clustering = bool(cfg.runtime.compute_clustering)
        self.out_root = Path(cfg.runtime.output_dir)
        self.out_root.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        model = self.model
        # Condition on the input only (no ground-truth geometry -> pure generation).
        if "imgs" in batch:
            data = {"imgs": batch["imgs"], "id_aug": batch["id_aug"]}
        else:
            data = {"pc": batch["pc"], "id_aug": batch["id_aug"]}
        pred_decoded, _ = model.inference(data, steps=self.steps, temperature=self.temperature)

        infer = _inference_acc if self.acc else _inference
        grid = infer(pred_decoded, model.vae_model.query, self.test_res, None)
        sdf = np.clip(grid[..., 0].float().cpu().numpy(), -1, 1) * self.clip_value
        udf = np.clip(grid[..., 1].float().cpu().numpy(), 0, 1) * self.clip_value

        for i in range(sdf.shape[0]):
            prefix = batch["prefix"][i]
            item_root = self.out_root / str(prefix)
            item_root.mkdir(parents=True, exist_ok=True)
            try:
                if "imgs" in batch:
                    from PIL import Image
                    Image.fromarray(batch["ori_img"][i].cpu().numpy()).save(str(item_root / "cond_img.png"))
                else:
                    pc = batch["pc"][i].float().cpu().numpy()
                    trimesh.PointCloud(pc[:, :3]).export(str(item_root / "cond_pc.ply"))

                v, f = sdf2mesh(sdf[i])
                if f.shape[0] == 0:
                    print(f"[{prefix}] empty SDF mesh, skipping")
                    continue
                sdf_mesh = trimesh.Trimesh(vertices=v, faces=f)
                sdf_mesh.export(str(item_root / "recon_sdf.ply"))

                vu, fu = udf2mesh(udf[i], self.udf_threshold)
                if fu.shape[0] > 0:
                    trimesh.Trimesh(vertices=vu, faces=fu).export(str(item_root / "recon_udf.ply"))

                vv = np.asarray(sdf_mesh.vertices)
                ff = np.asarray(sdf_mesh.faces)
                centers = vv[ff].mean(axis=1)
                face_res = _inference(pred_decoded[i:i + 1], model.vae_model.query, self.test_res, centers)
                np.save(str(item_root / "sdf_g"), np.clip(face_res[0, ..., 0].float().cpu().numpy(), -1, 1) * self.clip_value)
                np.save(str(item_root / "udf_g"), np.clip(face_res[0, ..., 1].float().cpu().numpy(), 0, 1) * self.clip_value)

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
    cfg_name = str(cli.pop("config")) if "config" in cli else "config_gen_pc.yaml"
    config_path = cfg_name if (os.path.isabs(cfg_name) or os.path.exists(cfg_name)) else str(here / cfg_name)
    cfg = OmegaConf.merge(OmegaConf.load(config_path), cli)
    # Resolve a relative name_list / data_root against the package dir (run-from-anywhere).
    nl = cfg.dataset.get("name_list")
    if nl not in (None, "", "None") and not str(nl).startswith("s3://"):
        if not os.path.isabs(str(nl)) and not os.path.exists(str(nl)):
            cfg.dataset.name_list = str(here / str(nl))
    dr = cfg.dataset.get("data_root")
    if dr not in (None, "", "None") and not str(dr).startswith("s3://"):
        if not os.path.isabs(str(dr)) and not os.path.exists(str(dr)):
            cfg.dataset.data_root = str(here / str(dr))
    return cfg


def main():
    cfg = load_config()
    print(OmegaConf.to_yaml(cfg))
    torch.set_float32_matmul_precision("high")
    assert cfg.model.name == "FusedModelFlow", f"vae_generate.py expects model.name=FusedModelFlow (got {cfg.model.name})"

    module = FlowGen(cfg)  # __init__ loads model.vae_weights into the (frozen) VAE
    state = load_flow_state(cfg.checkpoint)
    missing, unexpected = module.model.load_state_dict(state, strict=False)
    if missing:
        print(f"WARNING: {len(missing)} missing keys, e.g. {missing[:5]}")
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected keys, e.g. {unexpected[:5]}")
    if not missing and not unexpected:
        print("Flow checkpoint loaded (0 missing / 0 unexpected keys).")

    ds_conf = OmegaConf.to_container(cfg.dataset, resolve=True)
    ds_conf["num_samples"] = cfg.runtime.get("num_samples")
    if str(cfg.dataset.get("input", "pointcloud")) == "image":
        dataset = ImageDataset(ds_conf)
    else:
        dataset = PointCloudDataset(ds_conf)
    loader = DataLoader(dataset, batch_size=int(cfg.runtime.batch_size),
                        num_workers=int(cfg.runtime.num_workers), shuffle=False)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=int(cfg.runtime.gpu) if torch.cuda.is_available() else 1,
        precision=str(cfg.runtime.precision),
        logger=False, enable_checkpointing=False, enable_progress_bar=True,
    )
    trainer.test(module, dataloaders=loader)
    print(f"\nDone. Results in {module.out_root.resolve()}")


if __name__ == "__main__":
    main()
