"""Dataset for DualBrep VAE inference.

Reads per-shape ``.npz`` files (the ``implicit_*`` geometry format) and produces
the tensors the encoder consumes. A file is selected either from an explicit
name list or by globbing ``*.npz`` under ``data_root``. Both local directories
and ``s3://`` prefixes are supported.

Each ``.npz`` is expected to contain:
    surface_points        (Ns, 6)  xyz + normal, sampled on faces
    edge_points           (Ne, 6)  xyz + normal, sampled on BRep edges
    voronoi_points        (Nv, 6)  xyz + normal, interior/medial samples
    query_surface_points  (Q, 3)   query xyz for the surface field
    query_surface_sdf     (Q,)     truncated signed distance
    query_surface_udf     (Q,)     unsigned distance to nearest edge
    query_edge_points     (Q, 3)   query xyz near edges
    query_edge_sdf        (Q,)
    query_edge_udf        (Q,)
"""
import os
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

# Octahedral group (24 proper rotations); index 3 is the identity / canonical pose.
# Same set (and indexing) used by the model, rebuild.py and brep_post.
_OCTA = Rotation.create_group("O").as_matrix().astype(np.float32)


def _is_s3(path) -> bool:
    return str(path).startswith("s3://")


def _list_stems(data_root, ext):
    """Return sorted shape stems with the given extension under ``data_root``."""
    if _is_s3(data_root):
        from cloudpathlib import S3Path
        root = S3Path(str(data_root))
        return sorted(p.stem for p in root.glob(f"*{ext}"))
    root = Path(data_root)
    return sorted(p.stem for p in root.glob(f"*{ext}"))


def _resolve_names(data_root, name_list, ext):
    """Stems from ``name_list`` (txt of stems, local or s3://) or, if null, by globbing ``data_root``."""
    if name_list in (None, "", "None"):
        return _list_stems(data_root, ext)
    if _is_s3(name_list):
        from cloudpathlib import S3Path
        text = S3Path(str(name_list)).read_text()
    else:
        text = open(name_list).read()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return sorted({ln.split()[0] for ln in lines})


def _join(data_root, name):
    if _is_s3(data_root):
        return f"{data_root.rstrip('/')}/{name}"
    return str(Path(data_root) / name)


def _download_if_s3(path, tmp):
    if _is_s3(path):
        from cloudpathlib import S3Path
        local = Path(tmp) / Path(str(path)).name
        S3Path(str(path)).download_to(str(local))
        return str(local)
    return str(path)


def _load_npz(path):
    """np.load a local path or an s3:// path (downloaded to a temp file)."""
    if _is_s3(path):
        from cloudpathlib import S3Path
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / Path(str(path)).name
            S3Path(str(path)).download_to(str(local))
            return dict(np.load(local, allow_pickle=True))
    return np.load(path, allow_pickle=True)


class VAEDataset(Dataset):
    """Loads sampled point clouds + query fields for VAE reconstruction."""

    def __init__(self, v_conf):
        super().__init__()
        self.conf = v_conf
        self.data_root = v_conf["data_root"]
        self.clip_value = float(v_conf["clip_value"])
        self.is_aug = int(v_conf.get("is_aug", 0))
        # id_aug selects which octahedral rotation (0..23) is applied when is_aug==1.
        # The same fixed transform is used for every shape (no 24-way fan-out here).
        self.id_aug = int(v_conf.get("id_aug", 0))
        self.num_points = int(v_conf.get("num_points", 32768))
        self.n_supervision = int(v_conf.get("n_supervision", 32768))

        names = _resolve_names(self.data_root, v_conf.get("name_list"), ".npz")
        if v_conf.get("num_samples"):
            names = names[: int(v_conf["num_samples"])]
        if not names:
            raise FileNotFoundError(f"No .npz shapes found under {self.data_root}")
        self.names = names
        self.indexes = [_join(self.data_root, f"{n}.npz") for n in names]
        print(f"[VAEDataset] {len(self.indexes)} shapes from {self.data_root}")

    def __len__(self):
        return len(self.indexes)

    def _sample(self, arr, n, replace):
        idx = np.random.choice(np.arange(arr.shape[0]), n, replace=replace)
        return arr[idx]

    def __getitem__(self, v_idx):
        path = self.indexes[v_idx]
        content = _load_npz(path)

        surf = self._sample(content["surface_points"].astype(np.float32), self.num_points, False)
        edge = self._sample(content["edge_points"].astype(np.float32), self.num_points, False)
        voro = self._sample(content["voronoi_points"].astype(np.float32), self.num_points, True)

        qsp = content["query_surface_points"].astype(np.float32)
        qi = np.random.choice(np.arange(qsp.shape[0]), self.n_supervision, replace=False)
        qsp = qsp[qi]
        qss = content["query_surface_sdf"][qi].astype(np.float32)
        qsu = content["query_surface_udf"][qi].astype(np.float32)

        qep = content["query_edge_points"].astype(np.float32)
        ei = np.random.choice(np.arange(qep.shape[0]), self.n_supervision, replace=False)
        qep = qep[ei]
        qes = content["query_edge_sdf"][ei].astype(np.float32)
        qeu = content["query_edge_udf"][ei].astype(np.float32)

        return {
            "prefix": self.names[v_idx],
            "id_aug": self.id_aug,
            "clip_value": self.clip_value,
            "is_aug": self.is_aug,
            "sample_points_surfaces": surf,
            "sample_points_edges": edge,
            "sample_points_voronoi": voro,
            "query_surface_points": qsp,
            "query_edge_points": qep,
            "query_surface_sdf": qss,
            "query_edge_sdf": qes,
            "query_surface_udf": qsu,
            "query_edge_udf": qeu,
        }


class PointCloudDataset(Dataset):
    """Loads a raw point cloud / mesh for reconstruction with ``DualVAE_PC``.

    Input files are oriented point clouds (``.ply`` with per-vertex normals, e.g.
    ``abc20k/pc_test/*.ply``) or meshes (``.obj`` / ``.ply``; surface points are then
    sampled). The shape is normalized into the cube and returned as
    ``sample_points_surfaces`` (xyz + normal) — no edge/voronoi fields, which makes
    the encoder run in point-cloud (surface-only) mode.

    ``per_axis_norm`` mirrors the original two inference datasets:
      * False (default): uniform scaling, longest axis mapped to [-0.9, 0.9].
      * True: uniform scaling, then thin axes (<0.01 extent) stretched to 0.5; the
        per-axis ``norm_center`` / ``norm_scale`` are returned so the segmentation can
        be denormalized back to input coordinates.
    """

    def __init__(self, v_conf):
        super().__init__()
        self.conf = v_conf
        self.data_root = v_conf["data_root"]
        self.clip_value = float(v_conf["clip_value"])
        self.num_points = int(v_conf.get("num_points", 32768))
        self.per_axis_norm = bool(v_conf.get("per_axis_norm", False))
        self.ext = str(v_conf.get("ext", ".ply"))
        # Octahedral pose applied to the (normalized) cloud before encoding. The encoder
        # is not rotation-invariant, so a different pose gives a different reconstruction.
        # 3 = identity (canonical); the reconstruction is rotated back to input coords
        # after decoding, so cluster.ply and the final B-rep stay in the input frame.
        self.id_aug = int(v_conf.get("id_aug", 3))
        self.aug_R = _OCTA[self.id_aug]

        names = _resolve_names(self.data_root, v_conf.get("name_list"), self.ext)
        if v_conf.get("num_samples"):
            names = names[: int(v_conf["num_samples"])]
        if not names:
            raise FileNotFoundError(f"No {self.ext} shapes found under {self.data_root}")
        self.names = names
        self.files = [_join(self.data_root, f"{n}{self.ext}") for n in names]
        print(f"[PointCloudDataset] {len(self.files)} shapes from {self.data_root}")

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _read_points_normals(mesh, num_points):
        """Return (points, normals) before normalization, as float64."""
        if isinstance(mesh, trimesh.Trimesh):
            mesh.remove_unreferenced_vertices()
            pts, face_id = mesh.sample(num_points, return_index=True)
            normals = mesh.face_normals[face_id]
            return np.asarray(pts, np.float64), np.asarray(normals, np.float64), True
        # PointCloud: use the vertices and their stored normals.
        pts = np.asarray(mesh.vertices, np.float64)
        raw = mesh.metadata["_ply_raw"]["vertex"]["data"]
        normals = np.stack([raw["nx"], raw["ny"], raw["nz"]], axis=1).astype(np.float64)
        return pts, normals, False

    def __getitem__(self, v_idx):
        with tempfile.TemporaryDirectory() as tmp:
            local = _download_if_s3(self.files[v_idx], tmp)
            mesh = trimesh.load(local, process=False)

        center = np.asarray(mesh.bounding_box.centroid, np.float64)
        bbox = np.asarray(mesh.bounding_box.extents, np.float64)
        uniform_scale = float(np.max(bbox))

        # Per-axis extra stretch for thin axes (per_axis_norm only).
        per_axis_extra = np.ones(3)
        if self.per_axis_norm:
            extents = bbox / uniform_scale * 0.9 * 2
            for i in range(3):
                if extents[i] < 0.01:
                    per_axis_extra[i] = 0.5 / extents[i]
        # Effective per-axis divisor: v_norm = (v_orig - center) / norm_scale
        norm_scale = uniform_scale / (0.9 * 2 * per_axis_extra)

        is_mesh = isinstance(mesh, trimesh.Trimesh)
        if is_mesh and self.per_axis_norm:
            # Rebuild after non-uniform scaling so face normals are recomputed cleanly.
            new_verts = (np.asarray(mesh.vertices, np.float64) - center) / norm_scale[None, :]
            mesh = trimesh.Trimesh(vertices=new_verts, faces=mesh.faces, process=False)
            pts, face_id = mesh.sample(self.num_points, return_index=True)
            pts = np.asarray(pts, np.float64)
            normals = np.asarray(mesh.face_normals[face_id], np.float64)
        else:
            pts, normals, is_mesh = self._read_points_normals(mesh, self.num_points)
            pts = (pts - center) / norm_scale[None, :]
            if self.per_axis_norm:
                # Non-uniform scaling: normals transform by the inverse-transpose of the
                # diagonal scaling (multiply by norm_scale), then renormalize. Uniform
                # scaling preserves direction, so raw normals are passed through unchanged.
                normals = normals * norm_scale[None, :]
                normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-12)

        # Rotate the cloud (points + normals) into octahedral pose `id_aug` before encoding.
        # Rotation is orthogonal, so normals rotate the same way (no inverse-transpose).
        if self.id_aug != 3:   # _OCTA[3] == identity; skip the no-op rotation
            pts = pts @ self.aug_R.T
            normals = normals @ self.aug_R.T

        surf = np.concatenate([pts, normals], axis=1).astype(np.float32)
        out = {
            "prefix": self.names[v_idx],
            "clip_value": self.clip_value,
            "is_aug": 0,
            "id_aug": self.id_aug,   # tags the recon folder <prefix>_<id_aug>
            "aug_R": self.aug_R,     # so recon can be rotated back to input coords
            "sample_points_surfaces": surf,
            "pc": surf,
        }
        if self.per_axis_norm:
            out["norm_center"] = center.astype(np.float32)
            out["norm_scale"] = norm_scale.astype(np.float32)
        return out


class ImageDataset(Dataset):
    """Loads a single conditioning image per shape for image-conditioned generation.

    Input files are RGB images (``.png`` / ``.jpg``). Each is resized to ``img_size``
    (518 = 37*14, the DINOv2 ViT-L/14 patch grid the model was trained on) and
    ImageNet-normalized. To stay in distribution the image should resemble the training
    renders (a single centered object on a light background, roughly 45-degree FOV).
    """

    def __init__(self, v_conf):
        super().__init__()
        self.conf = v_conf
        self.data_root = v_conf["data_root"]
        self.clip_value = float(v_conf["clip_value"])
        self.ext = str(v_conf.get("ext", ".png"))
        self.img_size = int(v_conf.get("img_size", 518))

        names = _resolve_names(self.data_root, v_conf.get("name_list"), self.ext)
        if v_conf.get("num_samples"):
            names = names[: int(v_conf["num_samples"])]
        if not names:
            raise FileNotFoundError(f"No {self.ext} images found under {self.data_root}")
        self.names = names
        self.files = [_join(self.data_root, f"{n}{self.ext}") for n in names]
        print(f"[ImageDataset] {len(self.files)} images from {self.data_root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, v_idx):
        from PIL import Image
        from cond_model import img_transform
        with tempfile.TemporaryDirectory() as tmp:
            local = _download_if_s3(self.files[v_idx], tmp)
            img = Image.open(local).convert("RGB").resize((self.img_size, self.img_size))
        arr = np.asarray(img, dtype=np.uint8)
        return {
            "prefix": self.names[v_idx],
            "clip_value": self.clip_value,
            "imgs": img_transform(arr),   # (3, img_size, img_size), ImageNet-normalized
            "ori_img": arr,               # uint8 HxWx3, kept only to save cond_img.png
            "id_aug": 0,
        }
