"""DualBrep — B-rep rebuilder (stage 3a): segmented mesh -> face/edge grids + topology.

Takes a segmented surface mesh (``cluster.ply``: per-face ``label``, produced by the
reconstruction/clustering stage) and runs the ``Parametrizer`` model: each labelled face is
sampled into a point cloud, the network re-fits every face as a parametric surface grid and
predicts the face-face intersection edges and their topology.

Test-time rotation augmentation (``--rotations``): the mesh is rotated by octahedral
rotation ``M[k]`` before sampling, so the network sees pose ``k``; the output is written
under ``<name>_<k:02d>/`` so the post-processing stage (``postprocess.py``) undoes it with
``inv(M[k])``. Index 3 is the identity (canonical pose). Running all 24 rotations gives the
assembler 24 candidate reconstructions to seal into a valid solid.

Pipeline:  cluster.ply --rotate M[k] + per-face sample--> face_sample_points [F, P, 6]
           --Parametrizer.inference--> per-face surface grids + intersection edges + topology
           --> post.npz   (handed to postprocess.py to build the B-rep STEP)

Per (shape, rotation), results go to ``<out>/<name>_<k>/``:
    input_faces.ply   the per-face sampled points (colored by face label)
    recon_faces.ply   reconstructed B-rep faces (16x16 surface grids, colored by face)
    recon_edges.ply   reconstructed intersection edges (sampled curve points)
    post.npz          pred_face (F,16,16,6), pred_edge (E,16,3), pred_edge_face_connectivity (E,3)

Usage:
    python rebuild.py --input /path/to/<name>/cluster.ply --out output_brep --rotations all
    python rebuild.py --input /path/to/recon_dir          --out output_brep   # each <name>/ has cluster.ply
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import trimesh
import open3d as o3d
from scipy.spatial.transform import Rotation

from rebuild_model import Parametrizer, normalize_coord0516

DEFAULT_CKPT = "checkpoints/parametrizer.ckpt"

# Architecture config of the parametrizer checkpoint.
MODEL_CFG = {
    "name": "Parametrizer", "loss": "l1", "num_sample_points": 100,
    "with_normal": True, "in_channels": 6, "dim_shape": 768,
    "gaussian_weights": 1e-6, "max_intersections": 20,
    "edge_align_weight": -1, "freeze_face": False, "face_weights": None,
}

# Octahedral group (24 rotations); index 3 is the identity / canonical pose.
ROT = Rotation.create_group("O").as_matrix()

_PALETTE = (np.array([
    [228, 26, 28], [55, 126, 184], [77, 175, 74], [152, 78, 163], [255, 127, 0],
    [255, 255, 51], [166, 86, 40], [247, 129, 191], [153, 153, 153], [26, 188, 156],
    [52, 152, 219], [155, 89, 182], [241, 196, 15], [231, 76, 60], [46, 204, 113],
    [149, 165, 166], [243, 156, 18], [211, 84, 0], [127, 140, 141], [44, 62, 80],
], dtype=np.uint8))


def _color(i):
    return _PALETTE[i % len(_PALETTE)]


def load_state(path):
    """Lightning checkpoint (local or s3://) -> bare Parametrizer state_dict."""
    path = str(path)
    if path.startswith("s3://"):
        from cloudpathlib import S3Path
        tmp = tempfile.mkdtemp()
        local = Path(tmp) / Path(path).name
        print(f"Downloading checkpoint {path} ...")
        S3Path(path).download_to(str(local))
        path = str(local)
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    return {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}


def sample_faces(mesh, labels, n_points):
    """Segmented mesh -> (face_sample_points [F, n_points, 6], face_input_bbox [F, 6]).

    Per labelled face: open3d uniform sampling with triangle normals (global coords), and a
    per-face bbox (center+extent) from the face vertices via normalize_coord0516.
    """
    uniq = np.unique(labels)
    uniq = uniq[uniq != -1]
    fsp, fbb = [], []
    for lab in uniq:
        mc = mesh.copy()
        mc.update_faces(np.where(labels == lab)[0])
        tm = o3d.geometry.TriangleMesh()
        tm.vertices = o3d.utility.Vector3dVector(np.asarray(mc.vertices))
        tm.triangles = o3d.utility.Vector3iVector(np.asarray(mc.faces))
        tm.remove_unreferenced_vertices()
        verts = torch.from_numpy(np.asarray(tm.vertices)).unsqueeze(0)
        _, bbox = normalize_coord0516(verts)
        pc = tm.sample_points_uniformly(n_points, use_triangle_normal=True)
        pts = np.concatenate([np.asarray(pc.points), np.asarray(pc.normals)], axis=-1)
        fsp.append(pts.astype(np.float32))
        fbb.append(bbox[0].numpy().astype(np.float32))
    if not fsp:
        return None, None
    return torch.from_numpy(np.stack(fsp)), torch.from_numpy(np.stack(fbb))


def read_cluster(cluster_ply):
    mesh = trimesh.load(cluster_ply, process=False)
    fd = mesh.metadata["_ply_raw"]["face"]["data"]
    field = "label" if "label" in fd.dtype.names else "cluster"
    return mesh, np.asarray(fd[field])


def _grid_to_mesh(grid_xyz):
    h, w, _ = grid_xyz.shape
    verts = grid_xyz.reshape(-1, 3)
    a, b = np.meshgrid(np.arange(h - 1), np.arange(w - 1), indexing="ij")
    a, b = a.reshape(-1), b.reshape(-1)
    tl = a * w + b
    tr, bl, br = tl + 1, tl + w, tl + w + 1
    tris = np.concatenate([np.stack([tl, bl, br], 1), np.stack([tl, br, tr], 1)], axis=0)
    return verts, tris


def save_outputs(results, fsp, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_face = results["pred_face"]
    pred_edge = results.get("pred_edge", np.zeros((0, 16, 3)))
    conn = results.get("pred_edge_face_connectivity", np.zeros((0, 3)))

    fsp_np = fsp.cpu().numpy()
    trimesh.PointCloud(fsp_np[..., :3].reshape(-1, 3),
                       colors=np.repeat([_color(i) for i in range(fsp_np.shape[0])], fsp_np.shape[1], axis=0)
                       ).export(str(out_dir / "input_faces.ply"))

    verts, tris, vcol, voff = [], [], [], 0
    for i in range(pred_face.shape[0]):
        v, t = _grid_to_mesh(pred_face[i, ..., :3])
        verts.append(v); tris.append(t + voff)
        vcol.append(np.tile(_color(i), (v.shape[0], 1))); voff += v.shape[0]
    if verts:
        trimesh.Trimesh(vertices=np.concatenate(verts), faces=np.concatenate(tris),
                        vertex_colors=np.concatenate(vcol), process=False).export(str(out_dir / "recon_faces.ply"))
    if pred_edge.shape[0] > 0:
        trimesh.PointCloud(pred_edge[..., :3].reshape(-1, 3),
                           colors=np.repeat([_color(i) for i in range(pred_edge.shape[0])], pred_edge.shape[1], axis=0)
                           ).export(str(out_dir / "recon_edges.ply"))

    np.savez(str(out_dir / "post.npz"), pred_face=pred_face, pred_edge=pred_edge,
             pred_edge_face_connectivity=conn)
    return pred_face.shape[0], pred_edge.shape[0]


def collect_inputs(input_path):
    p = Path(input_path)
    if p.is_file():
        return [(p.parent.name or p.stem, p)]
    if (p / "cluster.ply").exists():
        return [(p.name, p / "cluster.ply")]
    return [(c.parent.name, c) for c in sorted(p.glob("*/cluster.ply"))]


def parse_rotations(spec):
    if spec == "all":
        return list(range(24))
    return [int(x) for x in str(spec).split(",") if x.strip() != ""]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="B-rep rebuilder: segmented mesh (cluster.ply) -> face/edge grids + topology.")
    ap.add_argument("--input", required=True,
                    help="cluster.ply, a folder with cluster.ply, or a parent of such folders.")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT, help="Parametrizer checkpoint (local or s3://).")
    ap.add_argument("--out", default="output_brep", help="Output directory.")
    ap.add_argument("--rotations", default="3",
                    help="Octahedral rotation ids for test-time aug: '3' (canonical), 'all' (0-23), or e.g. '3,7,11'.")
    ap.add_argument("--num_input_points", type=int, default=100, help="Points sampled per face.")
    ap.add_argument("--num_samples", type=int, default=None, help="Cap the number of shapes processed.")
    args = ap.parse_args()

    rotations = parse_rotations(args.rotations)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Parametrizer(dict(MODEL_CFG))
    missing, unexpected = model.load_state_dict(load_state(args.checkpoint), strict=False)
    print("Checkpoint loaded (0 missing / 0 unexpected keys)." if not missing and not unexpected
          else f"WARNING: {len(missing)} missing / {len(unexpected)} unexpected keys")
    model = model.to(device).eval()

    inputs = collect_inputs(args.input)
    if args.num_samples:
        inputs = inputs[: args.num_samples]
    if not inputs:
        raise FileNotFoundError(f"No cluster.ply found under {args.input}")
    print(f"{len(inputs)} shape(s) x {len(rotations)} rotation(s).")

    for name, cluster_ply in inputs:
        stem = str(name).split("_")[0]   # bare shape id; postprocess reads the rotation from "_<k>"
        mesh, labels = read_cluster(cluster_ply)
        for k in rotations:
            m = mesh.copy()
            T = np.eye(4); T[:3, :3] = ROT[k]
            m.apply_transform(T)
            fsp, fbb = sample_faces(m, labels, args.num_input_points)
            if fsp is None or fsp.shape[0] < 2:
                print(f"[{stem} rot{k:02d}] <2 faces, skipping")
                continue
            fsp, fbb = fsp.to(device), fbb.to(device)
            n = fsp.shape[0]
            data = {"face_sample_points": fsp, "face_input_bbox": fbb,
                    "face_attn_mask": torch.zeros((n, n), dtype=torch.bool, device=device)}
            results = model.inference(data)
            # The 24 rotation candidates are intermediate -> keep them under <out>/tmp/.
            od = Path(args.out) / "tmp" / f"{stem}_{k:02d}"
            nf, ne = save_outputs(results, fsp, od)
            print(f"[{stem} rot{k:02d}] {n} faces -> {nf} recon faces, {ne} edges -> {od}")

    print(f"\nDone. Candidates in {(Path(args.out) / 'tmp').resolve()}")
    print(f"Next:  python postprocess.py --input {Path(args.out)}")


if __name__ == "__main__":
    main()
