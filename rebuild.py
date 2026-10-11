"""DualBrep — B-rep rebuilder (stage 3a): segmented mesh -> face/edge grids + topology.

Takes a segmented surface mesh (``cluster.ply``: per-face ``label``, produced by the
reconstruction/clustering stage) and runs the ``Parametrizer`` model: each labelled face is
sampled into a point cloud, the network re-fits every face as a parametric surface grid and
predicts the face-face intersection edges and their topology. Predicted edges between faces
whose clusters are clearly apart in ``cluster.ply`` are dropped, and the per-face latents are
then optimised so that the decoded faces meet (``latent_optimise``): each edge lies on both of
its faces, the two half-edges of an adjacency coincide, and the edges of a face meet at its
corners. ``postprocess.py`` builds the B-rep from these candidates.

Test-time rotation augmentation (``--rotations``, default all 24): the mesh is rotated by
octahedral rotation ``M[k]`` before sampling (seeded per rotation), so the network sees pose
``k``; the output is written under ``<out>/tmp/<name>_<k:02d>/`` so the post-processing stage
(``postprocess.py``) undoes it with ``inv(M[k])``. Index 3 is the identity (canonical pose).
The 24 rotations give the assembler 24 candidate reconstructions to seal into a valid solid.

Pipeline:  cluster.ply --rotate M[k] + per-face sample--> face_sample_points [F, P, 6]
           --Parametrizer encode + adjacency (pruned) + latent optimisation + decode-->
           per-face surface grids + intersection edges + topology
           --> post.npz   (handed to postprocess.py to build the B-rep STEP)

Per (shape, rotation), results go to ``<out>/tmp/<name>_<k>/``:
    input_faces.ply   the per-face sampled points (colored by face label)
    recon_faces.ply   reconstructed B-rep faces (16x16 surface grids, colored by face)
    recon_edges.ply   reconstructed intersection edges (sampled curve points)
    post.npz          pred_face (F,16,16,6), pred_edge (E,16,3), pred_edge_face_connectivity (E,3)
The optimiser's loss terms per candidate (before / after) go to ``<out>/latent_opt.jsonl``.

The optimisation takes 10-15 s per candidate on a GPU; several processes can share one GPU
through CUDA MPS, each with ``--shard k/n`` (``run_pipeline.sh`` does this).

Usage:
    python rebuild.py --input /path/to/<name>/cluster.ply --out output_brep
    python rebuild.py --input /path/to/recon_dir          --out output_brep   # each <name>/ has cluster.ply
    python rebuild.py --input /path/to/recon_dir          --out output_brep --shard 0/16
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import open3d as o3d
from scipy.spatial.transform import Rotation

from rebuild_model import Parametrizer, normalize_coord0516, denormalize_coord0516, hermite_sample

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

# Latent optimisation: loss weights and fixed parameters.
LOPT_WEIGHTS = {"data": 1.0, "twin": 1.0, "ef": 1.0, "cover": 1.0, "move": 1.0, "vert": 1.0, "reg": 1e-3}
LOPT_DELTA = 0.05       # boundary points of a pair (i, j): points of face i within this of face j's points, and back
LOPT_MIN_BPTS = 4       # pairs with fewer boundary points are unobserved: no edge terms (kept in the output)
LOPT_VERT_CAP = 0.1     # end-point distances above this give no gradient

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


def touching_faces(mesh, labels, delta=0.05, min_verts=1):
    """Face pairs (indices into the sorted cluster labels) whose clusters are NOT clearly apart in the segmented
    mesh: >= min_verts mesh vertices of one cluster within delta of a vertex of the other. Clusters are separated by
    an unlabelled band (label -1) about 2-3 voxels wide, so adjacent faces are ~0.02 apart; delta 0.05 with one
    vertex only drops pairs that are clearly separated. (A count rule, delta 0.03 / >= 10 vertices, also dropped real
    contacts of slivers and short boundaries: on 95 test shapes it gained 1 shape and lost 2.)"""
    from scipy.spatial import cKDTree
    uniq = np.unique(labels)
    uniq = uniq[uniq != -1]
    index = {l: i for i, l in enumerate(uniq)}
    vlab = np.full(len(mesh.vertices), -1)
    for t, l in enumerate(labels):
        if l != -1:
            vlab[mesh.faces[t]] = index[l]
    keep = np.where(vlab >= 0)[0]
    pts, lab = np.asarray(mesh.vertices)[keep], vlab[keep]
    counts = {}                     # (a, b): vertices of cluster a with a vertex of cluster b within delta
    for i, nb in enumerate(cKDTree(pts).query_ball_point(pts, delta)):
        for b in set(lab[nb]):
            if b != lab[i]:
                counts[(lab[i], b)] = counts.get((lab[i], b), 0) + 1
    return {(min(a, b), max(a, b)) for (a, b), c in counts.items() if c >= min_verts}


# ---------------------------------------------------------------- test-time latent optimisation
# Faces are decoded from the per-face latent face_z as 16x16 grids G_i; the half-edge (i, j) is decoded as 16 uv
# samples ON face i's grid, E_ij = G_i(uv_ij), its twin E_ji on face j. Optimise face_z (Adam) on
#   w_data  mean d(p, G_i)              the face's sampled points stay on its grid
#   w_twin  chamfer(E_ij, E_ji)         the two half-edges of an adjacency coincide
#   w_ef    mean d(E_ij, G_j)           each half-edge also lies on the other face
#   w_cover mean d(b, E_ij)             observed boundary points of (i, j) lie on the edge (else edges shrink)
#   w_move  mean |G_i - G_i0|           grids stay where they were (else faces grow into their neighbours)
#   w_vert  wire closure + twin ends    corners: a face's half-edges meet, twins share their end points
#   w_reg   mean (z - z0)^2
# d(., G) = distance to the grid upsampled bilinearly to 48x48.

def _lopt_decode(model, z, fbb, pairs):
    dec = model.decode_face({"face_z": z, "face_bbox": fbb}, None)
    face = denormalize_coord0516(dec["face_norm"], dec["face_bbox"])                              # [F,16,16,6]
    if pairs.shape[0] == 0:
        return face, torch.zeros((0, 16, 3), device=z.device)
    feat = model.inter(z[pairs].reshape(pairs.shape[0], -1))
    uv = model.query_edge(feat, None)[..., :2]                                                    # [P,16,2]
    edge = hermite_sample(face[pairs[:, 0]][..., :3].permute(0, 3, 1, 2), uv).permute(0, 2, 1)   # [P,16,3]
    return face, edge


def _dense(face_xyz, res=48):
    g = F.interpolate(face_xyz.permute(0, 3, 1, 2), size=(res, res), mode="bilinear", align_corners=True)
    return g.permute(0, 2, 3, 1).reshape(face_xyz.shape[0], -1, 3)                               # [F,res*res,3]


def _boundary_points(fit, pairs, delta, max_pts=512):
    """Observed boundary of each pair (i, j): points of face i within delta of face j's points, and vice versa.
    Returns padded [P, M, 3], mask [P, M], count [P]."""
    out, cnt = [], []
    for a, b in pairs.tolist():
        if a == b:
            out.append(fit.new_zeros((0, 3)))
            cnt.append(0)
            continue
        d = torch.cdist(fit[a], fit[b])
        bp = torch.cat((fit[a][d.min(1)[0] < delta], fit[b][d.min(0)[0] < delta]), 0)[:max_pts]
        out.append(bp)
        cnt.append(bp.shape[0])
    m = max(max(cnt), 1)
    pad = fit.new_zeros((len(out), m, 3))
    mask = torch.zeros((len(out), m), dtype=torch.bool, device=fit.device)
    for t, bp in enumerate(out):
        pad[t, :bp.shape[0]] = bp
        mask[t, :bp.shape[0]] = True
    return pad, mask, torch.tensor(cnt, device=fit.device)


def _lopt_terms(face, edge, pts, pairs, twin, valid, bpts, bmask):
    g = _dense(face[..., :3])
    d_data = torch.cdist(pts, g).min(-1)[0].mean()
    z0 = d_data * 0
    if pairs.shape[0] == 0 or not valid.any():
        return d_data, z0, z0, z0
    d_ef = torch.cdist(edge[valid], g[pairs[valid, 1]]).min(-1)[0].mean()
    has = (twin >= 0) & valid
    d_twin = z0
    if has.any():
        d = torch.cdist(edge[has], edge[twin[has]])
        d_twin = 0.5 * (d.min(-1)[0].mean() + d.min(-2)[0].mean())
    e = F.interpolate(edge[valid].permute(0, 2, 1), size=64, mode="linear", align_corners=True).permute(0, 2, 1)
    dc = torch.cdist(bpts[valid], e).min(-1)[0]                  # each boundary point -> nearest edge point
    msk = bmask[valid]
    d_cover = (dc * msk).sum() / msk.sum().clamp(min=1)
    return d_data, d_twin, d_ef, d_cover


def _wire_groups(pairs, valid):
    """Per face i: the indices of its valid half-edges (i, j); their end points form face i's wire."""
    groups = []
    for f in torch.unique(pairs[valid, 0]).tolist():
        idx = torch.nonzero(valid & (pairs[:, 0] == f)).squeeze(1)
        if idx.numel() >= 2:
            groups.append(idx)
    return groups


def _vert_term(edge, groups, twin, valid, cap):
    """(1) Wire closure: every end point of a face's half-edge meets another end point of the same face's half-edges
    (its own other end counts, so a closed edge is not pulled). (2) The end points of twin half-edges coincide
    (either orientation). Distances above cap give no gradient."""
    def dist(a, b):     # eps inside the sqrt: finite gradient at 0
        return torch.sqrt(((a - b) ** 2).sum(-1) + 1e-12)

    ends = edge[:, [0, -1]]                                                  # [P,2,3]
    zero = edge.sum() * 0
    d_wire = zero
    if groups:
        m = []
        for g in groups:
            e = ends[g].reshape(-1, 3)
            d = dist(e[:, None], e[None]) + torch.eye(e.shape[0], device=e.device) * 1e6
            m.append(d.min(1)[0].clamp(max=cap))
        d_wire = torch.cat(m).mean()
    has = (twin >= 0) & valid
    d_tend = zero
    if has.any():
        a, b = ends[has], ends[twin[has]]
        s1 = dist(a[:, 0], b[:, 0]) + dist(a[:, 1], b[:, 1])
        s2 = dist(a[:, 0], b[:, 1]) + dist(a[:, 1], b[:, 0])
        d_tend = 0.5 * torch.minimum(s1, s2).clamp(max=2 * cap).mean()
    return d_wire + d_tend


def _prune_pairs(pairs, touching):
    """Drop the classifier's pairs whose clusters do not touch (touching_faces); seam pairs (a == b) are kept and
    moved to the end (the seeded rebuild depends on the edge order)."""
    pl = [tuple(p) for p in pairs.tolist()]
    out = [p for p in pl if p[0] != p[1] and (min(p), max(p)) in touching] + [p for p in pl if p[0] == p[1]]
    return torch.tensor(out, dtype=torch.long, device=pairs.device).reshape(-1, 2), len(pl) - len(out)


def latent_optimise(model, fsp, fbb, fit, touching, steps, lr, w):
    """Encode, take the classifier's adjacency (pruned with ``touching``), optimise face_z, decode.
    Returns (results in post.npz format, n_pruned, log record)."""
    n = fsp.shape[0]
    data = {"face_sample_points": fsp, "face_input_bbox": fbb,
            "face_attn_mask": torch.zeros((n, n), dtype=torch.bool, device=fsp.device)}
    z0 = model.encode(data, True)["face_z"]
    pairs = model.decode_edge(z0)["face_index"]
    n_pruned = 0
    if touching is not None:
        pairs, n_pruned = _prune_pairs(pairs, touching)
    key = {(a, b): t for t, (a, b) in enumerate(pairs.tolist())}
    twin = torch.tensor([key.get((b, a), -1) if a != b else -1 for a, b in pairs.tolist()],
                        device=fsp.device, dtype=torch.long)
    bpts, bmask, bcnt = _boundary_points(fit, pairs, LOPT_DELTA)
    valid = (pairs[:, 0] != pairs[:, 1]) & (bcnt >= LOPT_MIN_BPTS)  # unobserved pairs: likely false, no edge terms
    groups = _wire_groups(pairs, valid)

    def all_terms(face, edge):
        return [float(x) for x in _lopt_terms(face, edge, fit, pairs, twin, valid, bpts, bmask)] + \
               [float(_vert_term(edge, groups, twin, valid, LOPT_VERT_CAP))]

    face, edge = _lopt_decode(model, z0, fbb, pairs)
    before = all_terms(face, edge)
    g0 = face[..., :3]
    t0 = time.time()
    with torch.enable_grad():
        z = z0.clone().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=lr)
        for _ in range(steps):
            face, edge = _lopt_decode(model, z, fbb, pairs)
            d_data, d_twin, d_ef, d_cover = _lopt_terms(face, edge, fit, pairs, twin, valid, bpts, bmask)
            loss = (w["data"] * d_data + w["twin"] * d_twin + w["ef"] * d_ef + w["cover"] * d_cover
                    + w["reg"] * ((z - z0) ** 2).mean())
            if w["move"] > 0:
                loss = loss + w["move"] * torch.linalg.norm(face[..., :3] - g0, dim=-1).mean()
            if w["vert"] > 0:
                loss = loss + w["vert"] * _vert_term(edge, groups, twin, valid, LOPT_VERT_CAP)
            opt.zero_grad()
            loss.backward()
            opt.step()
    z = z.detach()
    face, edge = _lopt_decode(model, z, fbb, pairs)
    conn = torch.cat((torch.arange(pairs.shape[0], device=pairs.device)[:, None], pairs), 1).float()
    results = {"pred_face": face.float().cpu().numpy(), "pred_edge": edge.float().cpu().numpy(),
               "pred_edge_face_connectivity": conn.cpu().numpy()}
    rec = {"faces": n, "pairs": int(pairs.shape[0]), "pruned": n_pruned, "unobserved": int(((pairs[:, 0] != pairs[:, 1])
           & ~valid).sum()), "before": before, "after": all_terms(face, edge),
           "grid_move": float(torch.linalg.norm(face[..., :3] - g0, dim=-1).mean()), "opt_s": round(time.time() - t0, 2)}
    return results, n_pruned, rec


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
    ap.add_argument("--rotations", default="all",
                    help="Octahedral rotation ids for test-time aug: 'all' (0-23), '3' (canonical), or e.g. '3,7,11'.")
    ap.add_argument("--num_input_points", type=int, default=100, help="Points sampled per face.")
    ap.add_argument("--num_samples", type=int, default=None, help="Cap the number of shapes processed.")
    ap.add_argument("--lopt_steps", type=int, default=200, help="Latent optimisation: Adam steps.")
    ap.add_argument("--lopt_lr", type=float, default=2e-3, help="Latent optimisation: Adam learning rate.")
    ap.add_argument("--lopt_w", nargs="*", default=[], metavar="NAME=VALUE",
                    help=f"Override latent optimisation loss weights (default {LOPT_WEIGHTS}).")
    ap.add_argument("--num_fit_points", type=int, default=400,
                    help="Latent optimisation: points per face for the data and boundary terms.")
    ap.add_argument("--shard", default="0/1",
                    help="k/n: process shapes k, k+n, k+2n, ... (several processes on one GPU, e.g. under CUDA MPS).")
    args = ap.parse_args()
    weights = dict(LOPT_WEIGHTS)
    for kv in args.lopt_w:
        k, v = kv.split("=", 1)
        if k not in weights:
            ap.error(f"unknown --lopt_w weight {k!r}; one of {sorted(weights)}")
        weights[k] = float(v)
    shard_k, shard_n = (int(x) for x in args.shard.split("/"))

    rotations = parse_rotations(args.rotations)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Parametrizer(dict(MODEL_CFG))
    missing, unexpected = model.load_state_dict(load_state(args.checkpoint), strict=False)
    print("Checkpoint loaded (0 missing / 0 unexpected keys)." if not missing and not unexpected
          else f"WARNING: {len(missing)} missing / {len(unexpected)} unexpected keys")
    model = model.to(device).eval()
    for p in model.parameters():     # the latent optimisation changes the latents only
        p.requires_grad_(False)

    inputs = collect_inputs(args.input)
    if args.num_samples:
        inputs = inputs[: args.num_samples]
    if not inputs:
        raise FileNotFoundError(f"No cluster.ply found under {args.input}")
    inputs = inputs[shard_k::shard_n]
    print(f"{len(inputs)} shape(s) x {len(rotations)} rotation(s), latent optimisation {args.lopt_steps} steps, "
          f"weights {weights}.")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    log = open(Path(args.out) / (f"latent_opt_{shard_k}.jsonl" if shard_n > 1 else "latent_opt.jsonl"), "a")

    for name, cluster_ply in inputs:
        stem = str(name).split("_")[0]   # bare shape id; postprocess reads the rotation from "_<k>"
        mesh, labels = read_cluster(cluster_ply)
        touching = touching_faces(mesh, labels)
        for k in rotations:
            m = mesh.copy()
            T = np.eye(4); T[:3, :3] = ROT[k]
            m.apply_transform(T)
            o3d.utility.random.seed(1000 * k + 7)           # seeded samples: a candidate is reproducible
            fsp, fbb = sample_faces(m, labels, args.num_input_points)
            if fsp is None or fsp.shape[0] < 2:
                print(f"[{stem} rot{k:02d}] <2 faces, skipping")
                continue
            fsp, fbb = fsp.to(device), fbb.to(device)
            n = fsp.shape[0]
            o3d.utility.random.seed(1000 * k + 8)
            fit = sample_faces(m, labels, args.num_fit_points)[0][..., :3].to(device)
            results, n_pruned, rec = latent_optimise(model, fsp, fbb, fit, touching, args.lopt_steps,
                                                     args.lopt_lr, weights)
            # The 24 rotation candidates are intermediate -> keep them under <out>/tmp/.
            od = Path(args.out) / "tmp" / f"{stem}_{k:02d}"
            nf, ne = save_outputs(results, fsp, od)
            rec["cand"] = f"{stem}_{k:02d}"
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"[{stem} rot{k:02d}] {n} faces -> {nf} recon faces, {ne} edges ({n_pruned} pruned) -> {od}\n"
                  f"    data/twin/edge-face/cover/vert " + " ".join(
                      f"{b:.4f}->{a:.4f}" for b, a in zip(rec["before"], rec["after"])) + f", {rec['opt_s']}s",
                  flush=True)

    log.close()
    print(f"\nDone. Candidates in {(Path(args.out) / 'tmp').resolve()}")
    print(f"Next:  python postprocess.py --input {Path(args.out)}")


if __name__ == "__main__":
    main()
