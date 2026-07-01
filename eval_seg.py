"""Evaluate a reconstructed B-rep against ground-truth B-rep segmentation (ABC-20k).

The reconstructed solid produced by the pipeline (``postprocess.py`` ->
``<name>.step`` / ``recon_brep.step``) is decomposed with OpenCASCADE into its
faces / edges / vertices and their topology, then compared against the
precomputed ground truth.

Reported per structural element (surface / edge / vertex):
    F1, precision, recall via Hungarian matching of point groups
    (a pred/gt label pair matches when its symmetric point-to-point distance
     is below ``MATCH_THRESHOLD``), plus the chamfer distance of the union point
    cloud.
Reported for topology:
    face-edge and edge-vertex adjacency F1 / precision / recall, computed after
    mapping matched pred labels onto their gt counterparts.

Ground truth (one shape per ``<name>``) is built by ``prepare_seg_gt.py`` and
lives under ``--gt`` as::

    <name>.ply         surface points, per-vertex integer ``label`` = gt face id
    <name>_edge.ply    edge points,    per-vertex integer ``label`` = gt edge id
    <name>_vertex.ply  corner points   (one point per gt vertex)
    <name>_adj.npz     face_edge / edge_vertex boolean adjacency matrices

Both ``--gt`` and ``--pred`` may be local directories or ``s3://`` prefixes.

Examples::

    # quick test on 100 shapes, predictions + gt on S3
    python eval_seg.py --pred s3://.../sig_baseline/dualbrep_out/ --limit 100

    # full test set, local predictions
    python eval_seg.py --pred output_pipeline/brep --gt /data/gt_seg \\
        --list data/final_test.txt --out eval_out

    # re-print the summary from a previous run's cached per-shape results
    python eval_seg.py --report-only --out eval_out
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import boto3
import numpy as np
import ray
import trimesh
from botocore.config import Config
from cloudpathlib import S3Client, S3Path
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brep_post.occ_utils import (  # noqa: E402
    BRep_Tool,
    BRepAdaptor_Curve,
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_VERTEX,
    TopTools_IndexedMapOfShape,
    get_curve_length,
    get_primitives,
    get_triangulations,
    read_step_file,
    topexp,
)

# ---------------------------------------------------------------------------- #
# Defaults (overridable on the command line).
# ---------------------------------------------------------------------------- #
DEFAULT_GT = (
    "s3://autodesk-adpcdl-965535024567-p-ue1-internal-pdms-large-models/"
    "private/BrepDirectGen/yl-voronoi/abc20k/gt_seg/"
)
DEFAULT_LIST = (
    "s3://autodesk-adpcdl-965535024567-p-ue1-internal-pdms-large-models/"
    "private/BrepDirectGen/yl-voronoi/abc20k/final_test.txt"
)

# Sampling densities (must match prepare_seg_gt.py so pred/gt are comparable).
NUM_PER_M = 100     # edge samples per unit length
NUM_PER_M2 = 1000   # surface samples per unit area
# A pred/gt element pair counts as a match when its symmetric mean point
# distance is below this threshold (shapes are normalized to the unit box).
MATCH_THRESHOLD = 0.1

# Candidate locations of the predicted STEP inside ``<pred>/<name>/`` (or flat).
STEP_CANDIDATES = (
    "{name}/pp/recon_brep.step",
    "{name}/recon_brep.step",
    "{name}.step",
    "brep/{name}.step",
)


def _resolve_path(path_str, client=None):
    """Return an S3Path (for ``s3://``) or a local Path."""
    if str(path_str).startswith("s3://"):
        return S3Path(str(path_str), client=client)
    return Path(path_str)


def _open_text_lines(path_str):
    """Read a text file (local or S3) and return its stripped lines."""
    p = _resolve_path(path_str)
    with p.open() as f:
        return [line.strip() for line in f if line.strip()]


def download_file(remote_path, tmp_dir):
    """Download a (possibly S3) file into ``tmp_dir`` and return the local path."""
    remote_path = S3Path(str(remote_path), client=remote_path.client) \
        if isinstance(remote_path, S3Path) else Path(remote_path)
    local = Path(tmp_dir) / remote_path.name
    if isinstance(remote_path, S3Path):
        remote_path.download_to(str(local))
        return local
    return remote_path


def load_labeled_ply(v_path, tmp_dir):
    """Load a labeled point ply -> (Nx3 points, N integer labels)."""
    if isinstance(v_path, S3Path):
        v_path = download_file(v_path, tmp_dir)
    pc = trimesh.load(v_path, process=False)
    labels = np.array([item[3] for item in pc.metadata["_ply_raw"]["vertex"]["data"]])
    return np.array(pc.vertices), labels


# ---------------------------------------------------------------------------- #
# Metrics.
# ---------------------------------------------------------------------------- #
def compute_metric(pred_pc, gt_pc, pred_labels, gt_labels):
    """Match predicted element groups to gt groups and score F1/precision/recall.

    Builds the ``(n_pred, n_gt)`` cost matrix of symmetric mean nearest-neighbour
    distances between each pred group and each gt group, solves the optimal
    assignment, and counts a pair correct when its distance is below
    ``MATCH_THRESHOLD``. Returns ``(metric, good_match)`` where ``metric`` is
    ``[f1, precision, recall, num_pred, num_gt]`` and ``good_match`` maps a
    matched pred-group index -> gt-group index (used for the topology metric).
    """
    num_pred_labels = len(np.unique(pred_labels))
    num_gt_labels = len(np.unique(gt_labels))

    u_pred_labels = np.unique(pred_labels)
    u_gt_labels = np.unique(gt_labels)
    pred_groups = {lbl: pred_pc[pred_labels == lbl] for lbl in u_pred_labels}
    gt_groups = {lbl: gt_pc[gt_labels == lbl] for lbl in u_gt_labels}

    pred_trees = {lbl: cKDTree(pts) for lbl, pts in pred_groups.items()}
    gt_trees = {lbl: cKDTree(pts) for lbl, pts in gt_groups.items()}

    label_matrix = np.zeros((len(u_pred_labels), len(u_gt_labels)))
    for i, p_lbl in enumerate(u_pred_labels):
        p_subset = pred_groups[p_lbl]
        for j, g_lbl in enumerate(u_gt_labels):
            g_subset = gt_groups[g_lbl]
            if len(p_subset) == 0 or len(g_subset) == 0:
                continue
            dists_p_to_g, _ = gt_trees[g_lbl].query(p_subset, k=1)
            dists_g_to_p, _ = pred_trees[p_lbl].query(g_subset, k=1)
            label_matrix[i, j] = (np.mean(dists_p_to_g) + np.mean(dists_g_to_p)) / 2

    row_ind, col_ind = linear_sum_assignment(label_matrix)

    num_correct = 0
    good_match = {}
    for i in range(row_ind.shape[0]):
        if label_matrix[row_ind[i], col_ind[i]] < MATCH_THRESHOLD:
            num_correct += 1
            good_match[row_ind[i]] = col_ind[i]

    precision = num_correct / (num_pred_labels + 1e-6)
    recall = num_correct / (num_gt_labels + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    metric = np.array([f1, precision, recall, num_pred_labels, num_gt_labels])
    return metric, good_match


def chamfer(pred_pc, gt_pc):
    """Symmetric mean nearest-neighbour distance between two point clouds."""
    dist1, _ = cKDTree(pred_pc).query(gt_pc, k=1)
    dist2, _ = cKDTree(gt_pc).query(pred_pc, k=1)
    return (dist1.mean() + dist2.mean()) / 2


def eval_topo(pred_adj, gt_adj, row_match, col_match):
    """F1/precision/recall of a predicted adjacency matrix vs gt.

    ``pred_adj`` (bool ``[n_pred_row, n_pred_col]``) is remapped onto gt indices
    using the two match dicts before being compared to ``gt_adj``.
    """
    mapped = np.zeros_like(gt_adj)
    for i, j in zip(*np.where(pred_adj)):
        if i not in row_match or j not in col_match:
            continue
        mapped[row_match[i], col_match[j]] = 1
    tp = (mapped & gt_adj).sum()
    fp = (mapped & (~gt_adj)).sum()
    fn = ((~mapped) & gt_adj).sum()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return np.array([f1, precision, recall])


# ---------------------------------------------------------------------------- #
# Per-shape evaluation.
# ---------------------------------------------------------------------------- #
def _find_pred_step(pred_path, name, tmp_dir):
    """Return a local path to the predicted STEP for ``name``, or None."""
    for tmpl in STEP_CANDIDATES:
        cand = pred_path / tmpl.format(name=name)
        if cand.exists():
            return download_file(cand, tmp_dir) if isinstance(cand, S3Path) else cand
    return None


def _sample_step(step_file):
    """Decompose a STEP solid -> labeled surface / edge / vertex clouds + topology.

    Returns ``(surf_pc, surf_lbl, edge_pc, edge_lbl, vert_pc, fe_adj, ev_adj)``.
    """
    import open3d as o3d

    shape = read_step_file(str(step_file))
    faces = get_primitives(shape, TopAbs_FACE)
    edges = get_primitives(shape, TopAbs_EDGE, v_remove_half=True)
    verts = get_primitives(shape, TopAbs_VERTEX, v_remove_half=True)

    # Faces: poisson-disk sample each face, label by face index.
    surf = []
    for idx, face in enumerate(faces):
        v, f = get_triangulations(face)
        if len(f) == 0:
            continue
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(v)
        mesh.triangles = o3d.utility.Vector3iVector(f)
        area = mesh.get_surface_area()
        pcd = mesh.sample_points_poisson_disk(max(10, int(area * NUM_PER_M2)))
        pc = np.asarray(pcd.points)
        surf.append(np.concatenate([pc, idx * np.ones((pc.shape[0], 1))], axis=1))
    surf = np.concatenate(surf, axis=0)
    surf_pc, surf_lbl = surf[:, :3], surf[:, 3]

    # Edges: sample along each curve, label by edge index.
    edge = []
    for idx, e in enumerate(edges):
        curve = BRepAdaptor_Curve(e)
        length = get_curve_length(e)
        sample_u = np.linspace(curve.FirstParameter(), curve.LastParameter(),
                               num=max(10, int(length * NUM_PER_M)))
        pts = []
        for u in sample_u:
            p = curve.Value(u)
            pts.append([p.X(), p.Y(), p.Z()])
        pts = np.array(pts, dtype=np.float32)
        edge.append(np.concatenate([pts, idx * np.ones((pts.shape[0], 1))], axis=1))
    edge = np.concatenate(edge, axis=0)
    edge_pc, edge_lbl = edge[:, :3], edge[:, 3]

    # Vertices: corner points.
    vert_pc = np.array([[BRep_Tool.Pnt(v).X(), BRep_Tool.Pnt(v).Y(), BRep_Tool.Pnt(v).Z()]
                        for v in verts], dtype=np.float32)

    # Topology adjacency.
    fe_adj = np.zeros((len(faces), len(edges)), dtype=bool)
    ev_adj = np.zeros((len(edges), len(verts)), dtype=bool)
    face_maps = []
    for face in faces:
        fmap = TopTools_IndexedMapOfShape()
        topexp.MapShapes(face, TopAbs_EDGE, fmap)
        face_maps.append(fmap)
    edge_maps = []
    for idx, e in enumerate(edges):
        for id_face in range(len(faces)):
            if face_maps[id_face].Contains(e):
                fe_adj[id_face, idx] = True
        emap = TopTools_IndexedMapOfShape()
        topexp.MapShapes(e, TopAbs_VERTEX, emap)
        edge_maps.append(emap)
    for idx, v in enumerate(verts):
        for id_edge in range(len(edges)):
            if edge_maps[id_edge].Contains(v):
                ev_adj[id_edge, idx] = True

    return surf_pc, surf_lbl, edge_pc, edge_lbl, vert_pc, fe_adj, ev_adj


def _empty_result(num_gt_surf, num_gt_edge, num_gt_vert):
    """Zero metrics for a failed / missing prediction."""
    return dict(
        surface_metric=np.array([0, 0, 0, 0, num_gt_surf]), cd_surface=1.0,
        edge_metric=np.array([0, 0, 0, 0, num_gt_edge]), cd_edge=1.0,
        vertex_metric=np.array([0, 0, 0, 0, num_gt_vert]), cd_vertex=1.0,
        metric_fe=np.array([0, 0, 0]), metric_ev=np.array([0, 0, 0]),
    )


def process_file(name, pred_path_str, gt_path_str, out_dir, scale=1.0):
    """Evaluate one shape; cache ``<name>_eval.npz`` in ``out_dir`` and return metrics."""
    retry_config = Config(retries={"max_attempts": 10, "mode": "adaptive"})
    session = boto3.Session()
    session._session.set_default_client_config(retry_config)
    s3_client = S3Client(boto3_session=session)

    pred_path = _resolve_path(pred_path_str, s3_client)
    gt_path = _resolve_path(gt_path_str, s3_client)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        while True:
            try:
                # ---- Ground truth ----
                gt_surf_pc, gt_surf_lbl = load_labeled_ply(gt_path / f"{name}.ply", tmp)
                gt_edge_pc, gt_edge_lbl = load_labeled_ply(gt_path / f"{name}_edge.ply", tmp)
                gt_vert_pc, gt_vert_lbl = load_labeled_ply(gt_path / f"{name}_vertex.ply", tmp)
                num_gt_surf = len(np.unique(gt_surf_lbl))
                num_gt_edge = len(np.unique(gt_edge_lbl))
                num_gt_vert = len(np.unique(gt_vert_lbl))
                gt_topo_file = download_file(gt_path / f"{name}_adj.npz", tmp) \
                    if isinstance(gt_path, S3Path) else gt_path / f"{name}_adj.npz"
                gt_topo = np.load(gt_topo_file)
                gt_fe, gt_ev = gt_topo["face_edge"], gt_topo["edge_vertex"]

                # ---- Prediction ----
                result = _empty_result(num_gt_surf, num_gt_edge, num_gt_vert)
                try:
                    step_file = _find_pred_step(pred_path, name, tmp)
                    if step_file is None:
                        raise FileNotFoundError(f"no STEP found for {name}")
                    (surf_pc, surf_lbl, edge_pc, edge_lbl,
                     vert_pc, fe_adj, ev_adj) = _sample_step(step_file)
                    surf_pc = surf_pc * scale
                    edge_pc = edge_pc * scale
                    vert_pc = vert_pc * scale

                    m_surf, face_match = compute_metric(surf_pc, gt_surf_pc, surf_lbl, gt_surf_lbl)
                    m_edge, edge_match = compute_metric(edge_pc, gt_edge_pc, edge_lbl, gt_edge_lbl)
                    m_vert, vert_match = compute_metric(
                        vert_pc, gt_vert_pc, np.arange(len(vert_pc)), gt_vert_lbl)

                    result.update(
                        surface_metric=m_surf, cd_surface=chamfer(surf_pc, gt_surf_pc),
                        edge_metric=m_edge, cd_edge=chamfer(edge_pc, gt_edge_pc),
                        vertex_metric=m_vert, cd_vertex=chamfer(vert_pc, gt_vert_pc),
                        metric_fe=eval_topo(fe_adj, gt_fe, face_match, edge_match),
                        metric_ev=eval_topo(ev_adj, gt_ev, edge_match, vert_match),
                    )
                except Exception as e:  # noqa: BLE001 - a bad prediction scores zero, not a crash
                    print(f"[{name}] prediction failed: {e}")

                np.savez(out_dir / f"{name}_eval.npz", **result)
                return result
            except Exception as e:  # noqa: BLE001
                if "ListObjects" in str(e) or "Cannot download because path does" in str(e):
                    print(f"[{name}] network error, retrying...")
                    time.sleep(5)
                    continue
                print(f"[{name}] fatal: {e}")
                raise


# ---------------------------------------------------------------------------- #
# Aggregation / reporting.
# ---------------------------------------------------------------------------- #
def _report_block(title, metric, cd):
    print(f"======{title}======")
    print("F1:        ", metric[:, 0].mean())
    print("Precision: ", metric[:, 1].mean())
    print("Recall:    ", metric[:, 2].mean())
    print("Failure:   ", np.logical_and(metric[:, 0] < 0.001, metric[:, 1] < 0.001).sum() / len(metric))
    print("Perfect:   ", np.logical_and(metric[:, 0] > 0.999, metric[:, 1] > 0.999).sum() / len(metric))
    print("num_pred:  ", metric[:, 3].mean())
    print("num_gt:    ", metric[:, 4].mean())
    print("Chamfer:   ", np.median(cd))


def report(results):
    """Print aggregated metrics and return the single-line summary string."""
    m_surf = np.array([r["surface_metric"] for r in results])
    cd_surf = np.array([r["cd_surface"] for r in results])
    m_edge = np.array([r["edge_metric"] for r in results])
    cd_edge = np.array([r["cd_edge"] for r in results])
    m_vert = np.array([r["vertex_metric"] for r in results])
    cd_vert = np.array([r["cd_vertex"] for r in results])
    m_fe = np.array([r["metric_fe"] for r in results])
    m_ev = np.array([r["metric_ev"] for r in results])

    _report_block("Surface", m_surf, cd_surf)
    _report_block("Edge", m_edge, cd_edge)
    _report_block("Vertex", m_vert, cd_vert)
    print("======Topo======")
    print("Face-Edge   F1/P/R: ", m_fe[:, 0].mean(), m_fe[:, 1].mean(), m_fe[:, 2].mean())
    print("Edge-Vertex F1/P/R: ", m_ev[:, 0].mean(), m_ev[:, 1].mean(), m_ev[:, 2].mean())

    summary = " ".join(str(x) for x in [
        m_surf[:, 0].mean(), m_surf[:, 1].mean(), m_surf[:, 2].mean(),
        m_surf[:, 3].mean(), m_surf[:, 4].mean(), np.median(cd_surf),
        m_edge[:, 0].mean(), m_edge[:, 1].mean(), m_edge[:, 2].mean(),
        m_edge[:, 3].mean(), m_edge[:, 4].mean(), np.median(cd_edge),
        m_vert[:, 0].mean(), m_vert[:, 1].mean(), m_vert[:, 2].mean(),
        m_vert[:, 3].mean(), m_vert[:, 4].mean(), np.median(cd_vert),
        m_fe[:, 0].mean(), m_fe[:, 1].mean(), m_fe[:, 2].mean(),
        m_ev[:, 0].mean(), m_ev[:, 1].mean(), m_ev[:, 2].mean(),
    ])
    print("\n# Surface F1,P,R,num_pred,num_gt,CD | Edge ... | Vertex ... | FE F1,P,R | EV F1,P,R")
    print(summary)
    return summary


def load_cached_results(out_dir, names):
    """Read previously cached ``<name>_eval.npz`` files from ``out_dir``."""
    results = []
    for name in names:
        f = Path(out_dir) / f"{name}_eval.npz"
        if not f.exists():
            print(f"[{name}] cached result missing, skipping")
            continue
        d = np.load(f)
        results.append({k: d[k] for k in d.files})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", help="prediction root (local dir or s3:// prefix)")
    parser.add_argument("--gt", default=DEFAULT_GT, help="ground-truth root (local or s3://)")
    parser.add_argument("--list", dest="name_list", default=DEFAULT_LIST,
                        help="text file of shape names, one per line (local or s3://)")
    parser.add_argument("--out", default="eval_out",
                        help="local dir for per-shape results + summary (default: eval_out)")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N shapes (for quick tests)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="scale applied to predicted geometry before matching")
    parser.add_argument("--num-cpus", type=int, default=None, help="Ray CPU count")
    parser.add_argument("--serial", action="store_true", help="run without Ray (debug)")
    parser.add_argument("--report-only", action="store_true",
                        help="skip evaluation; aggregate cached results from --out")
    args = parser.parse_args()

    names = [line.split()[0] for line in _open_text_lines(args.name_list)]
    if args.limit is not None:
        names = names[:args.limit]
    print(f"In total {len(names)} shapes to evaluate.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        results = load_cached_results(out_dir, names)
    elif args.serial:
        results = [process_file(n, args.pred, args.gt, str(out_dir), args.scale)
                   for n in tqdm(names)]
    else:
        ray.init(num_cpus=args.num_cpus)
        remote = ray.remote(num_cpus=1)(process_file)
        tasks = [remote.remote(n, args.pred, args.gt, str(out_dir), args.scale) for n in names]
        results = [ray.get(t) for t in tqdm(tasks)]
        ray.shutdown()

    if not results:
        print("No results to report.")
        return
    summary = report(results)
    (out_dir / "summary.txt").write_text(summary + "\n")
    print(f"\nSummary written to {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
