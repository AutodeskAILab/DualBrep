"""Collect the implicit sample points for the dual-field VAE from a STEP + Voronoi field.

Given a STEP solid and the Voronoi mesh produced by ``Voronoi/calculate_voronoi``
(``voronoi.ply``), this samples the surface / edge / Voronoi point clouds and the
SDF / UDF query fields, and writes the ``.npz`` consumed by ``ae_reconstruct.py``
(``dataset.input=implicit``).

The STEP is normalized into the same [-0.9, 0.9] box that ``calculate_voronoi``
uses, so the two frames line up.

Output ``.npz`` keys (matching ``dataset.py``):
    surface_points  (100000, 6)  xyz + normal, sampled on faces
    edge_points     (100000, 6)  xyz + tangent, sampled on B-rep edges
    voronoi_points  (Nv, 6)      xyz + normal, sampled on the Voronoi field
    query_surface_points / _sdf / _udf     surface query xyz + signed/unsigned dist
    query_edge_points    / _sdf / _udf     edge query xyz + signed/unsigned dist

Examples::

    # single shape
    python prepare_implicit.py --step 00000164.step --voronoi voronoi.ply --out 00000164.npz

    # batch: one <name>.step per shape + one <name>.ply Voronoi mesh
    python prepare_implicit.py --step-dir steps/ --voronoi-dir voronoi/ --out-dir implicit/
"""

import argparse
import os
import random
import sys
from pathlib import Path

import igl
import numpy as np
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.gp import gp_Vec
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Extend.DataExchange import read_step_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brep_post.occ_utils import (  # noqa: E402
    diable_occ_log,
    get_primitives,
    get_triangulations,
    normalize_shape,
)

BOUNDING = 0.9  # must match calculate_voronoi's normalization


def compute_length(edge, sample_resolution=100):
    curve = BRepAdaptor_Curve(edge)
    range_start = curve.FirstParameter() if edge.Orientation() == 0 else curve.LastParameter()
    range_end = curve.LastParameter() if edge.Orientation() == 0 else curve.FirstParameter()
    pts = np.stack([[curve.Value(u).X(), curve.Value(u).Y(), curve.Value(u).Z()]
                    for u in np.linspace(range_start, range_end, num=sample_resolution)],
                   dtype=np.float32)
    return np.linalg.norm(pts[1:] - pts[:-1], axis=1).sum()


def sample_edge_points(edges, num_points):
    """Sample ``num_points`` xyz+tangent points spread over all edges by length."""
    if not edges:
        raise ValueError("shape has no B-rep edges to sample")
    total_length = sum(compute_length(e) for e in edges)
    num_points_per_m = num_points / total_length
    out = []
    for edge in edges:
        resolution = int(compute_length(edge) * num_points_per_m)
        curve = BRepAdaptor_Curve(edge)
        range_start = curve.FirstParameter() if edge.Orientation() == 0 else curve.LastParameter()
        range_end = curve.LastParameter() if edge.Orientation() == 0 else curve.FirstParameter()
        pts = []
        for u in np.linspace(range_start, range_end, num=resolution):
            pnt = curve.Value(u)
            tan = gp_Vec()
            curve.D1(u, pnt, tan)
            tan = tan.Normalized()
            pts.append([pnt.X(), pnt.Y(), pnt.Z(), tan.X(), tan.Y(), tan.Z()])
        out.append(np.array(pts, dtype=np.float32))
    return np.concatenate(out, axis=0)


def build_implicit_npz(step_file, voronoi_ply, out_npz, seed=0):
    """Build one implicit ``.npz`` from a STEP file and its Voronoi mesh."""
    step_file, voronoi_ply = Path(step_file), Path(voronoi_ply)
    if not step_file.exists():
        raise FileNotFoundError(f"STEP file not found: {step_file}")
    if not voronoi_ply.exists():
        raise FileNotFoundError(
            f"Voronoi mesh not found: {voronoi_ply} (run Voronoi/calculate_voronoi on the STEP first)")

    random.seed(seed)
    np.random.seed(seed)
    diable_occ_log()
    import open3d as o3d

    shape = read_step_file(str(step_file), verbosity=False)
    if shape.NbChildren() != 1:
        print(f"[warn] {step_file} has {shape.NbChildren()} components (expected 1)")
    shape = normalize_shape(shape, BOUNDING)[0]  # returns (shape, scale, bbox...)
    v, f = get_triangulations(shape, 0.001)
    v, f = np.asarray(v), np.asarray(f)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(f)
    edges = get_primitives(shape, TopAbs_EDGE, True)

    # --- Surface samples (xyz + normal) ---
    sp = mesh.sample_points_uniformly(100000, use_triangle_normal=True)
    surface_points = np.concatenate([np.asarray(sp.points), np.asarray(sp.normals)], axis=-1)

    # --- Edge samples (xyz + tangent) ---
    edge_points_whole = sample_edge_points(edges, 200000)
    if edge_points_whole.shape[0] < 200000:
        edge_points_whole = edge_points_whole[np.arange(200000) % edge_points_whole.shape[0]]
    edge_points = edge_points_whole[np.random.choice(edge_points_whole.shape[0], 100000, replace=False)]

    # --- Query points near edges (perturbed at several scales) ---
    q_edge_points = np.concatenate([
        edge_points_whole[..., :3] + np.random.normal(scale=s, size=(len(edge_points_whole), 3))
        for s in (0.001, 0.005, 0.007, 0.01)], axis=0)

    # --- Query points for the surface field: random + near-surface + near-Voronoi ---
    q_rand = np.random.uniform(-1.0, 1.0, (200000, 3))
    q_surf = np.asarray(mesh.sample_points_uniformly(200000).points)
    q_surf = np.concatenate([q_surf + np.random.normal(scale=s, size=q_surf.shape)
                             for s in (0.001, 0.005)], axis=0)

    voronoi_mesh = o3d.io.read_triangle_mesh(str(voronoi_ply))
    vp = voronoi_mesh.sample_points_uniformly(200000, use_triangle_normal=True)
    voronoi_points = np.concatenate([np.asarray(vp.points), np.asarray(vp.normals)], axis=-1)
    voronoi_udf, _, _, _ = igl.signed_distance(
        voronoi_points[:, :3], v, f, sign_type=igl.SIGNED_DISTANCE_TYPE_UNSIGNED)
    voronoi_points = voronoi_points[np.abs(voronoi_udf) < 0.2]

    q_voro = np.concatenate([voronoi_points[:, :3] + np.random.normal(scale=s, size=(len(voronoi_points), 3))
                             for s in (0.001, 0.005)], axis=0)
    q_surface_points = np.concatenate([q_rand, q_surf, q_voro], axis=0)

    voro_v = np.asarray(voronoi_mesh.vertices)
    voro_f = np.asarray(voronoi_mesh.triangles)
    # SDF: signed distance to the solid surface. UDF: unsigned distance to the Voronoi field.
    surface_sdf, _, _, _ = igl.signed_distance(q_surface_points, v, f, sign_type=igl.SIGNED_DISTANCE_TYPE_DEFAULT)
    edge_sdf, _, _, _ = igl.signed_distance(q_edge_points, v, f, sign_type=igl.SIGNED_DISTANCE_TYPE_DEFAULT)
    surface_udf, _, _, _ = igl.signed_distance(q_surface_points, voro_v, voro_f, sign_type=igl.SIGNED_DISTANCE_TYPE_UNSIGNED)
    edge_udf, _, _, _ = igl.signed_distance(q_edge_points, voro_v, voro_f, sign_type=igl.SIGNED_DISTANCE_TYPE_UNSIGNED)

    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_npz),
        surface_points=surface_points.astype(np.float16),
        edge_points=edge_points.astype(np.float16),
        voronoi_points=voronoi_points.astype(np.float16),
        query_surface_points=q_surface_points.astype(np.float16),
        query_edge_points=q_edge_points.astype(np.float16),
        query_surface_sdf=surface_sdf.astype(np.float16),
        query_edge_sdf=edge_sdf.astype(np.float16),
        query_surface_udf=surface_udf.astype(np.float16),
        query_edge_udf=edge_udf.astype(np.float16),
    )
    print(f"wrote {out_npz}  ({len(voronoi_points)} voronoi pts, "
          f"{len(q_surface_points)} surface / {len(q_edge_points)} edge queries)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", help="single STEP file")
    parser.add_argument("--voronoi", help="single Voronoi mesh (voronoi.ply from calculate_voronoi)")
    parser.add_argument("--out", help="output .npz path (single-shape mode)")
    parser.add_argument("--step-dir", help="directory of <name>.step (batch mode)")
    parser.add_argument("--voronoi-dir", help="directory of <name>.ply Voronoi meshes (batch mode)")
    parser.add_argument("--out-dir", help="output directory for <name>.npz (batch mode)")
    parser.add_argument("--list", dest="name_list", help="optional text file of names to process in batch mode")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.step and args.voronoi and args.out:
        build_implicit_npz(args.step, args.voronoi, args.out, args.seed)
        return

    if args.step_dir and args.voronoi_dir and args.out_dir:
        step_dir, voro_dir, out_dir = Path(args.step_dir), Path(args.voronoi_dir), Path(args.out_dir)
        if args.name_list:
            names = [l.split()[0] for l in Path(args.name_list).read_text().splitlines() if l.strip()]
        else:
            names = sorted(p.stem for p in step_dir.glob("*.step"))
        print(f"{len(names)} shapes to process")
        for name in names:
            voro = voro_dir / f"{name}.ply"
            if not voro.exists():
                print(f"[skip] {name}: no Voronoi mesh at {voro}")
                continue
            try:
                build_implicit_npz(step_dir / f"{name}.step", voro, out_dir / f"{name}.npz", args.seed)
            except Exception as e:  # noqa: BLE001
                print(f"[error] {name}: {e}")
        return

    parser.error("provide --step/--voronoi/--out (single) or --step-dir/--voronoi-dir/--out-dir (batch)")


if __name__ == "__main__":
    main()
