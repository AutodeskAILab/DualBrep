"""DualBrep — B-rep post-processing (stage 3b): face/edge grids -> watertight B-rep STEP.

Consumes the rotation candidates written by ``rebuild.py`` under ``<out>/tmp/<name>_<k>/``
and reconstructs a parametric B-rep with OpenCASCADE (``brep_post/``): each face is fit as a
trimmed B-spline surface, edges are sewn into wires/faces, and the faces are sewn into a
solid. Each candidate's rotation is undone with ``inv(M[k])``, so all candidates land in the
input coordinate frame.

Candidates are assembled in parallel with Ray by default (one CPU per candidate; ``--serial``
for a single-process loop). For each shape, the first rotation that seals into a valid
watertight solid is promoted to the output folder as ``<name>.step`` (+ ``<name>.ply``
triangulation); the per-candidate working files stay under ``tmp/``.

Usage:
    python postprocess.py --input output_brep
    python postprocess.py --input output_brep --num_cpus 16
    python postprocess.py --input output_brep --serial --max_optimize_iter 50
"""
import argparse
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from tqdm import tqdm


def _suffix_from_folder(folder_name):
    """Octahedral rotation id encoded as the 2nd token of '<stem>_<k>' (-1 -> identity)."""
    return int(folder_name.split("_")[1]) if "_" in folder_name else -1


def build_one(npz_path, here, drop_num, use_cuda, is_optimize_geom, max_optimize_iter):
    """Assemble a single candidate -> (folder_name, is_valid_solid, error_or_None).

    Leaves ``pp/recon_brep.step`` + ``pp/recon_brep.stl`` in the candidate folder on success.
    Self-contained (imports inside) so it runs both in-process and as a Ray task.
    """
    from pathlib import Path
    if here not in sys.path:
        sys.path.insert(0, here)
    from brep_post.occ_utils import diable_occ_log
    from brep_post.construct_brep import construct_brep_from_datanpz

    diable_occ_log()
    npz_path = Path(npz_path)
    pp = npz_path.parent / "pp"
    name = npz_path.parent.stem
    err = None
    try:
        construct_brep_from_datanpz(
            npz_path, pp,
            v_drop_num=drop_num, use_cuda=use_cuda, from_scratch=True,
            is_save_data=False, is_log=False,
            is_optimize_geom=is_optimize_geom, v_max_optimize_iter=max_optimize_iter,
            is_ray=False, suffix=_suffix_from_folder(name),
        )
    except Exception as e:
        err = str(e)[:160]
    return name, (pp / "success.txt").exists(), err


def _promote(cand_dir, out_step, out_ply):
    """Copy a valid candidate's solid to <stem>.step and write its <stem>.ply triangulation."""
    shutil.copy(cand_dir / "pp" / "recon_brep.step", out_step)
    stl = cand_dir / "pp" / "recon_brep.stl"
    if stl.exists():
        import trimesh
        trimesh.load(str(stl), process=True).export(str(out_ply))   # merge coincident verts -> clean mesh


def main():
    ap = argparse.ArgumentParser(description="Assemble rotation candidates into one B-rep STEP per shape (OpenCASCADE).")
    ap.add_argument("--input", required=True, help="rebuild.py output dir (candidates under <input>/tmp/<name>_<k>/).")
    ap.add_argument("--drop_num", type=int, default=0, help="Faces to drop when searching for a valid sub-solid.")
    ap.add_argument("--max_optimize_iter", type=int, default=200, help="Geometry-optimization iterations.")
    ap.add_argument("--no_optimize", action="store_true", help="Skip the boundary geometry optimization.")
    ap.add_argument("--use_cuda", action="store_true", help="Run geometry optimization on GPU.")
    ap.add_argument("--num_cpus", type=int, default=None, help="Max parallel Ray workers (default: all cores).")
    ap.add_argument("--serial", action="store_true", help="Disable Ray; build candidates one at a time.")
    args = ap.parse_args()

    root = Path(args.input)
    files = sorted((root / "tmp").glob("**/post.npz")) or sorted(root.glob("**/post.npz"))
    if not files:
        raise FileNotFoundError(f"No post.npz found under {root}/tmp")
    is_opt = not args.no_optimize
    cand_dir = {f.parent.stem: f.parent for f in files}      # <name>_<k> -> candidate dir
    print(f"Found {len(files)} candidate(s) under {root}/tmp")

    valid = {}    # name -> bool
    errors = []

    if args.serial:
        bar = tqdm(files)
        for f in bar:
            name, ok, err = build_one(f, _HERE, args.drop_num, args.use_cuda, is_opt, args.max_optimize_iter)
            valid[name] = ok
            if err:
                errors.append(f"{name}: {err}")
            bar.set_description(f"sealed: {sum(valid.values())}/{len(valid)}")
    else:
        import ray
        ray.init(ignore_reinit_error=True, num_cpus=args.num_cpus,
                 include_dashboard=False, log_to_driver=False)
        print(f"Ray parallel build on {int(ray.available_resources().get('CPU', 1))} CPUs.")
        remote_build = ray.remote(num_cpus=1)(build_one)
        pending = [remote_build.remote(str(f), _HERE, args.drop_num, args.use_cuda, is_opt, args.max_optimize_iter)
                   for f in files]
        bar = tqdm(total=len(pending))
        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            try:
                name, ok, err = ray.get(ready[0])
                valid[name] = ok
                if err:
                    errors.append(f"{name}: {err}")
            except Exception as e:
                errors.append(f"<task>: {e}")
            bar.update(1)
            bar.set_description(f"sealed: {sum(valid.values())}")
        ray.shutdown()

    # Group candidates by shape; promote the first rotation that sealed into a valid solid.
    by_shape = defaultdict(list)
    for name in cand_dir:
        by_shape[name.rsplit("_", 1)[0]].append(name)

    n_ok = 0
    print()
    for stem in sorted(by_shape):
        sealed = sorted(n for n in by_shape[stem] if valid.get(n))
        if sealed:
            _promote(cand_dir[sealed[0]], root / f"{stem}.step", root / f"{stem}.ply")
            n_ok += 1
            print(f"  {stem}: solid from {sealed[0]} ({len(sealed)}/{len(by_shape[stem])} rotations sealed) -> {stem}.step + {stem}.ply")
        else:
            print(f"  {stem}: no watertight solid in {len(by_shape[stem])} rotations")

    print(f"\nDone. {n_ok}/{len(by_shape)} shapes -> a valid B-rep in {root} (intermediate candidates under {root}/tmp).")
    if errors:
        print(f"{len(errors)} candidate(s) errored, e.g.:")
        for e in errors[:10]:
            print("  ", e)


if __name__ == "__main__":
    main()
