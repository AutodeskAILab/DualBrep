"""DualBrep — B-rep post-processing (stage 3b): face/edge grids -> watertight B-rep STEP.

Consumes the rotation candidates written by ``rebuild.py`` under ``<out>/tmp/<name>_<k>/``
and reconstructs a parametric B-rep with OpenCASCADE (``brep_post/``): each face is fit as a
trimmed B-spline surface, edges are sewn into wires/faces, and the faces are sewn into a
solid. Each candidate's rotation is undone with ``inv(M[k])``, so all candidates land in the
input coordinate frame.

Candidates are assembled in parallel with Ray by default (``--serial`` for a single-process
loop). A ``StartRegistry`` actor records, per candidate, the wall-clock time its task
actually began *executing* on a worker (not when it was submitted); the driver polls it and
force-kills (SIGKILL) any candidate that has been *running* longer than ``--timeout``
(default 1200s) -- queue / resource-wait time is never counted. Ray tasks are not retried
(``max_retries=0``), and up to ``--max_running`` (default 10000) candidates are kept in flight
(Ray still runs at most ``--num_cpus`` at a time, so this is a submission cap, not
oversubscription).

Each candidate is built twice at most, from the same seed (``--seed``, so a rebuild is reproducible):
first without the geometry optimization that precedes surface fitting and without hole filling
(the latent-optimised faces of ``rebuild.py`` already meet; the optimization, which fits edges to
face samples 0.05 apart, would undo that); if that gives no solid, with the optimization and, if
no tolerance seals it, with the holes left by faces that could not be trimmed closed by n-sided
patches (``BRepOffsetAPI_MakeFilling``). For each shape, one sealed rotation is promoted to
``<name>.step`` (+ ``<name>.ply`` triangulation): rotations sealed without patches first; among
the sealed rotations that are the same B-rep as the first one (same vertices), the one with the
smallest real gap between faces. Working files stay under ``tmp/``.

Usage:
    python postprocess.py --input output_brep
    python postprocess.py --input output_brep --num_cpus 32          # cap physical concurrency
    python postprocess.py --input output_brep --timeout 600
    python postprocess.py --input output_brep --serial --max_optimize_iter 50
"""
import argparse
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import ray
from tqdm import tqdm


def _suffix_from_folder(folder_name):
    """Octahedral rotation id encoded as the 2nd token of '<stem>_<k>' (-1 -> identity)."""
    return int(folder_name.split("_")[1]) if "_" in folder_name else -1


@ray.remote(max_restarts=-1, max_task_retries=-1)
class StartRegistry:
    """Records, per candidate, the wall-clock time its task actually began *executing* on a
    worker (not when it was submitted). The driver reads this to measure true running time,
    so candidates still queued / waiting for a CPU are never charged against the timeout.
    """
    def __init__(self):
        self._starts = {}

    def mark(self, key, ts):
        self._starts[key] = ts

    def prune(self, keys):
        # Batched removal of finished candidates (keeps the dict bounded).
        for k in keys:
            self._starts.pop(k, None)

    def get_stale(self, cutoff):
        # Only the few candidates that began running before ``cutoff`` (running longer than
        # the budget) -- a tiny payload regardless of how many tasks are in flight.
        return [(k, ts) for k, ts in self._starts.items() if ts < cutoff]


def build_one(npz_path, here, drop_num, use_cuda, is_optimize_geom, max_optimize_iter, registry=None,
              seed=0, record_gap=True, fill_holes=True, no_optimize_first=True):
    """Assemble a single candidate -> (folder_name, is_valid_solid, error_or_None, run_seconds).

    Runs the OpenCASCADE assembly in-process. ``run_seconds`` is measured from when this task
    actually starts executing on a worker; that start is stamped into ``registry`` so the
    driver can force-kill a candidate that has been *running* past the timeout (queue /
    resource-wait time is never counted). Leaves ``pp/recon_brep.step`` + ``pp/recon_brep.stl``
    on success. Self-contained (imports inside) so it runs both in-process and as a Ray task.
    ``no_optimize_first``: build without the geometry optimization and without hole filling first
    (marks ``pp/no_optimize.txt``); only if that gives no solid, build again with
    ``is_optimize_geom`` / ``fill_holes``. Each attempt starts from the same seed.
    """
    import time
    from pathlib import Path

    t_start = time.time()
    name = Path(npz_path).parent.stem
    if registry is not None:
        registry.mark.remote(name, t_start)   # true execution start, after any queue wait

    if here not in sys.path:
        sys.path.insert(0, here)
    from brep_post.occ_utils import diable_occ_log
    from brep_post.construct_brep import construct_brep_from_datanpz

    diable_occ_log()
    npz_path = Path(npz_path)
    pp = npz_path.parent / "pp"
    # (is_optimize_geom, fill_holes) per attempt; from_scratch=True clears pp/ before each one
    attempts = ([(False, False)] if no_optimize_first else []) + [(is_optimize_geom, fill_holes)]
    err = None
    for i, (opt, fill) in enumerate(attempts):
        if seed >= 0:    # wire building and sewing shuffle their inputs (random.shuffle)
            import random
            random.seed(seed)
        err = None
        try:
            construct_brep_from_datanpz(
                npz_path, pp,
                v_drop_num=drop_num, use_cuda=use_cuda, from_scratch=True,
                is_save_data=False, is_log=False,
                is_optimize_geom=opt, v_max_optimize_iter=max_optimize_iter,
                is_ray=False, suffix=_suffix_from_folder(name), v_fill_holes=fill,
            )
        except Exception as e:
            err = str(e)[:160]
        if (pp / "success.txt").exists():
            if no_optimize_first and i == 0:
                open(pp / "no_optimize.txt", "w").close()
            break
    if record_gap and (pp / "success.txt").exists():
        try:
            _save_gap_and_vertices(pp)
        except Exception as e:
            err = str(e)[:160]
    return name, (pp / "success.txt").exists(), err, time.time() - t_start


def _save_gap_and_vertices(pp):
    """Record a sealed candidate's real gap and its vertex positions in pp/gap.npz.

    The solid is saved with every tolerance forced to the sewing tolerance that worked (up to 0.08),
    which hides how far apart the faces really are. The real gap is the largest tolerance OCC needs
    once all tolerances are reset and recomputed (SameParameter + UpdateTolerances).
    """
    import numpy as np
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepLib import breplib
    from OCC.Core.ShapeAnalysis import ShapeAnalysis_ShapeTolerance
    from OCC.Core.ShapeFix import ShapeFix_ShapeTolerance
    from OCC.Core.TopAbs import TopAbs_VERTEX
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods
    from OCC.Extend.DataExchange import read_step_file

    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopLoc import TopLoc_Location

    shape = read_step_file(str(pp / "recon_brep.step"), verbosity=False)
    exp, pts = TopExp_Explorer(shape, TopAbs_VERTEX), []
    while exp.More():
        p = BRep_Tool.Pnt(topods.Vertex(exp.Current()))
        pts.append((p.X(), p.Y(), p.Z()))
        exp.Next()
    exp, bad_faces = TopExp_Explorer(shape, TopAbs_FACE), 0     # faces the mesher cannot triangulate
    while exp.More():
        face = topods.Face(exp.Current())
        BRepMesh_IncrementalMesh(face, 0.1, True, 0.1)
        bad_faces += BRep_Tool.Triangulation(face, TopLoc_Location()) is None
        exp.Next()
    ShapeFix_ShapeTolerance().LimitTolerance(shape, 1e-7, 1e-7)
    breplib.SameParameter(shape, 1e-7, True)
    breplib.UpdateTolerances(shape, False)
    np.savez(pp / "gap.npz", gap=ShapeAnalysis_ShapeTolerance().Tolerance(shape, 1),
             vertices=np.unique(np.round(np.array(pts), 6), axis=0), bad_faces=bad_faces)


def _pick_rotation(sealed, cand_dir, vert_tol=2e-2):
    """Among the sealed rotations, the smallest-gap one that is the same B-rep as the first sealed one
    (same vertices within vert_tol, both ways) and has no more faces the mesher cannot triangulate.
    Other rotations place the seams of periodic faces elsewhere, which costs vertex/edge F1 although
    the solid is equally valid."""
    import numpy as np

    def load(n):
        f = cand_dir[n] / "pp" / "gap.npz"
        return np.load(f) if f.exists() else None

    ref = load(sealed[0])
    if ref is None:
        return sealed[0]
    best, best_gap = sealed[0], float(ref["gap"])
    for n in sealed[1:]:
        d = load(n)
        if (d is None or len(d["vertices"]) != len(ref["vertices"]) or float(d["gap"]) >= best_gap
                or int(d["bad_faces"]) > int(ref["bad_faces"])):
            continue
        dist = np.linalg.norm(d["vertices"][:, None] - ref["vertices"][None], axis=2)
        if dist.min(1).max() < vert_tol and dist.min(0).max() < vert_tol:
            best, best_gap = n, float(d["gap"])
    return best


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
    ap.add_argument("--use_cuda", action="store_true", help="Run geometry optimization on GPU.")
    ap.add_argument("--max_running", type=int, default=10000, help="Max candidates kept in flight at once (submission cap).")
    ap.add_argument("--num_cpus", type=int, default=None, help="Ray logical CPUs = candidates physically running at once (default: all cores).")
    ap.add_argument("--timeout", type=int, default=1200, help="Per-candidate timeout in seconds of ACTUAL running time (queue wait excluded).")
    ap.add_argument("--serial", action="store_true", help="Disable Ray; build candidates one at a time (no timeout).")
    ap.add_argument("--seed", type=int, default=0, help="Per-candidate random seed (-1: unseeded).")
    args = ap.parse_args()

    root = Path(args.input)
    files = sorted((root / "tmp").glob("**/post.npz")) or sorted(root.glob("**/post.npz"))
    if not files:
        raise FileNotFoundError(f"No post.npz found under {root}/tmp")
    cand_dir = {f.parent.stem: f.parent for f in files}      # <name>_<k> -> candidate dir
    print(f"Found {len(files)} candidate(s) under {root}/tmp")

    valid = {}      # name -> bool
    errors = []
    runtimes = {}   # name -> actual run seconds
    n_timeout = 0

    if args.serial:
        bar = tqdm(files)
        for f in bar:
            name, ok, err, rt = build_one(f, _HERE, args.drop_num, args.use_cuda, True, args.max_optimize_iter,
                                          seed=args.seed)
            valid[name] = ok
            runtimes[name] = rt
            if err:
                errors.append(f"{name}: {err}")
            bar.set_description(f"sealed: {sum(valid.values())}/{len(valid)}")
    else:
        TASK_TIMEOUT_S = args.timeout
        MAX_RUNNING = args.max_running
        REAP_BATCH = 512           # max results harvested per ray.wait
        POLL_TIMEOUT_S = 5         # ray.wait wake-up cadence
        SWEEP_INTERVAL_S = 30      # how often to run the timeout sweep

        ray.init(ignore_reinit_error=True, num_cpus=args.num_cpus,
                 include_dashboard=False, log_to_driver=False)
        registry = StartRegistry.remote()
        # max_retries=0: a candidate that dies (OOM/segfault) or is cancelled is NOT retried.
        remote_build = ray.remote(num_cpus=1, max_retries=0)(build_one)
        n_cpu = int(ray.available_resources().get("CPU", 1))
        print(f"Ray build: <= {MAX_RUNNING} in flight, ~{n_cpu} running at once, "
              f"{TASK_TIMEOUT_S}s running-time timeout, no retries.")

        pending = list(files)
        running = {}      # ref -> post.npz Path
        done_keys = []    # finished/cancelled names, pruned from the registry in batches

        def submit(f):
            ref = remote_build.remote(str(f), _HERE, args.drop_num, args.use_cuda,
                                      True, args.max_optimize_iter, registry, seed=args.seed)
            running[ref] = f

        for _ in range(min(MAX_RUNNING, len(pending))):
            submit(pending.pop(0))

        bar = tqdm(total=len(files))
        last_sweep = time.time()
        while running:
            ready, _ = ray.wait(list(running.keys()),
                                num_returns=min(REAP_BATCH, len(running)),
                                timeout=POLL_TIMEOUT_S)
            for ref in ready:
                f = running.pop(ref)
                name = f.parent.stem
                try:
                    nm, ok, err, rt = ray.get(ref)
                    valid[nm] = ok
                    runtimes[nm] = rt
                    if err:
                        errors.append(f"{nm}: {err}")
                except Exception as e:
                    valid[name] = False
                    errors.append(f"{name}: {str(e)[:120]}")
                done_keys.append(name)
                bar.update(1)
                bar.set_description(f"sealed: {sum(1 for v in valid.values() if v)} | timeouts: {n_timeout}")
                if pending and len(running) < MAX_RUNNING:
                    submit(pending.pop(0))

            # --- Timeout sweep (throttled): kill candidates whose *running* time exceeds the
            # budget. Start times come from the registry (set when a task began executing), so
            # queued / resource-waiting candidates are never charged the wait time.
            now = time.time()
            if running and now - last_sweep > SWEEP_INTERVAL_S:
                try:
                    stale = ray.get(registry.get_stale.remote(now - TASK_TIMEOUT_S))
                except Exception as e:
                    # Registry actor died/restarted: re-seed every in-flight start with `now`
                    # (conservatively resets each clock) rather than crashing the driver.
                    for ref, f in running.items():
                        registry.mark.remote(f.parent.stem, now)
                    errors.append(f"<registry get_stale failed, re-seeded>: {str(e)[:80]}")
                    stale = []
                if stale:
                    stale_set = {k for k, _ in stale}
                    for ref, f in list(running.items()):
                        if f.parent.stem in stale_set:
                            ray.cancel(ref, force=True)      # SIGKILL the stuck worker
                            running.pop(ref)
                            valid[f.parent.stem] = False
                            errors.append(f"{f.parent.stem}: timeout>{TASK_TIMEOUT_S}s running")
                            done_keys.append(f.parent.stem)
                            n_timeout += 1
                            bar.update(1)
                    while pending and len(running) < MAX_RUNNING:
                        submit(pending.pop(0))
                if done_keys:
                    try:
                        registry.prune.remote(done_keys)
                        done_keys = []
                    except Exception:
                        pass    # keep done_keys for the next sweep
                last_sweep = now
        bar.close()
        ray.shutdown()

    # Group candidates by shape; promote one sealed rotation per shape (--promote).
    by_shape = defaultdict(list)
    for name in cand_dir:
        by_shape[name.rsplit("_", 1)[0]].append(name)

    n_ok = 0
    print()
    for stem in sorted(by_shape):
        sealed = sorted(n for n in by_shape[stem] if valid.get(n))
        unfilled = [n for n in sealed if not (cand_dir[n] / "pp" / "filled.txt").exists()]
        if sealed:
            pool = unfilled or sealed                     # patched solids only if nothing sealed without patches
            pick = _pick_rotation(pool, cand_dir)
            _promote(cand_dir[pick], root / f"{stem}.step", root / f"{stem}.ply")
            n_ok += 1
            note = (" (holes patched)" if not unfilled else
                    " (no geometry optimization)" if (cand_dir[pick] / "pp" / "no_optimize.txt").exists() else "")
            print(f"  {stem}: solid from {pick}{note} "
                  f"({len(sealed)}/{len(by_shape[stem])} rotations sealed) -> {stem}.step + {stem}.ply")
        else:
            print(f"  {stem}: no watertight solid in {len(by_shape[stem])} rotations")

    print(f"\nDone. {n_ok}/{len(by_shape)} shapes -> a valid B-rep in {root} (intermediate candidates under {root}/tmp).")
    if runtimes:
        rts = sorted(runtimes.values())
        mean = sum(rts) / len(rts)
        p95 = rts[min(len(rts) - 1, int(0.95 * len(rts)))]
        print(f"Per-candidate run time (actual, excl. queue): mean {mean:.1f}s, p95 {p95:.1f}s, "
              f"max {rts[-1]:.1f}s; {n_timeout} killed at the {args.timeout}s timeout.")
    if errors:
        print(f"{len(errors)} candidate(s) errored/timed out, e.g.:")
        for e in errors[:10]:
            print("  ", e)


if __name__ == "__main__":
    main()
