#!/usr/bin/env python3
"""
Crash-safe parallel VTK -> HDF5 converter.

Phase 1: Extract VTK -> .npz in parallel subprocesses (isolates VTK segfaults).
Phase 2: Merge all .npz into a single HDF5.

Usage:
  python convert_vtk_safe.py --vtk-dir data/train/vtk --out-h5 data/surface_data.hdf5
  python convert_vtk_safe.py --vtk-dir data/test/vtk  --out-h5 data/surface_data_test.hdf5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

H5_COMP = {"compression": "gzip", "compression_opts": 4, "shuffle": True}

# ---------------------------------------------------------------------------
# Worker script — executed in isolated subprocesses to avoid VTK segfaults
# taking down the main process. Each invocation handles a batch of files.
# ---------------------------------------------------------------------------
WORKER_SOURCE = r'''
import os, sys, json
import numpy as np

os.environ["PYVISTA_OFF_SCREEN"] = "true"
import pyvista as pv

vtk_paths = json.loads(sys.argv[1])
out_dir = sys.argv[2]

failed = []

for vtk_path in vtk_paths:
    cid = vtk_path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
    out_npz = os.path.join(out_dir, cid + ".npz")
    if os.path.exists(out_npz):
        continue
    try:
        mesh = pv.read(vtk_path)
        surf = mesh if isinstance(mesh, pv.PolyData) else mesh.extract_surface()
        points = np.asarray(surf.points, dtype=np.float32)
        npts = points.shape[0]
        if npts == 0:
            raise RuntimeError("0 points")

        # Normals
        normals = None
        for nm in ["Normals", "Normal", "normals", "normal"]:
            if nm in surf.point_data:
                arr = np.asarray(surf.point_data[nm], dtype=np.float32)
                if arr.shape == (npts, 3):
                    normals = arr
                    break
        if normals is None:
            nmesh = surf.compute_normals(point_normals=True, cell_normals=False, inplace=False)
            for nm in ["Normals", "Normal", "normals", "normal"]:
                if nm in nmesh.point_data:
                    normals = np.asarray(nmesh.point_data[nm], dtype=np.float32)
                    break
            del nmesh
        if normals is None:
            raise RuntimeError("Failed to compute normals")

        def get_scalar(name):
            for src in [surf, mesh]:
                if name in src.point_data:
                    arr = np.asarray(src.point_data[name], dtype=np.float32)
                    if arr.shape == (npts,):
                        return arr
            return np.full((npts,), np.nan, dtype=np.float32)

        cp   = get_scalar("cp")
        cf_x = get_scalar("cf_x")
        cf_y = get_scalar("cf_y")
        cf_z = get_scalar("cf_z")

        np.savez_compressed(out_npz, points=points, normals=normals,
                            cp=cp, cf_x=cf_x, cf_y=cf_y, cf_z=cf_z)

    except Exception as e:
        failed.append(f"{cid}: {e}")

if failed:
    print("FAILURES:" + json.dumps(failed), file=sys.stderr)
'''


def run_batch(python: str, worker_script: str, vtk_paths: list[str],
              npz_dir: str) -> list[str]:
    """Run a batch in a subprocess. Returns list of failure messages."""
    result = subprocess.run(
        [python, worker_script, json.dumps(vtk_paths), npz_dir],
        capture_output=True, timeout=600,
    )
    stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
    failures = []
    for line in stderr.splitlines():
        if line.startswith("FAILURES:"):
            failures.extend(json.loads(line[len("FAILURES:"):]))
    if result.returncode < 0:
        for vp in vtk_paths:
            cid = Path(vp).stem.lower()
            if not (Path(npz_dir) / f"{cid}.npz").exists():
                failures.append(f"{cid}: killed by signal {-result.returncode}")
    return failures


def _run_one_batch(args_tuple):
    """Wrapper for ProcessPoolExecutor — unpacks tuple and calls run_batch."""
    batch_id, python, worker_script, vtk_paths, npz_dir = args_tuple
    try:
        failures = run_batch(python, worker_script, vtk_paths, npz_dir)
        return batch_id, failures
    except subprocess.TimeoutExpired:
        failures = []
        for vp in vtk_paths:
            cid = Path(vp).stem.lower()
            if not (Path(npz_dir) / f"{cid}.npz").exists():
                failures.append(f"{cid}: timeout")
        return batch_id, failures


def merge_npz_to_hdf5(npz_dir: Path, out_h5: Path, source_dir: str) -> int:
    """Merge all .npz files into a single HDF5. Returns count."""
    npz_files = sorted(npz_dir.glob("*.npz"))
    if not npz_files:
        raise RuntimeError(f"No .npz files found in {npz_dir}")

    with h5py.File(out_h5, "w") as h5:
        for gname in ["points", "normals", "cp", "cf_x", "cf_y", "cf_z"]:
            h5.create_group(gname)

        h5.attrs["source_dir"] = source_dir
        h5.attrs["builder"] = "convert_vtk_safe.py"

        for i, npz_path in enumerate(npz_files):
            cid = npz_path.stem
            data = np.load(npz_path)
            h5["points"].create_dataset(cid, data=data["points"], chunks=True, **H5_COMP)
            h5["normals"].create_dataset(cid, data=data["normals"], chunks=True, **H5_COMP)
            h5["cp"].create_dataset(cid, data=data["cp"], chunks=True, **H5_COMP)
            h5["cf_x"].create_dataset(cid, data=data["cf_x"], chunks=True, **H5_COMP)
            h5["cf_y"].create_dataset(cid, data=data["cf_y"], chunks=True, **H5_COMP)
            h5["cf_z"].create_dataset(cid, data=data["cf_z"], chunks=True, **H5_COMP)
            data.close()

            if (i + 1) % 500 == 0:
                print(f"  Merged {i + 1}/{len(npz_files)} ...")

        h5.attrs["written_cases"] = len(npz_files)
    return len(npz_files)


def main():
    parser = argparse.ArgumentParser(description="Crash-safe parallel VTK -> HDF5.")
    parser.add_argument("--vtk-dir", type=Path, required=True)
    parser.add_argument("--out-h5", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100,
                        help="VTK files per subprocess (default: 100)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel subprocesses (default: 8)")
    parser.add_argument("--keep-npz", action="store_true",
                        help="Keep .npz cache after merge")
    args = parser.parse_args()

    vtk_files = sorted(args.vtk_dir.glob("case_*.vtk"))
    if not vtk_files:
        raise FileNotFoundError(f"No case_*.vtk in {args.vtk_dir}")

    python = sys.executable
    npz_dir = args.out_h5.parent / f".npz_cache_{args.out_h5.stem}"
    npz_dir.mkdir(parents=True, exist_ok=True)

    worker_file = npz_dir / "_worker.py"
    worker_file.write_text(WORKER_SOURCE)

    # Skip already-extracted files
    already_done = {p.stem for p in npz_dir.glob("*.npz")}
    remaining = [f for f in vtk_files if f.stem.lower() not in already_done]
    print(f"Total: {len(vtk_files)} | Already extracted: {len(already_done)} | "
          f"Remaining: {len(remaining)}")

    if remaining:
        # Build batches
        vtk_strs = [str(f) for f in remaining]
        batches = []
        for i in range(0, len(vtk_strs), args.batch_size):
            batches.append(vtk_strs[i:i + args.batch_size])

        print(f"Extracting in {len(batches)} batches x {args.batch_size} files, "
              f"{args.workers} workers ...\n")

        all_failures = []
        completed = 0

        # Run batches in parallel via thread pool (each spawns a subprocess)
        from concurrent.futures import ThreadPoolExecutor
        tasks = [
            (bid, python, str(worker_file), batch, str(npz_dir))
            for bid, batch in enumerate(batches)
        ]

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_one_batch, t): t for t in tasks}
            for future in as_completed(futures):
                batch_id, failures = future.result()
                completed += 1
                n_batch = len(batches[batch_id]) if batch_id < len(batches) else 0
                status = f"batch {batch_id+1}/{len(batches)} ({n_batch} files)"
                if failures:
                    all_failures.extend(failures)
                    print(f"  {status} — {len(failures)} failed")
                else:
                    print(f"  {status} — OK")

        npz_count = len(list(npz_dir.glob("*.npz")))
        print(f"\nExtraction done: {npz_count}/{len(vtk_files)} cases")
        if all_failures:
            print(f"Failures ({len(all_failures)}):")
            for msg in all_failures[:20]:
                print(f"  {msg}")
    else:
        npz_count = len(already_done)
        print("All files already extracted.")

    # Phase 2: merge
    print(f"\nMerging {npz_count} .npz files into {args.out_h5} ...")
    n = merge_npz_to_hdf5(npz_dir, args.out_h5, str(args.vtk_dir))
    print(f"Done: {n} cases in {args.out_h5}")

    if not args.keep_npz:
        import shutil
        shutil.rmtree(npz_dir)
        print(f"Cleaned up {npz_dir}")


if __name__ == "__main__":
    main()
