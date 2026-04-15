#!/usr/bin/env python3
"""
Visualize FiLM predictions on a VTK surface mesh.

Loads a VTK file (ground truth), runs the FiLM model on all mesh points,
and shows a side-by-side PyVista window: GT | Predicted | Error for cp, cf_x, cf_z.

Usage:
    python viz_vtk.py data/test/vtk/case_000.vtk
    python viz_vtk.py data/test/vtk/case_000.vtk --checkpoint checkpoints/film_ep2478_val0.056.pth
    python viz_vtk.py data/test/vtk/case_000.vtk --field cp
    python viz_vtk.py data/train/vtk/case_0000.vtk
"""

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import pyvista as pv
import pandas as pd
import h5py

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))
from models.film_model_v1 import FiLMNet

FIELD_LABELS = {"cp": "Cp", "cf_x": "Cf_x", "cf_z": "Cf_z"}
CMAPS        = {"cp": "RdBu_r", "cf_x": "viridis", "cf_z": "viridis"}


# ── helpers ────────────────────────────────────────────────────────────────────

def load_norm_stats(repo_root: Path) -> dict:
    p = repo_root / "norm_stats.json"
    raw = json.load(open(p))
    return {k: np.array(v, dtype=np.float32) for k, v in raw.items()}


def detect_split(vtk_path: Path):
    """Return ('test', csv, hdf5) or ('train', csv, hdf5) based on the path."""
    if "test" in vtk_path.parts:
        return (
            "test",
            REPO_ROOT / "csv_files" / "case_with_geom_params_test.csv",
            REPO_ROOT / "data" / "surface_data_test.hdf5",
        )
    return (
        "train",
        REPO_ROOT / "csv_files" / "case_with_geom_params_train.csv",
        REPO_ROOT / "data" / "surface_data.hdf5",
    )


def load_model(ckpt_path: Path, device: str) -> FiLMNet:
    model = FiLMNet(cond_dim=13, coord_dim=6, output_dim=3,
                    hidden_dim=256, num_layers=4, extra_layers=3)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Handle full checkpoint dicts (with optimizer, epoch, etc.)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


def lookup_case(case_key: str, csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    # Normalise column names
    if "Re_L" not in df.columns and "Re" in df.columns:
        df = df.rename(columns={"Re": "Re_L"})
    if "mesh" not in df.columns and "geom_name" in df.columns:
        df["mesh"] = df["geom_name"]

    # Build a normalised case_key column
    df["_key"] = df["case_name"].apply(lambda v: v.lower() if isinstance(v, str) else v)
    row = df[df["_key"] == case_key]
    if row.empty:
        raise ValueError(f"Case '{case_key}' not found in {csv_path}")
    return row.iloc[0]


def build_cond_vec(row: pd.Series, ns: dict) -> np.ndarray:
    flight_cols = ["Re_L", "M_inf", "alpha_deg"]
    shape_cols  = ["B1", "B2", "B3", "C1", "C2", "C3", "C4", "S1", "S2", "S3"]
    flight = np.array([row[c] for c in flight_cols], dtype=np.float32)
    shape  = np.array([row[c] for c in shape_cols],  dtype=np.float32)
    nf = (flight - ns["flight_mean"]) / ns["flight_std"]
    ns_ = (shape  - ns["shape_mean"])  / ns["shape_std"]
    return np.concatenate([nf, ns_]).astype(np.float32)


@torch.no_grad()
def run_inference(model, coords6: np.ndarray, cond: np.ndarray,
                  ns: dict, device: str, chunk: int = 200_000) -> np.ndarray:
    """Returns (N, 3) array of denormalised [cp, cf_x, cf_z]."""
    coords_t = torch.from_numpy(coords6).to(device, dtype=torch.float32)
    cond_t   = torch.from_numpy(cond).to(device, dtype=torch.float32)
    out_mu   = torch.tensor(ns["output_mean"], device=device).view(1, 3)
    out_sd   = torch.tensor(ns["output_std"],  device=device).view(1, 3)

    N = coords_t.shape[0]
    cond_b = cond_t.unsqueeze(0).expand(N, -1)
    preds = []
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        p = model(coords_t[s:e], cond_b[s:e])  # (M, 3) normalised
        p = p * out_sd + out_mu                  # denormalise
        preds.append(p.cpu().numpy())
    return np.concatenate(preds, axis=0)          # (N, 3)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Visualize FiLM predictions on a VTK mesh")
    ap.add_argument("vtk", type=Path, help="Path to a .vtk surface file")
    ap.add_argument("--checkpoint", type=Path,
                    default=REPO_ROOT / "checkpoints" / "film_best.pth",
                    help="Path to model checkpoint (default: film_best.pth)")
    ap.add_argument("--field", choices=["cp", "cf_x", "cf_z", "all"], default="all",
                    help="Which field to show (default: all)")
    ap.add_argument("--device", default=None,
                    help="torch device (default: mps > cuda > cpu)")
    args = ap.parse_args()

    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    vtk_path  = args.vtk.resolve()
    case_key  = vtk_path.stem          # e.g. "case_000" or "case_0000"
    split, csv_path, hdf5_path = detect_split(vtk_path)

    print(f"Case    : {case_key}  ({split} split)")
    print(f"Ckpt    : {args.checkpoint}")
    print(f"Device  : {args.device}")

    # 1. Load VTK (ground truth + mesh topology)
    mesh = pv.read(str(vtk_path))
    pts  = np.array(mesh.points, dtype=np.float32)   # (N, 3)
    N    = pts.shape[0]
    print(f"Points  : {N:,}")

    # 2. Norm stats
    ns = load_norm_stats(REPO_ROOT)

    # 3. Normals — from HDF5 (same normals used at training time)
    with h5py.File(hdf5_path, "r") as f:
        if case_key in f["normals"]:
            normals = f["normals"][case_key][()].astype(np.float32)  # (N, 3)
        else:
            print("  [warn] normals not in HDF5 — computing from mesh geometry")
            mesh_with_n = mesh.compute_normals(point_normals=True, cell_normals=False)
            normals = np.array(mesh_with_n.point_data["Normals"], dtype=np.float32)

    # 4. Normalise coordinates → coords6
    crange = ns["coord_max"] - ns["coord_min"] + 1e-12
    pts_n  = 2.0 * (pts - ns["coord_min"]) / crange - 1.0
    coords6 = np.concatenate([pts_n, normals], axis=-1)  # (N, 6)

    # 5. Condition vector
    row   = lookup_case(case_key, csv_path)
    cond  = build_cond_vec(row, ns)
    print(f"α={row['alpha_deg']:.2f}°  M={row['M_inf']:.3f}  Re={row['Re_L']:.2e}")

    # 6. Load model & infer
    model = load_model(args.checkpoint, args.device)
    pred  = run_inference(model, coords6, cond, ns, args.device)  # (N, 3)

    # 7. Ground truth arrays from VTK (physical units, already stored)
    gt = np.column_stack([
        np.array(mesh.point_data["cp"]).ravel(),
        np.array(mesh.point_data["cf_x"]).ravel(),
        np.array(mesh.point_data["cf_z"]).ravel(),
    ]).astype(np.float32)  # (N, 3)

    fields = ["cp", "cf_x", "cf_z"] if args.field == "all" else [args.field]

    # 8. Attach arrays to mesh copies for PyVista
    for i, f_name in enumerate(["cp", "cf_x", "cf_z"]):
        mesh[f"{f_name}_gt"]   = gt[:, i]
        mesh[f"{f_name}_pred"] = pred[:, i]
        mesh[f"{f_name}_err"]  = pred[:, i] - gt[:, i]

    # 9. Print per-field error summary
    print("\n{:<5} {:>10}  {:>10}  {:>10}  {:>10}".format(
        "Field", "MSE", "MAE", "RelL1(%)", "RelL2(%)"))
    print("-" * 52)
    for i, f_name in enumerate(["cp", "cf_x", "cf_z"]):
        diff   = pred[:, i] - gt[:, i]
        mse    = float(np.mean(diff ** 2))
        mae    = float(np.mean(np.abs(diff)))
        rel_l1 = float(np.sum(np.abs(diff)) / (np.sum(np.abs(gt[:, i])) + 1e-20))
        rel_l2 = float(np.sum(diff ** 2)   / (np.sum(gt[:, i] ** 2)   + 1e-20))
        print(f"{FIELD_LABELS[f_name]:<5} {mse:>10.3e}  {mae:>10.3e}  "
              f"{rel_l1*100:>9.2f}%  {rel_l2*100:>9.2f}%")

    # 10. Plot
    nf = len(fields)
    pl = pv.Plotter(shape=(nf, 3), window_size=(1600, 400 * nf))
    pl.set_background("white")

    col_titles = ["Ground Truth", "FiLM Predicted", "Error (Pred − GT)"]
    for row_i, f_name in enumerate(fields):
        label = FIELD_LABELS[f_name]
        cmap  = CMAPS[f_name]

        gt_vals   = mesh.point_data[f"{f_name}_gt"]
        pred_vals = mesh.point_data[f"{f_name}_pred"]
        err_vals  = mesh.point_data[f"{f_name}_err"]

        shared_clim = (float(gt_vals.min()), float(gt_vals.max()))
        err_abs     = float(np.abs(err_vals).max())
        err_clim    = (-err_abs, err_abs)

        pl.subplot(row_i, 0)
        pl.add_mesh(mesh.copy(), scalars=f"{f_name}_gt", cmap=cmap,
                    clim=shared_clim, scalar_bar_args={"title": label})
        pl.add_text(f"{label} — {col_titles[0]}", font_size=9)
        pl.camera_position = "yz"

        pl.subplot(row_i, 1)
        pl.add_mesh(mesh.copy(), scalars=f"{f_name}_pred", cmap=cmap,
                    clim=shared_clim, scalar_bar_args={"title": label})
        pl.add_text(f"{label} — {col_titles[1]}", font_size=9)
        pl.camera_position = "yz"

        pl.subplot(row_i, 2)
        pl.add_mesh(mesh.copy(), scalars=f"{f_name}_err", cmap="RdBu_r",
                    clim=err_clim, scalar_bar_args={"title": f"Δ{label}"})
        pl.add_text(f"{label} — {col_titles[2]}", font_size=9)
        pl.camera_position = "yz"

    pl.link_views()
    title = f"FiLM | {case_key}  α={row['alpha_deg']:.1f}°  M={row['M_inf']:.3f}"
    pl.add_text(title, position="upper_left", font_size=10)
    pl.show()


if __name__ == "__main__":
    main()
