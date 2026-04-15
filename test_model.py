# %%
#!/usr/bin/env python3
"""
Evaluate FiLMNet on TEST split using *all points of all cases* (TRUE shape only).

- Loads train-time norm_stats.json if available (reproducible normalization).
- Otherwise, computes stats from TRAIN CSV+HDF5 as a fallback.
- Iterates every design -> every case -> all points (chunked) with NO subsampling.
- Reports MSE, MAE, RelL1, RelL2 for cp, cf_x, cf_z.

Requires:
  - dataset.py providing UnifiedDesignDataset
  - models/film_model_v1.py providing FiLMNet
"""

import argparse, os, json, time, numpy as np, torch
from pathlib import Path

# --- Your modules ---
from dataset import UnifiedDesignDataset
from models.film_model_v1 import FiLMNet

# ========================== PATHS ==========================
REPO_ROOT = Path(__file__).parent.resolve()
_DATA = Path(os.environ.get("BLENDEDNET_DATA", REPO_ROOT / "data"))
_CSV = Path(os.environ.get("BLENDEDNET_CSV", REPO_ROOT / "csv_files"))

TRAIN_CSV = str(_CSV / "case_with_geom_params_train.csv")
TRAIN_H5 = str(_DATA / "surface_data.hdf5")
TEST_CSV = str(_CSV / "case_with_geom_params_test.csv")
TEST_H5 = str(_DATA / "surface_data_test.hdf5")

NORM_JSON = str(REPO_ROOT / "norm_stats.json")
DEVICE      = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
CHUNK_SIZE  = int(os.getenv("EVAL_CHUNK", "200000"))  # points per forward pass

# ── Paper baseline (BlendedNet ASME IDETC 2025, GT-conditioned) ────────────────
PAPER_BASELINE = {
    "cp":   dict(MSE=7.86e-3, MAE=3.72e-2, RelL1=13.52, RelL2=3.11),
    "cf_x": dict(MSE=2.80e-5, MAE=1.35e-3, RelL1=22.09, RelL2=7.74),
    "cf_z": dict(MSE=1.51e-5, MAE=7.96e-4, RelL1=30.01, RelL2=18.79),
}

# ===================== HELPERS ============================
def load_norm_stats_or_compute(train_csv, train_h5, norm_json_path):
    """
    Prefer loading train-time stats from JSON for exact reproducibility.
    If not found, compute from TRAIN data via dataset logic and save them.
    """
    if os.path.isfile(norm_json_path):
        raw = json.load(open(norm_json_path, "r"))
        return {k: np.array(v, dtype=np.float32) for k, v in raw.items()}

    tmp_train = UnifiedDesignDataset(csv_path=train_csv, hdf5_path=train_h5, norm_stats=None, mode="train")
    ns = tmp_train.norm_stats
    norm_stats = {k: np.array(v, dtype=np.float32) for k, v in ns.items()}
    try:
        with open(norm_json_path, "w") as f:
            json.dump({k: v.tolist() for k, v in norm_stats.items()}, f, indent=2)
    except Exception:
        pass
    return norm_stats

def load_model(model_path, device=DEVICE):
    model = FiLMNet(cond_dim=13, coord_dim=6, output_dim=3,
                    hidden_dim=256, num_layers=4, extra_layers=3)
    sd = torch.load(model_path, map_location=device, weights_only=False)
    # Handle full checkpoint dicts (with optimizer / epoch)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd)
    model.to(device).eval()
    return model

@torch.no_grad()
def compute_test_metrics_full(model, test_ds, device=DEVICE, chunk_size=CHUNK_SIZE):
    """
    Iterates every design -> every case -> all points (chunked), no subsampling.
    Returns dicts for MSE, MAE, RelL1, RelL2 keyed by ['cp', 'cf_x', 'cf_z'].
    """
    # Accumulators (per-channel: cp, cf_x, cf_z)
    sse = np.zeros(3, dtype=np.float64)      # sum of squared errors
    sae = np.zeros(3, dtype=np.float64)      # sum of absolute errors
    sum_gt_abs = np.zeros(3, dtype=np.float64)
    sum_gt_sq  = np.zeros(3, dtype=np.float64)
    total_pts  = 0

    # Convenience for un-normalization using the same stats the dataset exposes
    out_mu = torch.tensor(test_ds.output_mean, device=device).view(1, 3)
    out_sd = torch.tensor(test_ds.output_std,  device=device).view(1, 3)

    t_start = time.time()
    n_designs = len(test_ds)
    for d_idx in range(n_designs):
        design = test_ds[d_idx]  # list of case dicts for this mesh/design
        for sample in design:
            coords  = torch.from_numpy(sample["points"]).to(device=device, dtype=torch.float32)   # [N,6]
            targets = torch.from_numpy(sample["coeffs"]).to(device=device, dtype=torch.float32)   # [N,3] (normalized)
            cond    = torch.from_numpy(sample["flight_cond"]).to(device=device, dtype=torch.float32)  # [13]
            # Expand cond to per-point
            cond_b  = cond.unsqueeze(0).expand(coords.size(0), -1)  # [N,13]

            # Process in chunks to avoid OOM
            N = coords.size(0)
            for s in range(0, N, chunk_size):
                e = min(s + chunk_size, N)
                preds = model(coords[s:e], cond_b[s:e])      # [M,3], normalized
                # Unnormalize to physical units
                p_u = preds * out_sd + out_mu                # [M,3]
                t_u = targets[s:e] * out_sd + out_mu

                diff = p_u - t_u
                sse += (diff ** 2).sum(dim=0).detach().cpu().numpy()
                sae += diff.abs().sum(dim=0).detach().cpu().numpy()
                sum_gt_abs += t_u.abs().sum(dim=0).detach().cpu().numpy()
                sum_gt_sq  += (t_u ** 2).sum(dim=0).detach().cpu().numpy()
                total_pts  += (e - s)

    eps   = 1e-20
    mse   = sse / max(total_pts, 1)
    mae   = sae / max(total_pts, 1)
    rel1  = sae / (sum_gt_abs + eps)
    rel2  = sse / (sum_gt_sq  + eps)

    keys = ["cp", "cf_x", "cf_z"]
    return (
        dict(zip(keys, mse)),
        dict(zip(keys, mae)),
        dict(zip(keys, rel1)),
        dict(zip(keys, rel2)),
        total_pts,
        time.time() - t_start
    )

# ====================== MAIN =============================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate FiLMNet on the test set")
    ap.add_argument("--checkpoint", type=str,
                    default=str(REPO_ROOT / "checkpoints" / "film_best.pth"),
                    help="Path to model checkpoint (default: checkpoints/film_best.pth)")
    args = ap.parse_args()

    MODEL_PATH = args.checkpoint
    print(f"Device    : {DEVICE} | CHUNK_SIZE={CHUNK_SIZE}")
    print(f"Checkpoint: {MODEL_PATH}")

    # 1) Load (or compute) normalization stats
    NORM = load_norm_stats_or_compute(TRAIN_CSV, TRAIN_H5, NORM_JSON)

    # 2) Build TEST dataset
    test_ds = UnifiedDesignDataset(csv_path=TEST_CSV,
                                   hdf5_path=TEST_H5,
                                   norm_stats=NORM,
                                   mode="test")
    print(f"Test designs (meshes): {len(test_ds)}")

    # 3) Load model
    model = load_model(MODEL_PATH, device=DEVICE)

    # 4) Full, all-point evaluation
    mse, mae, rel_l1, rel_l2, npts, dt = compute_test_metrics_full(model, test_ds, device=DEVICE)

    # 5) Comparison table vs paper baseline
    ckpt_name = Path(MODEL_PATH).stem
    keys = ["cp", "cf_x", "cf_z"]

    print(f"\nEvaluated {npts:,} points across all test cases in {dt/60:.1f} min\n")
    print(f"{'':30s} {'Cp':>20s}  {'Cf_x':>20s}  {'Cf_z':>20s}")
    print("-" * 94)

    def _row(label, vals_dict, fmt):
        parts = [f"{vals_dict[k]:{fmt}}" for k in keys]
        print(f"  {label:<28s} {'  '.join(f'{p:>20s}' for p in parts)}")

    print(f"  {'Metric':<28s} {'Cp':>20s}  {'Cf_x':>20s}  {'Cf_z':>20s}")
    print("  " + "-" * 90)
    for metric, our_vals, paper_key in [
        ("MSE",   mse,    "MSE"),
        ("MAE",   mae,    "MAE"),
        ("Rel L1 (%)", {k: rel_l1[k]*100 for k in keys}, "RelL1"),
        ("Rel L2 (%)", {k: rel_l2[k]*100 for k in keys}, "RelL2"),
    ]:
        # Our model row
        our_strs = {k: f"{our_vals[k]:.4e}" if "%" not in metric else f"{our_vals[k]:.2f}%" for k in keys}
        _row(f"Ours ({ckpt_name}): {metric}", our_strs, "s")
        # Paper baseline row
        paper_strs = {k: (f"{PAPER_BASELINE[k][paper_key]:.4e}" if "%" not in metric
                          else f"{PAPER_BASELINE[k][paper_key]:.2f}%") for k in keys}
        _row(f"BlendedNet (paper): {metric}", paper_strs, "s")
        print()

    # Compact delta summary
    print("\n  Δ = Ours − Paper  (negative = better than paper)")
    print("  " + "-" * 54)
    print(f"  {'':28s} {'Cp':>8s}  {'Cf_x':>8s}  {'Cf_z':>8s}")
    for metric, our_vals, paper_key, pct in [
        ("MSE",    mse,    "MSE",   False),
        ("MAE",    mae,    "MAE",   False),
        ("RelL1%", {k: rel_l1[k]*100 for k in keys}, "RelL1", True),
        ("RelL2%", {k: rel_l2[k]*100 for k in keys}, "RelL2", True),
    ]:
        deltas = {k: our_vals[k] - PAPER_BASELINE[k][paper_key] for k in keys}
        fmt = ".2f" if pct else ".2e"
        parts = "  ".join(f"{deltas[k]:>+8{fmt}}" for k in keys)
        print(f"  {metric:<28s} {parts}")
