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

import os, json, time, numpy as np, torch
from pathlib import Path

# --- Your modules ---
from dataset import UnifiedDesignDataset
from models.film_model_v1 import FiLMNet

# ========================== PATHS ==========================
# Run with notebook cwd = repo root, or set BLENDEDNET_DATA / BLENDEDNET_CSV / BLENDEDNET_MODEL.
REPO_ROOT = Path.cwd().resolve()
_DATA = Path(os.environ.get("BLENDEDNET_DATA", REPO_ROOT / "data"))
_CSV = Path(os.environ.get("BLENDEDNET_CSV", REPO_ROOT / "csv_files"))

TRAIN_CSV = str(_CSV / "case_with_geom_params_train.csv")
TRAIN_H5 = str(_DATA / "surface_data.hdf5")
TEST_CSV = str(_CSV / "case_with_geom_params_test.csv")
TEST_H5 = str(_DATA / "surface_data_test.hdf5")

NORM_JSON = str(REPO_ROOT / "norm_stats.json")
MODEL_PATH = str(Path(os.environ.get("BLENDEDNET_MODEL", REPO_ROOT / "checkpoints" / "film_best.pth")))
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE  = int(os.getenv("EVAL_CHUNK", "200000"))  # points per forward pass

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
    # Match your training config exactly
    model = FiLMNet(cond_dim=13, coord_dim=6, output_dim=3,
                    hidden_dim=256, num_layers=4, extra_layers=3)
    sd = torch.load(model_path, map_location=device)
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
    print(f"Device: {DEVICE} | CHUNK_SIZE={CHUNK_SIZE}")

    # 1) Load (or compute) normalization stats
    NORM = load_norm_stats_or_compute(TRAIN_CSV, TRAIN_H5, NORM_JSON)

    # 2) Build TEST dataset with the SAME stats (no randomness here)
    #    NOTE: mode='test' formats case keys as case_000, case_001, ...
    test_ds = UnifiedDesignDataset(csv_path=TEST_CSV,
                                   hdf5_path=TEST_H5,
                                   norm_stats=NORM,
                                   mode="test")
    print(f"Test designs (meshes): {len(test_ds)}")

    # 3) Load model
    model = load_model(MODEL_PATH, device=DEVICE)

    # 4) Full, all-point evaluation
    mse, mae, rel_l1, rel_l2, npts, dt = compute_test_metrics_full(model, test_ds, device=DEVICE)

    # 5) Pretty print
    print(f"\nEvaluated {npts:,} points across all cases in {dt/60:.1f} min\n=== TRUE shape metrics (test, ALL points) ===")
    for k in ["cp", "cf_x", "cf_z"]:
        print(f"{k:4s} | MSE={mse[k]:.6e}  MAE={mae[k]:.6e}  RelL1={rel_l1[k]:.6e}  RelL2={rel_l2[k]:.6e}")
