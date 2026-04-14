#!/usr/bin/env python3
import os, json, time, numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import UnifiedDesignDataset, design_collate_fn, split_designs
from models.film_model_v1 import FiLMNet

# ----------------------- Paths -----------------------
REPO_ROOT = Path(__file__).resolve().parent
_DATA = Path(os.environ.get("BLENDEDNET_DATA", REPO_ROOT / "data"))
_CSV = Path(os.environ.get("BLENDEDNET_CSV", REPO_ROOT / "csv_files"))

CSV = str(_CSV / "case_with_geom_params_train.csv")
H5  = str(_DATA / "surface_data.hdf5")
NORM_JSON = str(REPO_ROOT / "norm_stats.json")
CKPT_DIR  = Path(os.environ.get("BLENDEDNET_CHECKPOINTS", REPO_ROOT / "checkpoints"))
CKPT_DIR.mkdir(parents=True, exist_ok=True)
BEST_PATH  = CKPT_DIR / "film_best.pth"
FINAL_PATH = CKPT_DIR / "film_final.pth"
CFG_PATH   = CKPT_DIR / "train_config.json"

# ----------------------- Hyperparams -----------------------
EPOCHS = int(os.getenv("FILM_EPOCHS", "20000"))
BATCH_SIZE = int(os.getenv("FILM_BATCH_SIZE", "64"))
LR = float(os.getenv("FILM_LR", "5e-4"))
TRAIN_RATIO = float(os.getenv("FILM_TRAIN_RATIO", "0.9"))
NUM_WORKERS = int(os.getenv("FILM_NUM_WORKERS", "0"))

# ----------------------- Device & AMP -----------------------
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(0)
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    device_name = "Apple Metal (MPS)"
else:
    device = torch.device("cpu")
    device_name = "CPU"
print(f"Using device: {device_name}")

use_amp = device.type == "cuda"
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# ----------------------- Load norm stats -----------------------
if os.path.isfile(NORM_JSON):
    raw = json.load(open(NORM_JSON))
    NORM = {k: np.array(v, dtype=np.float32) for k, v in raw.items()}
else:
    tmp_ds = UnifiedDesignDataset(csv_path=CSV, hdf5_path=H5, norm_stats=None, mode="train")
    NORM = {k: np.array(v, dtype=np.float32) for k, v in tmp_ds.norm_stats.items()}
    with open(NORM_JSON, "w") as f:
        json.dump({k: v.tolist() for k, v in NORM.items()}, f, indent=2)
    tmp_ds.close()

# ----------------------- Dataset & Loaders -----------------------
ds_full = UnifiedDesignDataset(csv_path=CSV, hdf5_path=H5, norm_stats=NORM, mode="train")
train_ds, val_ds = split_designs(ds_full, train_ratio=TRAIN_RATIO)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=design_collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=device.type == "cuda",
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=design_collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=device.type == "cuda",
)

# ----------------------- Model -----------------------
model = FiLMNet(cond_dim=13, coord_dim=6, output_dim=3,
                hidden_dim=256, num_layers=4, extra_layers=3).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)

# ----------------------- Save config -----------------------
cfg = dict(
    csv=str(CSV), h5=str(H5), norm_json=str(NORM_JSON),
    epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, train_ratio=TRAIN_RATIO,
    device=str(device), amp=use_amp, model=dict(cond_dim=13, coord_dim=6, output_dim=3,
                                                hidden_dim=256, num_layers=4, extra_layers=3)
)
json.dump(cfg, open(CFG_PATH, "w"), indent=2)

# ----------------------- Train/Val Loop -----------------------
LOSS_PLOT = CKPT_DIR / "loss_curves.png"
train_losses, val_losses = [], []
best_val = float("inf")
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    count = 0

    for coords, conds, targets in train_loader:
        coords, conds, targets = coords.to(device), conds.to(device), targets.to(device)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(coords, conds)
            loss = torch.mean((pred - targets) ** 2)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        running += loss.item() * coords.size(0)
        count   += coords.size(0)

    train_mse = running / max(count, 1)

    # ---- validation ----
    model.eval()
    v_running, v_count = 0.0, 0
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
        for coords, conds, targets in val_loader:
            coords, conds, targets = coords.to(device), conds.to(device), targets.to(device)
            pred = model(coords, conds)
            vloss = torch.mean((pred - targets) ** 2)
            v_running += vloss.item() * coords.size(0)
            v_count   += coords.size(0)
    val_mse = v_running / max(v_count, 1)

    train_losses.append(train_mse)
    val_losses.append(val_mse)

    if epoch % 500 == 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogy(train_losses, label="train", linewidth=0.8)
        ax.semilogy(val_losses, label="val", linewidth=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.set_title(f"FiLM Training — epoch {epoch}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(LOSS_PLOT, dpi=150)
        plt.close(fig)

    # Save best
    saved = False
    if val_mse < best_val:
        best_val = val_mse
        torch.save(model.state_dict(), BEST_PATH)
        saved = True

    if saved or epoch % 10 == 0 or epoch == 1:
        tag = "  <-- BEST" if saved else ""
        print(f"[epoch {epoch:05d}] train={train_mse:.4e} | val={val_mse:.4e}{tag}")

# ----------------------- Save final -----------------------
torch.save(model.state_dict(), FINAL_PATH)
dt = time.time() - t0
print(f"\nDone in {dt/60:.1f} min")
print(f"Best weights:  {BEST_PATH}")
print(f"Final weights: {FINAL_PATH}")
ds_full.close()
