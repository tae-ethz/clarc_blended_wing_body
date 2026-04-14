# train_with_frozen_stats.py
import os, json, time, numpy as np, torch
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import UnifiedDesignDataset, design_collate_fn, split_designs
from models.film_model_v1 import FiLMNet

# ----------------------- Paths -----------------------
CSV = "/home/nicksung/Desktop/nicksung/bwb_pp/data/case_with_geom_params.csv"
H5  = "/home/nicksung/Desktop/nicksung/bwb_pp/data/surface_data.hdf5"
NORM_JSON = "norm_stats.json"   # produced from train-only stats earlier
CKPT_DIR  = Path("./checkpoints"); CKPT_DIR.mkdir(parents=True, exist_ok=True)
BEST_PATH  = CKPT_DIR / "film_best.pth"
FINAL_PATH = CKPT_DIR / "film_final.pth"
CFG_PATH   = CKPT_DIR / "train_config.json"

# ----------------------- Hyperparams -----------------------
EPOCHS = 20000
BATCH_SIZE = 64
LR = 5e-4
TRAIN_RATIO = 0.9

# ----------------------- Device & AMP -----------------------
cuda_avail = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_avail else "cpu")
if cuda_avail:
    print(f"✅ CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ CUDA not available. Using CPU.")

use_amp = cuda_avail  # flip to False if you don't want AMP
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# ----------------------- Load norm stats -----------------------
raw = json.load(open(NORM_JSON))
NORM = {k: np.array(v, dtype=np.float32) for k, v in raw.items()}

# ----------------------- Dataset & Loaders -----------------------
ds_full = UnifiedDesignDataset(csv_path=CSV, hdf5_path=H5, norm_stats=NORM, mode="train")
train_ds, val_ds = split_designs(ds_full, train_ratio=TRAIN_RATIO)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=design_collate_fn)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=design_collate_fn)

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

    # Save best
    if val_mse < best_val:
        best_val = val_mse
        torch.save(model.state_dict(), BEST_PATH)
        print(f"[epoch {epoch:03d}] train_mse={train_mse:.4e} | val_mse={val_mse:.4e}  <-- saved BEST")
    else:
        print(f"[epoch {epoch:03d}] train_mse={train_mse:.4e} | val_mse={val_mse:.4e}")

# ----------------------- Save final -----------------------
torch.save(model.state_dict(), FINAL_PATH)
dt = time.time() - t0
print(f"\nDone in {dt/60:.1f} min")
print(f"Best weights:  {BEST_PATH}")
print(f"Final weights: {FINAL_PATH}")
