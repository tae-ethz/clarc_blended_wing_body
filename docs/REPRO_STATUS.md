# FiLM Reproduction Status

## Goal

Reproduce the original `film_model_v1` training and evaluation setup from the upstream BlendedNet repository using the local dataset under `data/`.

The intended reproduction path is:

1. Build merged train/test CSV files from the raw split files.
2. Convert raw train/test VTK surface files into HDF5 surface datasets.
3. Train the original FiLM model with the same architecture and hyperparameters.
4. Evaluate on the held-out test split and compare errors against the paper / upstream README.

## What Has Already Been Done

### Repository setup

- Forked the repository to `tae-ethz/clarc_blended_wing_body`.
- Set git remotes so:
  - `origin` points to the fork
  - `upstream` points to Nicolas Sung's original repository
- Fixed local branch tracking so `main` tracks `origin/main`.

### Local dataset layout

- Copied the local dataset into `data/`.
- Flattened the dataset so the top-level local data layout is:
  - `data/train`
  - `data/test`
  - `data/README.txt`
  - `data/LICENSE.txt`
  - `data/model.vsp3`
- Confirmed both train and test splits include:
  - `case_data.dat`
  - `geom_params.ini`
  - `vtk/`

### Documentation cleanup

- Moved the original upstream `README.md` to:
  - `docs/ORIGINAL_REPO_README.md`
- Added a planning document:
  - `PLAN.md`

### Code changes already made

The following files were edited to support a local-path reproduction workflow:

- `dataset.py`
  - made the loader tolerate the local raw schema
  - supports `Re` -> `Re_L`
  - supports `Cmy` -> `CMy`
  - supports `geom_name` -> `mesh`
  - supports the shape-column schemas with either `S2` or `X3`

- `train_model.py`
  - rewired to use local paths under `data/` and `csv_files/`
  - still uses the original `film_model_v1`
  - still keeps the original architecture defaults:
    - `cond_dim=13`
    - `coord_dim=6`
    - `output_dim=3`
    - `hidden_dim=256`
    - `num_layers=4`
    - `extra_layers=3`
  - computes `norm_stats.json` from train data if it is missing

- `create_hdf5.py`
  - changed from a hard-coded one-off script to a CLI tool
  - now accepts:
    - `--data-dir`
    - `--out-h5`
    - `--offset`
    - `--limit`
    - `--append`
  - this was done to make train/test conversion possible from the local dataset and to support batched recovery attempts

- `prepare_blendednet_data.py`
  - added as a wrapper to:
    - build merged train/test CSVs
    - build train/test HDF5 files from the raw VTK directories

- `convert_vtk_safe.py` (new, 2026-04-14)
  - crash-safe parallel VTK -> HDF5 converter
  - phase 1: extracts VTK -> `.npz` intermediates in parallel subprocesses (isolates VTK segfaults)
  - phase 2: merges all `.npz` into a single HDF5
  - supports `--workers` (default 8), `--batch-size` (default 100), `--keep-npz`
  - resumes from existing `.npz` cache if interrupted

## What Worked

### CSV generation (complete, 2026-04-14)

- train CSV: `csv_files/case_with_geom_params_train.csv` — **8830 rows**, 21 columns
- test CSV: `csv_files/case_with_geom_params_test.csv` — **870 rows**, 21 columns

### HDF5 generation (complete, 2026-04-14)

- train HDF5: `data/surface_data.hdf5` — **8831 cases**
- test HDF5: `data/surface_data_test.hdf5` — **869 cases**

Built using `convert_vtk_safe.py` with subprocess isolation per VTK file.

Each case contains: `points (N,3)`, `normals (N,3)`, `cp (N,)`, `cf_x (N,)`, `cf_y (N,)`, `cf_z (N,)` — all float32, gzip-compressed.

### Dependency installation

The following Python packages were installed locally for the preprocessing path:

- `pyvista`
- `vtk`
- `h5py`

## What Failed (resolved)

### VTK reader instability on macOS

During long preprocessing runs in a single process, the VTK C++ reader would accumulate state and eventually segfault (signal 11). This was misdiagnosed as file corruption — a full scan of all 9702 VTK files confirmed every file reads successfully in isolation.

Two files consistently segfault during `compute_normals()` even in isolated subprocesses:

- `data/train/vtk/case_7709.vtk` — segfault (signal 11)
- `data/test/vtk/case_319.vtk` — segfault (signal 11)

These 2 files (0.02% of the dataset) are excluded from the HDF5 datasets. All other VTK files read and convert successfully.

### HDF5 corruption after append-mode recovery (no longer relevant)

The original `prepare_blendednet_data.py` used append-mode HDF5 writes. When a VTK segfault killed the process mid-write, the HDF5 file was left in a corrupted state. This is fully resolved by `convert_vtk_safe.py`, which extracts to `.npz` intermediates first and writes the HDF5 once at the end.

## Current State

All preprocessing is complete. The data pipeline is ready for training.

- CSV files: complete (8830 train, 870 test)
- HDF5 files: complete (8831 train, 869 test)
- Missing cases: 2 (case_7709 train, case_319 test) — VTK segfault in normals computation

## What Should Be Done Next

### Immediate next step: train the FiLM model on Euler (RTX 4090)

Data has been rsync'd to `/cluster/scratch/taebersold/clarc_blended_wing_body/`.

#### 1. Setup on Euler

```bash
cd /cluster/scratch/taebersold/clarc_blended_wing_body

# Move data files into expected subdirs (rsync landed them flat)
mkdir -p data csv_files
mv surface_data.hdf5 surface_data_test.hdf5 data/
mv case_with_geom_params_*.csv geom_params_*.csv csv_files/

# norm_stats.json stays in repo root (already there)

# Clone the repo if not already present, or git pull
git pull origin main

# Create venv and install deps
python3 -m venv .venv
.venv/bin/pip install torch torchvision h5py pandas numpy matplotlib
```

#### 2. Train

```bash
# Submit as a GPU job, or run interactively:
FILM_EPOCHS=20000 FILM_NUM_WORKERS=4 .venv/bin/python train_model.py
```

Environment variables for tuning:
- `FILM_EPOCHS` — default 20000, start with 1000 to verify convergence
- `FILM_BATCH_SIZE` — default 64, can increase on 4090 (24 GB VRAM)
- `FILM_NUM_WORKERS` — default 0, set to 4-8 on Linux to overlap data loading with GPU
- `FILM_LR` — default 5e-4

CUDA AMP is enabled automatically when `torch.cuda.is_available()`.

Loss curve saved to `checkpoints/loss_curves.png` every 500 epochs (overwritten).
Best model saved to `checkpoints/film_best.pth` whenever val MSE improves.

#### 3. Evaluate

```bash
.venv/bin/python test_model.py
```

Compare MSE, MAE, RelL1, RelL2 for cp, cf_x, cf_z against upstream README.

#### Important notes

- `dataset.py` loads point data lazily from HDF5 (not into RAM). This is critical — the HDF5 is 15 GB. Setting `NUM_WORKERS > 0` on Linux is fine (h5py handles it).
- `norm_stats.json` was pre-computed from train data on macOS and transferred. Do NOT delete it — recomputing scans the full HDF5.
- Two VTK cases are missing from the HDF5s (VTK segfault on macOS): train/case_7709, test/case_319. This is negligible (0.02%).
- If HDF5 is on network storage (`$SCRATCH`), consider copying to `$TMPDIR` for the job to avoid I/O bottleneck:
  ```bash
  cp data/surface_data.hdf5 $TMPDIR/
  BLENDEDNET_DATA=$TMPDIR .venv/bin/python train_model.py
  ```

## Files Relevant To This Work

- `PLAN.md`
- `docs/ORIGINAL_REPO_README.md`
- `docs/REPRO_STATUS.md`
- `dataset.py`
- `train_model.py`
- `test_model.py`
- `create_hdf5.py`
- `convert_vtk_safe.py`
- `prepare_blendednet_data.py`
