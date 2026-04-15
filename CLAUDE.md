# clarc_blended_wing_body

FiLM-modulated MLP surrogate for BWB aerodynamic surface loads (Cp, Cf_x, Cf_z).
Trained on 8,831 FUN3D RANS cases from the MIT Lincoln Lab / DeCoDe BlendedNet dataset.

## Scaling & Coordinate Conventions

The geometry is **non-dimensional** (unit-scale, C₁ = 1 m model). Re encodes physical
vehicle size via flow similarity. The dataset sampled centerline length ∈ [0.1, 10] m
(log-uniform) and computed Re from altitude + Mach + length. All CFD ran at unit scale.
See Sung et al. "BlendedNet++", arXiv:2512.03280, Section 3 and Table 5.

**Coordinates**: x ∈ [0, 1.16], y ∈ [-1.06, 1.06], z ∈ [-0.09, 0.09] (meters at unit scale).
Normalized to [-1, 1] via `coord_min/max` in `norm_stats.json` before feeding to the net.
Surface normals are concatenated raw (not normalized to [-1,1]).

**Shape params in CSV are in mm** (C₁ = 1000 mm). The SDF net uses ratios to C₁ (e.g.,
B₁/C₁ ∈ [0.10, 0.20]). Multiply SDF params by 1000 to get FiLM conditioning values.
Sweep angles (S₁, S₂, S₃) are in degrees in both conventions.

**Conditioning (13-dim, z-normalized with `norm_stats.json`):**
- `[0:3]` flight: Re_L (raw, NOT log₁₀), M_inf, alpha_deg
- `[3:13]` shape: B1, B2, B3, C1 (=1000, dead), C2, C3, C4, S1, S2, S3

**Note**: BlendedNet++ paper uses log₁₀ Re_L. This codebase uses raw Re_L. No log
transform exists in `dataset.py`. Switching to log₁₀ would require retraining.

## B101 Integration

This project is a component of the **B101 BWB Multi-Physics MDO Benchmark**.
The parent overview lives at:

```
/Users/taebersold/Code/GLUE2-Applications/b101_bwb/overview.md
```

**When you make meaningful changes** to this project (new checkpoint, architecture change,
updated normalization stats, changed I/O contract, new training results), update the
corresponding sections in that overview file:

1. The **status cell** in the "Subproject Repositories" table
2. The **"Aero Loads"** summary section (checkpoint name, val MSE, epoch)
3. The **"Cp/Cf FiLM net"** row in the "Surrogates (detail)" table
4. The detailed **"Cp/Cf FiLM net"** section (architecture, inputs, outputs)
5. The **"Open Items"** checklist if any items are completed or added
