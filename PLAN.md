# Surrogate Modeling Plan

> **Part of [B101 BWB MDO Benchmark](/Users/taebersold/Code/GLUE2-Applications/b101_bwb/overview.md)**. Update that overview when checkpoints, architecture, or training results change.

## Goal

Build a differentiable surrogate benchmark for blended-wing-body aerodynamics that supports:

- Fast prediction of global coefficients: `CL`, `CD`, `Cm`
- Fast prediction of distributed surface fields: `Cp`, `Cf_x`, `Cf_z`
- Smooth gradients with respect to geometry parameters and query-point inputs
- Batched inference across many geometries for methods experiments

This is intended as a research benchmark for ML / optimization methods, not as a production engineering surrogate.

## Current Assumptions

- We already have the geometry parameters, so PointNet is not needed for the main workflow.
- We want differentiable models, so we will use MLP-based surrogates rather than tree-based models.
- Query surfaces may come from `torchcad` meshing and may be relatively coarse / uniform.
- We can accept some loss of local load fidelity as long as the setup is consistent and benchmarkable.

## Surrogate Split

We will use two separate models.

### 1. Global Aerodynamic Surrogate

Input:

- Geometry parameters
- Flight conditions

Output:

- `CL`
- `CD`
- `Cm`

Model:

- A simple smooth MLP
- Prefer smooth activations such as `SiLU`, `Softplus`, or similar over plain `ReLU`
- Keep the network small and fast enough for large batched evaluation

Why:

- This avoids reconstructing global coefficients by integrating predicted surface fields
- It is simpler, cheaper, and better aligned with the intended benchmark use case

### 2. Field Surrogate

Input per query point:

- Surface coordinates: `x, y, z`
- Surface normals: `n_x, n_y, n_z`
- Geometry parameters
- Flight conditions

Output per query point:

- `Cp`
- `Cf_x`
- `Cf_z`

Model:

- A conditional MLP / FiLM-style neural field
- Conditioning comes from geometry parameters + flight conditions
- The field is evaluated at arbitrary query points and normals

Why:

- This gives distributed aerodynamic quantities on whatever surface discretization we supply
- It stays differentiable with respect to inputs
- It does not require mesh connectivity, only points and normals

## Notes On Targets

- `Cp` is the pressure coefficient and is expected to be the primary field target for structural loading.
- `Cf_x` and `Cf_z` are skin-friction coefficient components in the body axes.
- For early structural-style studies, `Cp` will likely matter most.
- `Cf_x` / `Cf_z` can be retained as optional or secondary targets if they help downstream metrics.

## Data Status

Local dataset currently available in:

- `data/`

Available now:

- Integrated case data for `CL`, `CD`, `CMy`
- Geometry parameter files

Missing from the copied dataset snapshot:

- Distributed `vtk` surface outputs

Implication:

- We can begin with the global surrogate immediately.
- The field surrogate depends on obtaining the distributed surface data or a preprocessed equivalent.

## Mesh Strategy

- The benchmark will likely use a coarse or semi-fine query mesh from `torchcad`.
- We may not have robust local mesh refinement.
- That is acceptable for this benchmark as long as the meshing procedure is fixed and clearly documented.

Planned stance:

- Use a consistent meshing recipe across experiments
- Accept reduced local fidelity if necessary
- Run resolution studies later to quantify sensitivity
- Frame results relative to the chosen discretization, not as high-fidelity engineering truth

## Differentiability Requirements

We want gradients with respect to:

- Geometry parameters
- Flight conditions
- Query-point coordinates

Important caveat:

- Gradients with respect to the mesh are only meaningful through the chosen mesh parameterization
- If normals are computed in a non-differentiable preprocessing step, gradients through normals may be approximate or unavailable

Practical interpretation:

- Treat points and normals as direct model inputs
- Optimize with respect to geometry parameters and query locations when needed
- Keep model activations smooth enough to avoid overly noisy gradients

## Inference Efficiency

Global surrogate:

- Should be very cheap
- Suitable for large batched evaluation

Field surrogate:

- Cost scales with `number of geometries x number of query points`
- Batched inference over many geometries should be feasible with chunking

Implementation note:

- The conditioning branch should be computed once per geometry / flight case, then broadcast over query points
- Avoid recomputing conditioning independently for every point if possible

## Planned Work Sequence

1. Fork the repository and set remotes cleanly.
2. Build a clean training pipeline for the global `CL/CD/Cm` MLP using the local dataset.
3. Define a smooth MLP architecture and normalization scheme for the global surrogate.
4. Evaluate accuracy, calibration, and gradient behavior for the global model.
5. Obtain the distributed surface dataset needed for `Cp`, `Cf_x`, `Cf_z`.
6. Build the field surrogate as a conditional neural field.
7. Add efficient batched / chunked inference for large query sets.
8. Run mesh-resolution studies and report sensitivity.

## Success Criteria

- Global surrogate predicts `CL`, `CD`, `Cm` accurately and robustly across the dataset split
- Field surrogate predicts usable surface loads on user-supplied query meshes
- Both surrogates are differentiable and stable enough for optimization / methods experiments
- Inference is fast enough to evaluate many geometries in parallel
- The benchmark setup is simple, reproducible, and clearly documented
