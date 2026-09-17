# Pattern: Fully Vectorized High-Throughput Batch Processing & Unified Realization

## Summary
- **The Ideal / Objective Goal**: Zero sequential Python `for` loops across candidate molecules in post-processing and thermodynamic evaluation. All molecules in a batch ($B=64$) are stacked into padded device tensors $(B, N_{\rm pad}, 3)$ and $(B, N_{\rm pad})$ and processed concurrently along Axis 0. Multiple distinct physical models (Generalized Born solvation, EGNN quantum MLFF, and Universal Delta-KRR) are realized in a single unified `Tensor.realize(...)` invocation, collapsing host-device synchronization stalls to near zero.
- **Problem**: In batch simulation pipelines, looping sequentially through loaded molecules with `for mat in loaded_materials:` running per-molecule EGNN forward passes, per-molecule Generalized Born electrostatics, and per-molecule KRR calls triggers 63 redundant host-device synchronization points per batch, increases kernel launch overhead by $64\times$, and drops GPU utilization from $>90\%$ to $<10\%$.
- **Root Cause**: Submitting individual small tensor jobs $(1, N_i, 3)$ to the GPU prevents the hardware warp scheduler from filling streaming multiprocessors (SMs), serializes kernel dispatch queues, and incurs CPU driver latency on every single molecule.
- **Actionable Fix**:
  1. Stack all candidate coordinates into a padded batch tensor $(B, N_{\rm pad}, 3)$ and atomic numbers into $(B, N_{\rm pad})$.
  2. Execute batched Generalized Born electrostatics: `gb_solver.compute_solvation_free_energy(coords, charges, z, mask)`.
  3. Execute batched EGNN energy and force evaluation: `egnn_ff.compute_energy_and_forces(coords, z, mask)`.
  4. Execute batched Delta-KRR residual inference: `predict_krr_residual_tensor(z_mol, d_phys, s_solv)`.
  5. Realize all output tensors simultaneously in a single call: `Tensor.realize(tot_solv_t, gb_mean_t, q_mean_t, u_egnn_t, f_egnn_t, krr_res_t, krr_density_t)`.
  6. Extract per-molecule scalars and atomic vectors via vectorized NumPy slicing (`q_mean_np[b, :n_sites_real]`).
- **Related Skills / Modules**: `tinygrad-performance`, `dens_city.utils.pipeline`, `dens_city.cdft`

---

## The Ideal Architectural Standard

```
+--------------------------------------------------------------------------------+
|                   IDEAL BATCHED POST-PROCESSING PIPELINE                       |
+--------------------------------------------------------------------------------+
|  Input: Loaded materials [mat_0, mat_1, ... mat_{B-1}]                        |
|                                                                                |
|  1. Batched Device Stacking:                                                   |
|     coords_t: (B, N_pad, 3)                                                    |
|     z_t     : (B, N_pad)                                                       |
|     mask_t  : (B, N_pad, 1)                                                    |
|                                                                                |
|  2. Parallel Graph Dispatch:                                                   |
|     GB Solvation  --> gb_tensor_b (B,)                                         |
|     EGNN MLFF     --> u_egnn_b (B,), f_egnn_b (B, N_pad, 3)                    |
|     Delta-KRR     --> krr_res_b (B, 1), krr_density_b (B, 1)                   |
|                                                                                |
|  3. Unified Device Realization (1 Sync Point):                                 |
|     Tensor.realize(gb_tensor_b, u_egnn_b, f_egnn_b, krr_res_b, krr_density_b)  |
|                                                                                |
|  4. Vector Sliced Extraction:                                                  |
|     For each b in 0..B-1:                                                      |
|       q_b = q_np[b, :mat_b.num_sites]                                          |
|       f_b = f_np[b, :mat_b.num_sites]                                          |
+--------------------------------------------------------------------------------+
```

```python
# The Ideal: Fully Vectorized Batch Evaluation
coords_np = np.zeros((batch_size, N_pad, 3), dtype=np.float32)
z_np = np.zeros((batch_size, N_pad), dtype=np.float32)
mask_np = np.zeros((batch_size, N_pad, 1), dtype=np.float32)

for b, mat in enumerate(loaded_materials):
    n = mat.num_sites
    coords_np[b, :n] = mat.coords
    z_np[b, :n] = mat.atomic_numbers
    mask_np[b, :n] = 1.0

coords_t = Tensor(coords_np, dtype=dtypes.float32)
z_t = Tensor(z_np, dtype=dtypes.float32)
mask_t = Tensor(mask_np, dtype=dtypes.float32)

# Unified parallel forward passes
gb_solv = gb_solver.compute_solvation_free_energy(coords_t, charges_t, z_t, mask_t)
u_egnn, f_egnn = egnn_model.compute_energy_and_forces(coords_t, z_t, mask_t)
krr_res, krr_dens = predict_krr_residual_tensor(h_mol_t, d_phys_t, s_solv_t)

# Single realization point for the entire batch
Tensor.realize(gb_solv, u_egnn, f_egnn, krr_res, krr_dens)

# Vectorized slice extraction
gb_vals = gb_solv.numpy()
f_vals = f_egnn.numpy()
for b, mat in enumerate(loaded_materials):
    real_f = f_vals[b, :mat.num_sites]
    f_rms = float(np.sqrt(np.mean(real_f ** 2)))
```

---

## Anti-Patterns to Contrast Against (What Bad Code Looks Like)

- ❌ **Anti-Pattern (Serial Molecule Looping)**: Writing `for mat in loaded_materials: egnn.compute_energy(mat.coords)` inside a batch task executor. This serializes $B$ independent evaluations, launches thousands of tiny GPU kernels, and blocks asynchronous prefetching.
- ❌ **Anti-Pattern (Scattered Synchronous Realizations)**: Calling `Tensor.realize()` or `.numpy()` sequentially for each output tensor across multiple models (`gb.realize(); u.realize(); f.realize(); krr.realize()`), forcing multiple redundant round-trip host-device synchronizations.
- ❌ **Anti-Pattern (Per-Atom Dynamic Allocation)**: Allocating new NumPy coordinate arrays and GPU buffers with varying sizes $N_i$ for each individual candidate instead of reusing padded contiguous batch buffers.
