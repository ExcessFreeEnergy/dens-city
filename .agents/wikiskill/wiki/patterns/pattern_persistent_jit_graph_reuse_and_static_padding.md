# Pattern: Universal Static Tensor Padding & In-Place JIT Execution Graph Reuse

## Summary
- **The Ideal / Objective Goal**: Zero JIT kernel recompilation overhead between arbitrary molecular batches. Execution graphs are traced and compiled exactly once at application startup. Thousands of diverse molecular systems execute against a fixed, persistent `@TinyJit` execution schedule by dynamically binding data in-place via `.assign()` into pre-allocated static buffers.
- **Problem**: Changing molecular sizes, varying site counts, or dynamic convolution kernel lengths between batches repeatedly invalidate Tinygrad's JIT cache (`TinyJit`), triggering expensive GPU kernel recompilation on every batch that destroys throughput (e.g. 21.5s vs 0.02s per batch).
- **Root Cause**: Tinygrad's compiler keys compiled execution schedules strictly by tensor buffer shapes, memory strides, and UOp DAG layouts. Any shape variation (even a 1-element difference in grouped convolution kernel size) forces a schedule miss and full re-tracing.
- **Actionable Fix**: 
  1. Fix all molecular batch shapes to static maximum bounds (e.g., $B=64, N_{\rm pad}=128$).
  2. Pad all grouped convolution kernels (Rosenfeld FMT weights and WCA attractive dispersion) to universal static dimensions ($K_{\rm fmt}=41, K_{\rm att}=129$) with symmetric center padding and exact zeros outside compact physical support ($R_i \le \sigma_{\rm max}/2$).
  3. Re-bind batch parameters in-place using `.assign()` on persistent singleton solver instances (`get_or_create_batched_cdft`) without reconstructing optimizer state buffers (`opt.m`, `opt.v`).
- **Related Skills / Modules**: `tinygrad-jit`, `dens_city.cdft`, `dens_city.boltzmann`

---

## The Ideal Architectural Standard

In a high-throughput computational physics engine, compiling kernels during simulation loops is considered an architectural defect. The ideal execution pipeline satisfies:

$$\text{JIT Kernel Recompilations after Batch 0} = 0$$

```
+--------------------------------------------------------------------------------+
|                             IDEAL JIT LIFECYCLE                                |
+--------------------------------------------------------------------------------+
|  Startup: Pre-allocate static device buffers (B=64, N_pad=128, K_att=129)       |
|  Batch 0: Trace and compile @TinyJit execution graph on initial batch          |
|  Batch 1..M:                                                                   |
|    1. Pad batch data to static shapes (B, N_pad) and (B, 1, K_static, 1)       |
|    2. Re-bind buffers in-place: tensor_buf.assign(new_data).realize()          |
|    3. Reset optimizer moments in-place: opt.m.assign(zeros).realize()           |
|    4. Execute precompiled JIT schedule with zero tracing overhead (<1 ms launch)|
+--------------------------------------------------------------------------------+
```

### Static Grouped Convolution Invariant
For 1D planar Fundamental Measure Theory convolutions, weight kernels have compact support within the molecular radius:
$$\text{supp}(w_\alpha) \subseteq [-\sigma_i/2, +\sigma_i/2]$$
Since $\sigma_{\rm eff} \le 12.0\text{ \AA}$ and $\Delta z \approx 0.1\text{ \AA}$, the physical interaction range never exceeds $K_{\rm att} = 129$ points. Padding the kernel with exact zeros outside this support has zero physical effect on the convolved density field:
$$\int (w_\alpha(z') + 0) \rho(z - z') dz' = \int w_\alpha(z') \rho(z - z') dz'$$

```python
# The Ideal: Statically Padded Grouped Convolutions
def stack_grouped_kernels(kernel_arrays: List[np.ndarray], target_k: int = 129) -> Tensor:
    stacked = np.zeros((batch_size, 1, target_k, 1), dtype=np.float32)
    for b, arr in enumerate(kernel_arrays):
        k_len = len(arr)
        if k_len > target_k:
            crop = (k_len - target_k) // 2
            arr = arr[crop : crop + target_k]
            k_len = target_k
        start = (target_k - k_len) // 2
        stacked[b, 0, start : start + k_len, 0] = arr
    return Tensor(stacked, dtype=dtypes.float32).realize()
```

### In-Place Buffer Assignment Standard
Instead of re-instantiating `BatchedTinyCDFT(...)` or `nn.optim.Adam(...)` for each batch, assign new inputs into the existing device memory locations:

```python
# The Ideal: In-Place Parameter & Moment Binding
def reset_batch(self, batch: MolecularBatch) -> None:
    arrays = self._extract_batch_arrays(batch)
    self.dz.assign(Tensor(arrays["dz_vals"]).reshape(1, self.batch_size, 1, 1)).realize()
    self.bulk_density.assign(Tensor(arrays["rho_bulk_vals"]).reshape(1, self.batch_size, 1, 1)).realize()
    self.att_kernel.assign(self._stack_grouped_kernels(arrays["raw_att_kernels"], self.STATIC_ATT_K)).realize()
    self.psi.assign(Tensor(arrays["psi_init_list"])).realize()
    self.psi.grad = None

    # Reset Adam momentum without destroying compiled buffers
    if hasattr(self.opt, "m"):
        for m in self.opt.m: m.assign(Tensor.zeros_like(m)).realize()
    if hasattr(self.opt, "v"):
        for v in self.opt.v: v.assign(Tensor.zeros_like(v)).realize()
```

---

## Anti-Patterns to Contrast Against (What Bad Code Looks Like)

- ❌ **Anti-Pattern (Dynamic Kernel Allocation)**: Sizing convolution kernel tensors dynamically per batch as `max(len(k) for k in kernels)`. This changes tensor shape on every batch and triggers Tinygrad JIT invalidation on 100% of batches.
- ❌ **Anti-Pattern (Object Re-Instantiation)**: Calling `cdft = BatchedTinyCDFT(...)` inside the batch loop, creating fresh tensors and forcing fresh JIT graph compilation every step.
- ❌ **Anti-Pattern (Re-Creating Optimizers)**: Creating a new `nn.optim.Adam(...)` per batch, which allocates fresh momentum buffers that are invisible to previous JIT schedules.
- ❌ **Anti-Pattern (Ragged Input Batches)**: Feeding raw molecule site counts into JIT evaluators without uniform static padding ($N_{\rm pad}=128$).
