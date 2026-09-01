# Pattern: Tinygrad JIT Graph Caching & Realization Safe Practices

## Summary
- **Problem**: Tinygrad recompiles execution kernel graphs repeatedly, causing extreme slowdowns, or crashes on un-realized tensors during CPU/GPU transfers.
- **Root Cause**: Passing dynamically shaped tensors or non-contiguous slices into `@TinyJit` functions invalidates Tinygrad's compiled graph cache, forcing a recompilation on every step.
- **Actionable Fix**: Fix input tensor shapes and buffer memory strides beforehand, avoid dynamic control flow inside JIT functions, and explicitly `.realize()` outputs before accessing `.numpy()`.
- **Related Skills / Modules**: `tinygrad-jit`, `dens_city.boltzmann`, `dens_city.cdft`

## Deep Root Cause Analysis
Tinygrad's `@TinyJit` traces tensor operations and compiles them into optimized GPU/C kernels keyed by the exact tensor buffer shapes, strides, and memory layouts.
If a batch loop passes inputs of varying lengths \(N_1, N_2, \dots\), Tinygrad discards the compiled schedule and triggers a full kernel recompilation on each batch, degrading throughput by \(100\times\).
Furthermore, calling `.numpy()` or `.item()` on a lazy computation graph without calling `.realize()` forces synchronous execution and blocks asynchronous batch prefetching pipelines.

## Verified Solution & Action Rules
1. Pad variable-length molecular inputs to fixed maximum sizes \((B, N_{\max}, 3)\) using contiguous tensor masking.
2. Structure `@TinyJit` decorated functions with strictly static input tensor shapes.
3. Call `.realize()` on computed loss or potential tensors before extracting metrics.

```python
# Verified Implementation Pattern
from tinygrad import Tensor, TinyJit

@TinyJit
def compute_energy_jit(coords: Tensor, atom_types: Tensor, mask: Tensor) -> Tensor:
    # Static shapes: coords (B, N, 3), atom_types (B, N), mask (B, N)
    energy = compute_microscopic_hamiltonian(coords, atom_types, mask)
    return energy.realize()
```

## Anti-Patterns to Avoid
- ❌ **Anti-Pattern**: Passing ragged lists or changing batch sizes into `@TinyJit`.
- ❌ **Anti-Pattern**: Calling `.item()` or `.numpy()` inside inner optimization loops before `.realize()`.
- ❌ **Anti-Pattern**: Using Python `if/else` on Tensor values inside a traced computation graph (use `.where()` instead).
