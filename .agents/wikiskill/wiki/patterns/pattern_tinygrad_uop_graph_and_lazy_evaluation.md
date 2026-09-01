# Pattern: Tinygrad UOp DAG Architecture & Lazy Computation Semantics

## Summary
- **Problem**: Inefficient tensor execution, redundant kernel dispatches, or memory bandwidth bottlenecks in `tinygrad` neural networks and physics solvers.
- **Root Cause**: Misunderstanding `tinygrad`'s internal compiler representation. Unlike frameworks with multiple intermediate representations, `tinygrad` operates on a single universal dialect from high-level Python tensors down to hardware command buffers: the **UOp DAG**.
- **Actionable Fix**: Structure operations to leverage lazy movement ops (zero-arithmetic views), avoid premature materialization, and preserve shape/memory stride invariants so tinygrad's compiler can fuse elementwise and reduction kernels.
- **Related Skills / Modules**: `cdft-wikiskill`, `dens_city.cdft.tiny_cdft`, `dens_city.boltzmann`

## Deep Technical Specification (tinyspec.tex)
All nodes in tinygrad form a Directed Acyclic Graph (DAG) of **UOps**:
\[\text{UOp} = (\mathrm{op}, \; \mathrm{src}, \; \mathrm{arg}, \; \mathrm{tag})\]
Each UOp maintains 5 derived properties computed at graph construction time:
1. **`dtype`**: Data type (e.g. `dtypes.float32`, `dtypes.int32`, `dtypes.bool`).
2. **`shape`**: Tuple of dimensions.
3. **`device`**: Target hardware device string or tuple (for multi-device sharding).
4. **`addrspace`**: Memory tier (`GLOBAL`, `LOCAL` shared memory, `REG` registers).
5. **`min_max`**: Analytical interval bounds \([\min, \max]\) used by the optimizer for dead code elimination and range checks.

### UOp Categorization:
- **Source Ops (Leaves)**:
  - `Buffer`: Concrete memory slot with shape, device, and addrspace.
  - `Const`: Scalar constant with shape `()`.
  - `Param`: Symbolic placeholder with shape, substituted inside `@function` / `Function` graphs.
- **Movement Ops (Zero-Arithmetic Metadata Views)**:
  - `Permute`: Transpose / axis reordering.
  - `Reshape`: Row-major reinterpret (\(\prod s_k = \prod s'_k\)).
  - `Expand`: Broadcast / prepend axes.
  - `Pad` / `Shrink`: Offset boundary padding and slicing.
  - `Index` / `Stack` / `Bitcast`.
  - *Movement ops perform no ALU computation and allocate no memory buffers on device.*
- **ALU Elementwise Ops**:
  - Primitives: `Add`, `Mul`, `Max`, `Mod`, `Idiv`, `CmpLt`, `CmpNe`, `Where`.
  - Decomposed: `Neg` (\(A \times -1\)), `Sub` (\(A + -B\)), `Div` (\(A \times 1/B\)), `Exp2`, `Log2`, `Sin`, `Sqrt`, `Pow`.
- **Load & Store Ops (Memory Side Effects)**:
  - `Load`: Pulls from buffer into an anonymous buffer; changes device or addrspace (replaces legacy `Copy` and `Contiguous`).
  - `Store`: Only op with observable side effect; writes `val` into `buf`.
  - `After`: Passthrough of buffer guaranteeing dependency ordering (`Assign` is `Store` followed by `After`).
  - `Sink`: Root node collecting multiple side effects.

## Verified Implementation Pattern
```python
# Tinygrad idiomatic lazy pipeline pattern
from tinygrad import Tensor, dtypes


def fused_cdft_step(rho: Tensor, v_ext: Tensor, mu: float, kt: float) -> Tensor:
    # All arithmetic here is lazily fused into a single UOp graph
    # No GPU kernel is dispatched until .realize() is explicitly called
    eff_potential = v_ext - mu
    psi_target = -eff_potential / kt
    delta = psi_target - rho.log()  # or log-free parameterization

    # Fused elementwise step: compiles to single ALU kernel
    updated_psi = (delta.abs() < 1e-4).where(psi_target, psi_target + 0.1 * delta)
    return updated_psi.realize()  # Explicit compile & execute barrier
```

## Anti-Patterns to Avoid
- ❌ **Anti-Pattern**: Calling `.numpy()` or `.item()` inside training or solver loops (breaks lazy graph fusion and forces synchronous CPU-GPU stall).
- ❌ **Anti-Pattern**: Inserting manual `.contiguous()` calls between movement ops when not required by device boundary or codegen.
- ❌ **Anti-Pattern**: Creating Python control flow (`if/else`) that inspects tensor values dynamically within graph execution.
