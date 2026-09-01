# Pattern: Tinygrad Lowering Pipeline, AxisTypes, and Kernel Optimizations

## Summary
- **Problem**: Tensor operations failing to achieve near-peak GPU memory bandwidth or tensor-core utilization in `tinygrad`.
- **Root Cause**: Suboptimal kernel iteration schedules, poor workgroup/local dimension splitting, or memory strides that prevent auto-vectorization (`UPCAST`) and tensor core (`TC` / `Wmma`) emission in tinygrad's compiler.
- **Actionable Fix**: Understand the 8-stage lowering pipeline and structure reduction and matrix operations to match the compiler's range factorization and `AxisType` hierarchy.
- **Related Skills / Modules**: `cdft-wikiskill`, `dens_city.cdft.tiny_cdft`

## Deep Technical Specification (tinyspec.tex)
Tinygrad lowers high-level UOp tensor graphs to native machine instructions through an 8-stage compilation pipeline:
1. **`Callify`**: Transforms the tensor graph into a single stateless function.
2. **`Rangeify`**: Determines kernel boundaries and splits, decomposing operations to scalar shape `()`.
3. **`Optimize`**: Inserts shared/local memory buffers, splits and swaps iteration ranges, classifies execution axes, and binds tensor core primitives.
4. **`Expand`**: Expands parallel range axes into concrete grid dimensions.
5. **`Instruction Selection`**: Selects target ISA machine instructions (including `Wmma` matrix multiply-accumulate and devectorization).
6. **`Linearize`**: Topologically sorts the UOp DAG into a linear instruction sequence.
7. **`Register/Memory Plan`**: Allocates and reuses `GLOBAL`, `LOCAL`, and `REG` memory storage for values with non-overlapping lifetimes.
8. **`Render`**: Outputs human-readable target source code (Metal, OpenCL, C, CUDA) or direct binary machine code.

### Range AxisTypes in the Iteration Space:
| AxisType | Letter | Origin / Split | Semantics |
| :--- | :--- | :--- | :--- |
| `DEVICE` | `d` | Multi-device sharding | Distributed execution dimension across devices. |
| `GLOBAL` | `g` | Kernel grid | GPU global workgroup / block dimension. |
| `LOCAL` | `l` | Split from `g`, `L` (inner) | Workgroup local thread dimension (shares `LOCAL` memory). |
| `WARP` | `w` | Created by `TC` | Warp-level SIMD lanes for hardware tensor cores. |
| `THREAD` | `t` | Split from `g` (outer) | Multi-core CPU thread parallelism. |
| `LOOP` | `L` | Sequential | Generic serial loop (initial state of iterations). |
| `REDUCE` | `R` | Reduction axis | Serial accumulation loop. |
| `GROUP_REDUCE` | `G` | Split from `R` | Parallel tree reduction in shared memory. |
| `UPCAST` | `u` | Split from `g`, `l`, `L` (inner) | Register-level SIMD vectorization (e.g. `float4`). |
| `UNROLL` | `r` | Split from `R`, `G` (inner) | Fully unrolled loop in registers. |

### Schedule Transforms (`OptOps`):
- `Split(axis, factor, target)`: Splits iteration count into outer and inner axes, assigning the inner axis a specialized `AxisType` (e.g. `UPCAST` or `LOCAL`).
- `Padto(axis, multiple)`: Pads dimension to next power of 2 or multiple of 16/32 with validity masks for hardware alignment.
- `TC(reduce_idx)`: Binds warp tensor core instructions (`Wmma`).

## Verified Solution & Action Rules
1. Align reduction and inner spatial dimensions to multiples of 4, 8, or 16 so the compiler can apply `UPCAST` vectorization.
2. For convolution or FFT kernels, ensure padding is applied via `Pad` so `Padto` can vectorize boundary bins without branching.
3. Keep batch dimensions separate from spatial axes to allow parallel mapping to `GLOBAL` dimensions.

```python
# Verified pattern: structure inner dimensions for vectorization
def prepare_tensor_for_vectorized_ops(t: Tensor, alignment: int = 16) -> Tensor:
    # Ensure inner dimension is padded to alignment for efficient UPCAST
    last_dim = t.shape[-1]
    if last_dim % alignment != 0:
        pad_amount = alignment - (last_dim % alignment)
        # Pad right on last axis
        padding = [(0, 0)] * (len(t.shape) - 1) + [(0, pad_amount)]
        return t.pad(tuple(padding))
    return t
```

## Anti-Patterns to Avoid
- ❌ **Anti-Pattern**: Using prime or odd numbers for reduction dimensions without padding (forces un-vectorized single-scalar `LOOP` and disables `UPCAST`).
- ❌ **Anti-Pattern**: Excessive kernel splitting by creating artificial data dependencies between elementwise operations.
