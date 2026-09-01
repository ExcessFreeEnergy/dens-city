# Pattern: Tinygrad Decomposed Tensor Idioms & Multi-Device Collectives

## Summary
- **Problem**: Writing custom GPU kernels or falling back to Python CPU loops for operations like `arange`, `gather`, `scatter_add`, `prefix_sum`, or multi-device sharding.
- **Root Cause**: Unawareness of tinygrad's fundamental design: all complex operations decompose into pure movement ops, elementwise broadcasting, and standard `sum`/`max` reductions without requiring custom device kernels.
- **Actionable Fix**: Use tinygrad's canonical decomposed primitives to express complex indexing, cumulative sums, and inter-device collectives entirely in native graph operations.
- **Related Skills / Modules**: `cdft-wikiskill`, `dens_city.cdft`, `dens_city.boltzmann`

## Deep Technical Specification (tinyspec.tex)

### 1. General Matrix Multiply (`gemm`)
Decomposes into two broadcast reshapes, elementwise multiplication, and a single reduction:
\[C[M, N] = A[M, K] \cdot B[K, N] \iff (A.\mathrm{reshape}(M, K, 1) \times B.\mathrm{reshape}(1, K, N)).\mathrm{sum}(1)\]

### 2. Cumulative Prefix Sum (`prefix_sum`) via Sliding Window
Constructs a cumulative sum without sequential dependencies using the tile-and-shrink window idiom:
```python
def prefix_sum(T: Tensor) -> Tensor:
    n = T.shape[0]
    x = T.pad(((n - 1, 0),))  # (2n - 1,)
    x = x.reshape(1, 2 * n - 1).expand(n + 1, 2 * n - 1)  # Tile
    x = x.reshape((n + 1) * (2 * n - 1)).shrink(((0, 2 * n * n),))
    x = x.reshape(n, 2 * n).shrink(((0, n), (0, n)))  # Sliding windows
    return x.sum(-1)  # Parallel reduction
```

### 3. Range Construction (`arange`)
Derived strictly from prefix sum over constant ones:
```python
def arange(n: int) -> Tensor:
    return prefix_sum(Tensor(1).reshape(1).expand(n)) - 1
```

### 4. Indexed `gather` and `scatter_add`
Evaluated via one-hot indicator comparison and reduction:
```python
def gather(T: Tensor, idx: Tensor) -> Tensor:
    # out[i] = T[idx[i]]
    K = T.shape[0]
    pos = arange(K).reshape(K, 1)
    mask = (pos == idx.reshape(1, -1)).cast(T.dtype)
    return (T.reshape(K, 1) * mask).sum(0)


def scatter_add(T: Tensor, idx: Tensor, val: Tensor) -> Tensor:
    # T[idx[i]] += val[i]
    K, D = T.shape[0], idx.shape[0]
    pos = arange(K).reshape(K, 1)
    mask = (pos == idx.reshape(1, D)).cast(T.dtype)
    return T + (mask * val.reshape(1, D)).sum(1)
```

### 5. Multi-Device Collectives
For an \(n\)-tuple device \(D = (d_0, \dots, d_{n-1})\):
- **`broadcast`**: `T.reshape(1, s).expand(n, s).copy(D).replicated(0)`
- **`scatter`**: `T.copy(D)` (shards axis 0 across devices)
- **`gather`**: `T.copy(D[0])` (collects shards onto master device)
- **`allgather`**: `T.reshape(1, n*s).expand(n, n*s).copy(D).replicated(0)`
- **`reduce_scatter`**: `T.reshape(n, n, s//n).permute(1, 0, 2).copy(D).sum(1).reshape(s)`
- **`allreduce`**: `allgather(reduce_scatter(T))`

## Verified Implementation Pattern
In `dens-city` planar cDFT, convolution kernels with spatially dependent coordinates use decomposed indexing to avoid GPU thread divergencies:

```python
def batched_gather_density(rho: Tensor, sample_indices: Tensor) -> Tensor:
    # rho: (Batch, Nz), sample_indices: (Batch, K)
    Nz = rho.shape[1]
    pos = Tensor.arange(Nz).reshape(1, Nz, 1)
    idx = sample_indices.unsqueeze(1)
    mask = (pos == idx).cast(rho.dtype)
    return (rho.unsqueeze(-1) * mask).sum(1)
```

## Anti-Patterns to Avoid
- ❌ **Anti-Pattern**: Using Python `for i in range(len(idx)): T[idx[i]] = ...` (forces synchronous kernel launch per element).
- ❌ **Anti-Pattern**: Incurring host-device memory roundtrips to compute index mappings on CPU numpy.
