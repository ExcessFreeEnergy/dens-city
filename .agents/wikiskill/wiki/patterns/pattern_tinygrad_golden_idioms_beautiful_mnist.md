# Pattern: Tinygrad Golden Standards & Idiomatic Training Loop Design

## Summary
- **Problem**: Inefficient, fragmented training loops in `tinygrad` characterized by repeated Python graph retracing, separate synchronous optimizer steps (`opt.step()`), CPU-GPU dataloader bottlenecks, or broken BatchNorm states in `@TinyJit`.
- **Root Cause**: Writing `tinygrad` code like PyTorch. PyTorch relies on an eager runtime where `loss.backward()` and `optimizer.step()` are executed imperatively on CPU. In `tinygrad`, the entire forward pass, backward autodiff, parameter update, and mini-batch indexing can be compiled into a single fused `@TinyJit` execution graph via `@function`, `@Context(TRAINING=1)`, and `loss.realize(*opt.schedule_step())`.
- **Actionable Fix**: Follow the canonical `beautiful_mnist.py` standard:
  1. Wrap model forward passes with the `@function` decorator for symbolic graph capture with `Param` placeholders.
  2. Decorate training steps with `@TinyJit` and `@Context(TRAINING=1)`.
  3. Fuse forward, backward, and optimizer updates in one graph execution via `loss.backward()` followed by `loss.realize(*opt.schedule_step())`.
  4. Perform mini-batch sampling on-device via `Tensor.randint(...)` (Threefry PRNG).
  5. Reset hardware counters (`GlobalCounters.reset()`) per step for clean profiling.
- **Related Skills / Modules**: `cdft-wikiskill`, `dens_city.boltzmann`, `dens_city.cdft`

## Deep Technical Architecture (`tinyspec.tex` + `beautiful_mnist.py`)

### 1. The `@function` Decorator
As specified in `tinyspec.tex`, `@function` traces a tensor function lazily without device execution, building a reusable UOp graph:
- Replaces concrete input buffers with symbolic `Param(k)` placeholders.
- Wraps the body in a `Tuple` UOp: `Function(Tuple(body), x, y)`.
- Returns elements via `GetTuple(idx)`.
- **Benefit**: The graph is parameterized and differentiable ("Gradient-able"), allowing seamless reuse without recompiling Python function calls.

```python
class MolecularModel:

    @function
    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)
```

### 2. Full-Step Fusion via `opt.schedule_step()`
In standard naive loops, developers call:
```python
# ❌ Anti-Pattern: Two separate dispatches
loss.backward()
loss.realize()
opt.step()  # Separate dispatch and synchronization
```
In the golden standard:
```python
#  Golden Pattern: Unified single-graph execution
opt.zero_grad()
loss = self(X[samples]).sparse_categorical_crossentropy(Y[samples]).backward()
return loss.realize(*opt.schedule_step())
```
`opt.schedule_step()` returns the update UOp tensors. Passing them as variable arguments to `loss.realize(*...)` causes tinygrad's compiler to schedule and fuse the backward adjoints and weight updates into the same command buffer.

### 3. Compilation State Management: `@Context(TRAINING=1)`
Layers such as `nn.BatchNorm` need to accumulate running statistics during training and freeze them during evaluation. In `@TinyJit`:
- Training function:
  ```python
  @TinyJit
  @Context(TRAINING=1)
  def train_step(self, X: Tensor, Y: Tensor) -> Tensor:
      ...
```
- Evaluation function:
  ```python
  @TinyJit
  def get_test_acc(self, X: Tensor, Y: Tensor) -> Tensor:
      return (self(X).argmax(axis=1) == Y).mean() * 100
```
Because the evaluation method lacks `TRAINING=1`, BatchNorm automatically switches to static inference without updating running statistics.

### 4. On-Device Random Sampling (Threefry PRNG)
Instead of indexing batches on CPU via Python lists or NumPy arrays and sending slices across the PCIe bus, sample indices directly on device:
```python
samples = Tensor.randint(batch_size, high=X_train.shape[0])
batch_x = X_train[samples]
```
`Tensor.randint` lowers directly to the decomposed `Threefry` 5-round ARX (add-rotate-xor) PRNG specified in `tinyspec.tex`, executing entirely in GPU registers.

## Verified Implementation Pattern
```python
# Canonical Tinygrad training harness
from tinygrad import Context, GlobalCounters, Tensor, TinyJit, function, nn
from tinygrad.helpers import getenv, trange


class EnergyPredictor:

    def __init__(self, in_features: int, hidden: int):
        self.layers = [
            nn.Linear(in_features, hidden),
            Tensor.relu,
            nn.BatchNorm(hidden),
            nn.Linear(hidden, 1),
        ]

    @function
    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)

    @TinyJit
    @Context(TRAINING=1)
    def train_step(self, X: Tensor, Y: Tensor, opt) -> Tensor:
        opt.zero_grad()
        samples = Tensor.randint(getenv("BS", 256), high=X.shape[0])
        pred = self(X[samples])
        loss = (pred - Y[samples]).square().mean().backward()
        return loss.realize(*opt.schedule_step())

    @TinyJit
    def eval_step(self, X: Tensor, Y: Tensor) -> Tensor:
        return (self(X) - Y).abs().mean()
```

## Anti-Patterns to Avoid
- ❌ **Anti-Pattern**: Omitting `@Context(TRAINING=1)` on `@TinyJit` training functions (leads to un-updated BatchNorm or erratic dropout behavior).
- ❌ **Anti-Pattern**: Separating `loss.realize()` and `opt.step()` inside a JIT function (prevents compiler kernel fusion of gradients and optimizer updates).
- ❌ **Anti-Pattern**: Using CPU PyTorch or NumPy data-loaders to slice mini-batches in every step when full datasets can fit in GPU memory.
