# mlx-quantum

[![CI](https://github.com/umarsheikh303/mlx-quantum/actions/workflows/ci.yml/badge.svg)](https://github.com/umarsheikh303/mlx-quantum/actions/workflows/ci.yml)

Fast, differentiable quantum machine learning in pure [Apple MLX](https://github.com/ml-explore/mlx).

Statevector simulation runs on the Metal GPU and is differentiable end-to-end
through MLX autodiff — so a quantum layer trains like any other `mlx.nn` module,
with no custom gradient code. Validated against Qiskit to float32 precision, and
**~100–500× faster** than driving Qiskit's `EstimatorQNN` from Python for small
circuits (see [`examples/benchmark_vs_qiskit.py`](examples/benchmark_vs_qiskit.py)).

## Install

```bash
uv add mlx-quantum
# or
pip install mlx-quantum
```

Requires Python ≥ 3.13 and Apple Silicon. The library depends only on MLX and
NumPy; Qiskit is optional and used solely to cross-validate/benchmark.

## Quickstart

`QuantumLayer` is a trainable `mlx.nn.Module`. Drop it into a model and train:

```python
import mlx.core as mx
import mlx.nn as nn
from mlx_quantum import QuantumLayer

class HybridMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Linear(8, 4)
        self.qnn = QuantumLayer(num_qubits=4, reps=2)   # trainable quantum layer
        self.post = nn.Linear(4, 3)

    def __call__(self, x):
        x = mx.tanh(self.pre(x)) * mx.pi                # encode into rotation angles
        return self.post(self.qnn(x))
```

The quantum weights are ordinary MLX parameters — `nn.value_and_grad` and any
optimizer update them automatically:

```python
loss_and_grad = nn.value_and_grad(model, loss_fn)
loss, grads = loss_and_grad(model, x, y)
optimizer.update(model, grads)
```

## Building custom circuits

`QuantumLayer` runs a hardware-efficient ansatz, but the simulator primitives are
public — build any circuit as a plain differentiable function:

```python
import mlx.core as mx
from mlx_quantum import zero_state, apply_1q, apply_2q, expval_all_z, H, ry, CX

def circuit(x, weights):            # x: (batch, n) angles, weights: (n,)
    n = x.shape[1]
    state = zero_state(x.shape[0], n)
    for q in range(n):
        state = apply_1q(state, H, q)
    for q in range(n):
        state = apply_1q(state, ry(x[:, q]), q)          # per-sample encoding
    for q in range(n):
        state = apply_1q(state, ry(weights[q]), q)        # trainable
    for q in range(n - 1):
        state = apply_2q(state, CX, q, q + 1)
    return expval_all_z(state)       # <Z> per qubit, shape (batch, n)

grads = mx.grad(lambda w: mx.sum(circuit(x, w)))(weights)  # just works
```

Gates provided: `H, X, Y, Z, rx, ry, rz, CX, CZ`. Add your own — a single-qubit
gate is any `(2, 2)` complex `mx.array`; a two-qubit gate is a `(2, 2, 2, 2)`
tensor `[out0, out1, in0, in1]`.

## How it works

A statevector is a complex `mx.array` of shape `(batch,) + (2,) * num_qubits`;
qubit `q` sits on axis `q + 1` in little-endian order (matching Qiskit). Gates are
applied as `einsum` contractions, so the entire simulation is differentiable and
GPU-resident. Because there is no custom `vjp` and no NumPy round-trip, `mx.grad`
differentiates the circuit directly — including through complex amplitudes.

Two MLX specifics the implementation works around: the initial state is built as a
constant (not an in-place assignment, which compiles to an unsupported complex
GPU scatter), and gates are contractions rather than `take`/gather (whose backward
is also a scatter).

## Examples

```bash
uv run python examples/simple_mlp.py                             # hybrid MLP training
uv run --extra examples python examples/benchmark_vs_qiskit.py   # speed + accuracy vs Qiskit
```

## Tests

```bash
uv run pytest
```

Covers gate/statevector correctness, layer training, a finite-difference gradient
check, and cross-validation of both forward values and gradients against Qiskit
(skipped automatically if Qiskit is not installed).

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

MIT — see [LICENSE](LICENSE).
