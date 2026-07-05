# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0]

Initial release: a native, GPU-accelerated, differentiable quantum-machine-learning
library in pure Apple MLX.

### Added
- `mlx_quantum.statevector` — batched complex statevector simulation with gates
  applied as `einsum` contractions: `zero_state`, `apply_1q`, `apply_2q`,
  `expval_z`, `expval_all_z`, and gates `H, X, Y, Z, rx, ry, rz, CX, CZ`.
- `QuantumLayer` — a trainable `mlx.nn.Module` running a hardware-efficient
  ansatz; its weights train by ordinary MLX autodiff, no custom vjp.
- Examples: `simple_mlp.py` (hybrid classical-quantum MLP) and
  `benchmark_vs_qiskit.py` (accuracy + speed vs Qiskit's `EstimatorQNN`).
- Tests covering statevector correctness, layer training, a finite-difference
  gradient check, and cross-validation of forward values and gradients against
  Qiskit (auto-skipped when Qiskit is absent).

### Notes
- Runtime dependencies are MLX and NumPy only; Qiskit and matplotlib are optional
  (`[examples]` extra), used for validation and benchmarking.
- Validated identical to Qiskit to float32 precision, and ~100–1800× faster than
  driving `EstimatorQNN` from Python for 2–8 qubit circuits.

[Unreleased]: https://github.com/umarsheikh303/mlx-quantum/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/umarsheikh303/mlx-quantum/releases/tag/v0.1.0
