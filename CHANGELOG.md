# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-07-06

### Fixed
- README images and doc links now use absolute GitHub URLs so they render on PyPI
  (relative paths only resolved on GitHub).
- `sdist` now ships an explicit file list, keeping stray working-tree files out of
  the published source distribution.

## [0.1.0] - 2026-07-05

Initial release: a native, GPU-accelerated, differentiable quantum-machine-learning
library in pure Apple MLX.

### Added
- `mlx_quantum.statevector` — batched complex statevector simulation with gates
  applied as `einsum` contractions: `zero_state`, `apply_1q`, `apply_2q`,
  `expval_z`, `expval_all_z`, and gates `H, X, Y, Z, rx, ry, rz, CX, CZ`. Qubit
  ordering is little-endian, matching Qiskit's amplitude layout exactly.
- `QuantumLayer` — a trainable `mlx.nn.Module` running a hardware-efficient
  ansatz; its weights train by ordinary MLX autodiff, no custom vjp.
- Examples: `simple_mlp.py` (hybrid classical-quantum MLP) and
  `benchmark_vs_qiskit.py` (accuracy + speed vs Qiskit's `EstimatorQNN`).
- Validation suite (`benchmarks/validate.py`) producing five scope-labeled graphs:
  accuracy vs qubits, two-baseline speedup (vs `EstimatorQNN` and vs Aer),
  wall-time to the memory cliff, error distribution over a random-circuit sweep,
  and a training-curve overlay.
- Tests covering statevector/gate correctness, gate unitarity, state-norm
  preservation, the little-endian convention, layer training, a finite-difference
  gradient check, and forward + weight-gradient + input-gradient parity with
  Qiskit across a random gate sweep (shared engine in `tests/_circuits.py`;
  Qiskit-dependent tests auto-skip when Qiskit is absent).

### Notes
- Runtime dependencies are MLX and NumPy only; Qiskit, Aer, and matplotlib are
  optional (`[examples]` extra), used for validation and benchmarking.
- Forward values and gradients match Qiskit to ~1e-6 (noiseless statevector,
  float32); ~100–400× faster end-to-end than driving `EstimatorQNN` from Python,
  and ~1.7–3× faster than Aer's compiled statevector on the forward pass.

[Unreleased]: https://github.com/UmarGit/mlx_quantum/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/UmarGit/mlx_quantum/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/UmarGit/mlx_quantum/releases/tag/v0.1.0
