# Validation & benchmarks

Correctness and performance evidence for `mlx-quantum`. Scope for every
measurement: **noiseless statevector, float32**.

```bash
uv run --extra examples python benchmarks/validate.py
```

Regenerates the five graphs in this directory. Requires the dev/`examples`
dependencies (Qiskit, qiskit-machine-learning, qiskit-algorithms, qiskit-aer,
matplotlib).

## What each graph shows

| File | Question it answers |
|------|---------------------|
| `accuracy_vs_qubits.png` | Forward error vs Qiskit stays at ~1e-6. The batch-summed gradient error rises to ~1e-5 by 8 qubits — pure float32 accumulation (16×8 terms summed into one number), still ≥5 significant figures; the per-circuit gradient error stays ~1e-6 (see the distribution). |
| `speedup_vs_qubits.png` | Two honest baselines: end-to-end vs `EstimatorQNN` (forward+gradient), and kernel-level vs Aer's compiled statevector (forward only). |
| `walltime_vs_qubits.png` | Absolute wall-time, extended until MLX hits the memory cliff. EstimatorQNN is forward+grad; Aer and MLX are forward only. |
| `error_distribution.png` | Error histogram over a random-circuit sweep covering every gate (up to 150 draws; parameter-free draws are skipped, leaving n≈142) — the 1e-6 is not cherry-picked. |
| `training_overlay.png` | Same circuit/init/data/optimizer trains identically via `mx.grad` and via `qnn.backward`. |

## References used

- **Forward:** `qiskit.quantum_info.Statevector`.
- **Gradients:** `qiskit_algorithms.gradients.ReverseEstimatorGradient` (adjoint /
  reverse-mode — the efficient, best-case Qiskit gradient, so the speedup claim
  is conservative).
- **End-to-end baseline:** `qiskit_machine_learning.neural_networks.EstimatorQNN`
  (`forward` + `backward` driven from Python).
- **Kernel baseline:** `qiskit_aer.primitives.EstimatorV2` (compiled statevector),
  forward only — Aer does not expose a matching gradient, so the kernel line is a
  forward-vs-forward comparison.

The gate-level parity engine (`tests/_circuits.py`) is shared with the test suite,
so the graphs and `pytest` check the same thing.
