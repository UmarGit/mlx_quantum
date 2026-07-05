"""Benchmark the native MLX simulator against Qiskit's EstimatorQNN.

Runs the same circuit both ways, confirms they agree to float32 precision, then
times a forward+gradient workload. Writes examples/benchmark.png.

    uv run --extra examples python examples/benchmark_vs_qiskit.py
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt
import mlx.core as mx

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN

from mlx_quantum import QuantumLayer

REPS = 1
BATCH, ITERS = 64, 40
QUBIT_SIZES = [2, 4, 6, 8]


def equivalent_qnn(num_qubits):
    qc = QuantumCircuit(num_qubits)
    x = ParameterVector("x", num_qubits)
    w = ParameterVector("w", num_qubits * REPS)
    qc.h(range(num_qubits))
    for i in range(num_qubits):
        qc.ry(x[i], i)
    for i in range(num_qubits):
        qc.ry(w[i], i)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    observables = [
        SparsePauliOp.from_list([("I" * (num_qubits - i - 1) + "Z" + "I" * i, 1.0)])
        for i in range(num_qubits)
    ]
    return EstimatorQNN(
        circuit=qc, input_params=list(x), weight_params=list(w),
        observables=observables, input_gradients=True,
    )


def time_mlx(layer, x):
    grad_fn = mx.grad(lambda _layer: mx.sum(_layer(x)))
    mx.eval(layer(x), grad_fn(layer)["weight"])  # warm up
    start = time.perf_counter()
    for _ in range(ITERS):
        mx.eval(layer(x), grad_fn(layer)["weight"])
    return time.perf_counter() - start


def time_qiskit(qnn, x, w):
    start = time.perf_counter()
    for _ in range(ITERS):
        qnn.forward(x, w)
        qnn.backward(x, w)
    return time.perf_counter() - start


def main():
    rng = np.random.default_rng(0)
    mlx_times, qiskit_times = [], []

    for n in QUBIT_SIZES:
        w = rng.uniform(-1, 1, n * REPS).astype(np.float32)
        x = rng.standard_normal((BATCH, n)).astype(np.float32)

        layer = QuantumLayer(n, REPS, initial_weights=w)
        qnn = equivalent_qnn(n)

        fwd_diff = np.abs(np.asarray(layer(mx.array(x))) - qnn.forward(x, w)).max()

        t_mlx = time_mlx(layer, mx.array(x))
        t_qk = time_qiskit(qnn, x, w)
        mlx_times.append(t_mlx)
        qiskit_times.append(t_qk)
        print(f"{n} qubits | forward diff {fwd_diff:.1e} | "
              f"MLX {t_mlx*1e3:7.1f} ms | Qiskit {t_qk:6.2f} s | speedup {t_qk/t_mlx:6.0f}x")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(QUBIT_SIZES, qiskit_times, "o-", label="Qiskit (CPU)", lw=2)
    ax.plot(QUBIT_SIZES, mlx_times, "s-", label="MLX native (Metal)", lw=2)
    ax.set(xlabel="number of qubits", ylabel=f"time for {BATCH}x{ITERS} forward+grad (s)",
           title="Quantum layer: MLX vs Qiskit")
    ax.set_yscale("log")
    ax.set_xticks(QUBIT_SIZES)
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    out = os.path.join(os.path.dirname(__file__), "benchmark.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
