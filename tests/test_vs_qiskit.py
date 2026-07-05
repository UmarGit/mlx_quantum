"""Cross-validate the native MLX simulator against Qiskit (skipped if not installed)."""

import numpy as np
import mlx.core as mx
import pytest

from mlx_quantum import QuantumLayer

qiskit = pytest.importorskip("qiskit")
pytest.importorskip("qiskit_machine_learning")

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN


def equivalent_qnn(num_qubits, reps):
    qc = QuantumCircuit(num_qubits)
    x = ParameterVector("x", num_qubits)
    w = ParameterVector("w", num_qubits * reps)
    qc.h(range(num_qubits))
    for i in range(num_qubits):
        qc.ry(x[i], i)
    k = 0
    for _ in range(reps):
        for i in range(num_qubits):
            qc.ry(w[k], i)
            k += 1
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
    observables = [
        SparsePauliOp.from_list([("I" * (num_qubits - i - 1) + "Z" + "I" * i, 1.0)])
        for i in range(num_qubits)
    ]
    return EstimatorQNN(
        circuit=qc,
        input_params=list(x),
        weight_params=list(w),
        observables=observables,
        input_gradients=True,
    )


@pytest.mark.parametrize("num_qubits,reps", [(3, 1), (4, 2)])
def test_forward_and_gradient_match_qiskit(num_qubits, reps):
    rng = np.random.default_rng(0)
    w0 = rng.uniform(-1, 1, num_qubits * reps).astype(np.float32)
    x = rng.uniform(-2, 2, (6, num_qubits)).astype(np.float32)

    layer = QuantumLayer(num_qubits, reps, initial_weights=w0)
    qnn = equivalent_qnn(num_qubits, reps)

    mlx_out = np.asarray(layer(mx.array(x)))
    assert np.abs(mlx_out - qnn.forward(x, w0)).max() < 1e-5

    cot = np.ones_like(mlx_out)
    _, weight_grad = qnn.backward(x, w0)
    qiskit_dw = np.einsum("bo,bow->w", cot, weight_grad)
    mlx_dw = np.asarray(mx.grad(lambda l: mx.sum(l(mx.array(x))))(layer)["weight"])
    assert np.abs(qiskit_dw - mlx_dw).max() < 1e-4
