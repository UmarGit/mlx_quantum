"""Cross-validate the native MLX simulator against Qiskit (skipped if not installed).

Scope for all comparisons: noiseless statevector, float32. Tolerances are set for
float32 accumulation (~1e-6); gradients use Qiskit's ReverseEstimatorGradient.
"""

import numpy as np
import mlx.core as mx
import pytest

from mlx_quantum import QuantumLayer

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_machine_learning")
pytest.importorskip("qiskit_algorithms")

import _circuits as C
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN

FWD_TOL = 1e-5
GRAD_TOL = 1e-4


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
        circuit=qc, input_params=list(x), weight_params=list(w),
        observables=observables, input_gradients=True,
    )


@pytest.mark.parametrize("num_qubits,reps", [(3, 1), (4, 2)])
def test_forward_weight_and_input_gradient_match_qiskit(num_qubits, reps):
    rng = np.random.default_rng(0)
    w0 = rng.uniform(-1, 1, num_qubits * reps).astype(np.float32)
    x = rng.uniform(-2, 2, (6, num_qubits)).astype(np.float32)

    layer = QuantumLayer(num_qubits, reps, initial_weights=w0)
    qnn = equivalent_qnn(num_qubits, reps)

    # forward
    assert np.abs(np.asarray(layer(mx.array(x))) - qnn.forward(x, w0)).max() < FWD_TOL

    # gradients: contract the QNN Jacobians with an all-ones cotangent
    input_grad, weight_grad = qnn.backward(x, w0)
    cot = np.ones((x.shape[0], num_qubits))
    qiskit_dw = np.einsum("bo,bow->w", cot, weight_grad)
    qiskit_dx = np.einsum("bo,boi->bi", cot, input_grad)

    mlx_dw = np.asarray(mx.grad(lambda l: mx.sum(l(mx.array(x))))(layer)["weight"])
    mlx_dx = np.asarray(mx.grad(lambda z: mx.sum(layer(z)))(mx.array(x)))

    assert np.abs(qiskit_dw - mlx_dw).max() < GRAD_TOL
    assert np.abs(qiskit_dx - mlx_dx).max() < GRAD_TOL


def test_gate_sweep_forward_and_gradient_parity():
    """Random circuits over every exposed gate; report the worst error over the sweep."""
    max_fwd = max_grad = 0.0
    checked = 0
    for seed in range(20):
        rng = np.random.default_rng(1000 + seed)
        n = int(rng.integers(2, 6))
        depth = int(rng.integers(1, 4))
        ops, num_params = C.random_spec(n, depth, rng)
        if num_params == 0:
            continue
        vals = rng.uniform(-np.pi, np.pi, num_params)
        qc, _, obs = C.build_qiskit(ops, n, num_params)

        f_mlx = float(C.mlx_expectation(ops, n, mx.array(vals.astype(np.float32))))
        g_mlx = np.asarray(
            mx.grad(lambda p: C.mlx_expectation(ops, n, p))(mx.array(vals.astype(np.float32))),
            np.float64,
        )
        max_fwd = max(max_fwd, abs(f_mlx - C.qiskit_forward(qc, obs, vals)))
        max_grad = max(max_grad, np.abs(g_mlx - C.qiskit_gradient(qc, obs, vals)).max())
        checked += 1

    assert checked >= 10
    assert max_fwd < FWD_TOL, f"max forward error {max_fwd:.2e}"
    assert max_grad < GRAD_TOL, f"max gradient error {max_grad:.2e}"
