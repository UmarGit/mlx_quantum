"""Shared circuit-spec engine for parity checks.

A *spec* is a list of gate ops runnable both on the MLX simulator and as a Qiskit
circuit, so forward values and gradients can be compared on identical circuits
covering every gate the library exposes. Single source of truth for the
correctness claims (used by the tests and the benchmark script).
"""

import numpy as np
import mlx.core as mx

from mlx_quantum import statevector as sv

SINGLE_FIXED = {"h": sv.H, "x": sv.X, "y": sv.Y, "z": sv.Z}
SINGLE_ROT = {"rx": sv.rx, "ry": sv.ry, "rz": sv.rz}
TWO_FIXED = {"cx": sv.CX, "cz": sv.CZ}


def random_spec(num_qubits, depth, rng):
    """Random circuit touching every gate type; returns (ops, num_params)."""
    ops, n_params = [], 0
    singles = list(SINGLE_FIXED) + list(SINGLE_ROT)
    for _ in range(depth):
        for q in range(num_qubits):
            g = rng.choice(singles)
            if g in SINGLE_ROT:
                ops.append((g, q, n_params))
                n_params += 1
            else:
                ops.append((g, q, None))
        for q in range(num_qubits - 1):
            ops.append(("cx" if rng.random() < 0.5 else "cz", q, q + 1))
    return ops, n_params


def run_mlx(ops, num_qubits, params):
    """Evolve |0...0> under the spec; returns the statevector, shape (1,)+(2,)*n."""
    state = sv.zero_state(1, num_qubits)
    for name, a, b in ops:
        if name in SINGLE_FIXED:
            state = sv.apply_1q(state, SINGLE_FIXED[name], a)
        elif name in SINGLE_ROT:
            state = sv.apply_1q(state, SINGLE_ROT[name](params[b]), a)
        else:
            state = sv.apply_2q(state, TWO_FIXED[name], a, b)
    return state


def mlx_expectation(ops, num_qubits, params):
    """Sum of <Z_q> over qubits, as an MLX scalar (differentiable in ``params``)."""
    return mx.sum(sv.expval_all_z(run_mlx(ops, num_qubits, params)))


def build_qiskit(ops, num_qubits, num_params):
    """The same spec as a parameterized Qiskit circuit plus the sum-of-Z observable."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import SparsePauliOp

    qc = QuantumCircuit(num_qubits)
    p = ParameterVector("p", num_params)
    for name, a, b in ops:
        if name in SINGLE_FIXED:
            getattr(qc, name)(a)
        elif name in SINGLE_ROT:
            getattr(qc, name)(p[b], a)
        else:
            getattr(qc, name)(a, b)
    obs = SparsePauliOp.from_list(
        [("I" * (num_qubits - q - 1) + "Z" + "I" * q, 1.0) for q in range(num_qubits)]
    )
    return qc, list(p), obs


def qiskit_forward(qc, obs, values):
    from qiskit.quantum_info import Statevector

    bound = qc.assign_parameters(values) if len(values) else qc
    return float(np.real(Statevector.from_instruction(bound).expectation_value(obs)))


def qiskit_gradient(qc, obs, values):
    from qiskit_algorithms.gradients import ReverseEstimatorGradient

    result = ReverseEstimatorGradient().run([qc], [obs], [np.asarray(values)]).result()
    return np.asarray(result.gradients[0])
