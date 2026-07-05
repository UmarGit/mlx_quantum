"""Qubit-ordering convention: the MLX statevector must match Qiskit's little-endian
layout amplitude-for-amplitude. An asymmetric circuit is used so a reversal would
be caught (a symmetric one could pass either way)."""

import numpy as np
import mlx.core as mx
import pytest

from mlx_quantum import statevector as sv

pytest.importorskip("qiskit")
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def test_x_on_qubit0_sets_least_significant_bit():
    # X on qubit 0 -> basis state |...001> = index 1 in little-endian.
    state = sv.zero_state(1, 3)
    state = sv.apply_1q(state, sv.X, 0)
    amp = np.asarray(state).reshape(-1)
    assert np.argmax(np.abs(amp)) == 1


def test_full_statevector_matches_qiskit_ordering():
    n = 3
    # Distinct rotation on every qubit -> asymmetric amplitudes.
    angles = [0.5, 1.3, -0.8]

    state = sv.zero_state(1, n)
    for q, a in enumerate(angles):
        state = sv.apply_1q(state, sv.H, q)
        state = sv.apply_1q(state, sv.ry(mx.array(a)), q)
    state = sv.apply_2q(state, sv.CX, 0, 1)
    mlx_amp = np.asarray(state).reshape(-1)

    qc = QuantumCircuit(n)
    for q, a in enumerate(angles):
        qc.h(q)
        qc.ry(a, q)
    qc.cx(0, 1)
    qiskit_amp = Statevector.from_instruction(qc).data

    assert np.allclose(mlx_amp, qiskit_amp, atol=1e-6)


def test_z_expectation_is_per_qubit_not_reversed():
    # RY(pi) on qubit 0 flips it to |1>: <Z_0> = -1, others +1.
    n = 3
    state = sv.zero_state(1, n)
    state = sv.apply_1q(state, sv.ry(mx.array(np.pi)), 0)
    z = np.asarray(sv.expval_all_z(state))[0]
    assert np.allclose(z, [-1.0, 1.0, 1.0], atol=1e-5)
