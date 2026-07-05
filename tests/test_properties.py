"""Property tests: unitarity of every gate and norm preservation of the state.

These catch bugs that a diff-against-Qiskit can miss (e.g. a gate that is wrong
but self-consistent between the two implementations)."""

import numpy as np
import mlx.core as mx
import pytest

from mlx_quantum import statevector as sv


def _unitary_1q(gate):
    u = np.asarray(gate)
    return np.allclose(u.conj().T @ u, np.eye(2), atol=1e-6)


def _unitary_2q(gate):
    u = np.asarray(gate).reshape(4, 4)
    return np.allclose(u.conj().T @ u, np.eye(4), atol=1e-6)


@pytest.mark.parametrize("gate", [sv.H, sv.X, sv.Y, sv.Z])
def test_fixed_single_qubit_gates_unitary(gate):
    assert _unitary_1q(gate)


@pytest.mark.parametrize("factory", [sv.rx, sv.ry, sv.rz])
@pytest.mark.parametrize("theta", [-2.3, -0.5, 0.0, 1.1, 3.0])
def test_rotation_gates_unitary(factory, theta):
    assert _unitary_1q(factory(mx.array(theta)))


@pytest.mark.parametrize("gate", [sv.CX, sv.CZ])
def test_two_qubit_gates_unitary(gate):
    assert _unitary_2q(gate)


def test_norm_preserved_through_random_circuit():
    rng = np.random.default_rng(0)
    n = 4
    state = sv.zero_state(3, n)
    state = _randomize_batch(state, n, rng)  # break the trivial |0> start

    def norm2(s):
        return np.sum(np.abs(np.asarray(s)) ** 2, axis=tuple(range(1, s.ndim)))

    assert np.allclose(norm2(state), 1.0, atol=1e-5)
    for _ in range(20):
        q = int(rng.integers(0, n))
        state = sv.apply_1q(state, sv.ry(mx.array(rng.uniform(-3, 3))), q)
        c, t = rng.choice(n, size=2, replace=False)
        state = sv.apply_2q(state, sv.CX, int(c), int(t))
        assert np.allclose(norm2(state), 1.0, atol=1e-5)


def _randomize_batch(state, n, rng):
    for q in range(n):
        state = sv.apply_1q(state, sv.H, q)
        state = sv.apply_1q(state, sv.ry(mx.array(rng.uniform(-3, 3))), q)
    return state
