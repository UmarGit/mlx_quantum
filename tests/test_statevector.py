import math

import numpy as np
import mlx.core as mx

from mlx_quantum import statevector as sv


def test_zero_state_shape_and_value():
    state = sv.zero_state(3, 2)
    assert state.shape == (3, 2, 2)
    amp = np.asarray(state).reshape(3, -1)
    assert np.allclose(amp[:, 0], 1.0) and np.allclose(amp[:, 1:], 0.0)


def test_bell_state():
    state = sv.zero_state(1, 2)
    state = sv.apply_1q(state, sv.H, 0)
    state = sv.apply_2q(state, sv.CX, 0, 1)
    amp = np.asarray(state).reshape(-1)
    assert np.allclose(amp, [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)], atol=1e-6)


def test_z_expectation():
    state = sv.zero_state(1, 2)
    assert np.allclose(np.asarray(sv.expval_all_z(state)), [[1.0, 1.0]], atol=1e-6)
    state = sv.apply_1q(state, sv.X, 0)
    assert np.allclose(np.asarray(sv.expval_all_z(state)), [[-1.0, 1.0]], atol=1e-6)


def test_ry_rotates_expectation():
    state = sv.zero_state(1, 1)
    state = sv.apply_1q(state, sv.ry(mx.array(math.pi / 2)), 0)
    assert abs(float(sv.expval_z(state, 0)[0])) < 1e-6  # RY(pi/2)|0> -> <Z> = 0
