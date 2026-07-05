"""Batched, differentiable statevector simulation in pure MLX.

A statevector is a complex ``mx.array`` of shape ``(batch,) + (2,) * num_qubits``.
Qubit ordering is little-endian to match Qiskit: qubit 0 is the fastest-varying
index, so flattening the state reproduces Qiskit's amplitude order exactly. Gates
are applied as ``einsum`` contractions, so the whole simulation runs on the Metal
GPU and is differentiable end to end via ``mx.grad`` — no custom vjp.
"""

import math
import string

import numpy as np
import mlx.core as mx

_AXES = string.ascii_lowercase


def zero_state(batch: int, num_qubits: int) -> mx.array:
    """The |0...0> state, broadcast over a batch."""
    amp = np.zeros((1,) + (2,) * num_qubits, np.complex64)
    amp.flat[0] = 1.0
    return mx.array(np.broadcast_to(amp, (batch,) + (2,) * num_qubits).copy())


def _axis(qubit: int, n: int) -> int:
    """Little-endian: qubit 0 is the last axis (fastest index when flattened)."""
    return n - 1 - qubit


def apply_1q(state: mx.array, gate: mx.array, qubit: int) -> mx.array:
    """Apply a single-qubit gate. ``gate`` is ``(2, 2)`` or, per-sample, ``(batch, 2, 2)``."""
    n = state.ndim - 1
    axes = _AXES[:n]
    q = _axis(qubit, n)
    src, tgt = "B" + axes, "B" + axes[:q] + "Z" + axes[q + 1:]
    if gate.ndim == 2:
        return mx.einsum(f"{src},Z{axes[q]}->{tgt}", state, gate)
    return mx.einsum(f"{src},BZ{axes[q]}->{tgt}", state, gate)


def apply_2q(state: mx.array, gate: mx.array, q0: int, q1: int) -> mx.array:
    """Apply a two-qubit gate given as a ``(2, 2, 2, 2)`` tensor ``[out0, out1, in0, in1]``."""
    n = state.ndim - 1
    axes = _AXES[:n]
    a0, a1 = _axis(q0, n), _axis(q1, n)
    tgt = list("B" + axes)
    tgt[1 + a0], tgt[1 + a1] = "Y", "Z"
    return mx.einsum(f"B{axes},YZ{axes[a0]}{axes[a1]}->{''.join(tgt)}", state, gate)


def expval_z(state: mx.array, qubit: int) -> mx.array:
    """<Z> on one qubit, shape ``(batch,)``."""
    probs = mx.abs(state) ** 2
    keep = _axis(qubit, state.ndim - 1)
    axes = [1 + j for j in range(state.ndim - 1) if j != keep]
    marg = mx.sum(probs, axis=axes) if axes else probs
    return marg[:, 0] - marg[:, 1]


def expval_all_z(state: mx.array) -> mx.array:
    """<Z> on every qubit, shape ``(batch, num_qubits)``."""
    return mx.stack([expval_z(state, q) for q in range(state.ndim - 1)], axis=1)


def _const(matrix) -> mx.array:
    return mx.array(np.array(matrix, np.complex64))


H = _const([[1, 1], [1, -1]]) / math.sqrt(2)
X = _const([[0, 1], [1, 0]])
Y = _const([[0, -1j], [1j, 0]])
Z = _const([[1, 0], [0, -1]])


def _two_qubit(fn) -> mx.array:
    g = np.zeros((2, 2, 2, 2), np.complex64)
    for c in (0, 1):
        for t in (0, 1):
            oc, ot, val = fn(c, t)
            g[oc, ot, c, t] = val
    return mx.array(g)


CX = _two_qubit(lambda c, t: (c, t ^ c, 1.0))
CZ = _two_qubit(lambda c, t: (c, t, -1.0 if c and t else 1.0))


def rx(theta: mx.array) -> mx.array:
    c, s = mx.cos(theta / 2), mx.sin(theta / 2)
    return _rot(c, -1j * s, -1j * s, c)


def ry(theta: mx.array) -> mx.array:
    c, s = mx.cos(theta / 2), mx.sin(theta / 2)
    return _rot(c, -s, s, c)


def rz(theta: mx.array) -> mx.array:
    p = _expi(theta / 2)
    zero = mx.zeros_like(theta).astype(mx.complex64)
    return _rot(mx.conj(p), zero, zero, p)


def _expi(phase: mx.array) -> mx.array:
    return (mx.cos(phase) + 1j * mx.sin(phase)).astype(mx.complex64)


def _rot(a, b, c, d) -> mx.array:
    """Stack four (scalar or batched) entries into a ``(*shape, 2, 2)`` gate."""
    a, b, c, d = (mx.array(v).astype(mx.complex64) for v in (a, b, c, d))
    row0 = mx.stack([a, b], axis=-1)
    row1 = mx.stack([c, d], axis=-1)
    return mx.stack([row0, row1], axis=-2)
