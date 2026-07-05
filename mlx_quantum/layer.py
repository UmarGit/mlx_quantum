import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .statevector import zero_state, apply_1q, apply_2q, expval_all_z, H, ry, CX


class QuantumLayer(nn.Module):
    """A trainable quantum layer that behaves like any other ``mlx.nn`` module.

    Runs a hardware-efficient ansatz natively in MLX, so its weights train by
    ordinary gradient descent on the Metal GPU — no Qiskit, no custom autodiff.

        H on all qubits
        RY(input[q]) on each qubit q          (angle encoding)
        reps x [ RY(weight) on each qubit; CX ring ]

    Call with ``x`` of shape ``(num_qubits,)`` or ``(batch, num_qubits)``; returns
    <Z> per qubit, shape ``(num_qubits,)`` or ``(batch, num_qubits)``.
    """

    def __init__(self, num_qubits: int, reps: int = 1, initial_weights: np.ndarray | None = None):
        super().__init__()
        self.num_qubits = num_qubits
        self.reps = reps
        if initial_weights is None:
            initial_weights = np.random.uniform(-1, 1, reps * num_qubits)
        self.weight = mx.array(np.asarray(initial_weights, np.float32))

    def __call__(self, x: mx.array) -> mx.array:
        was_1d = x.ndim == 1
        if was_1d:
            x = x[None]

        n = self.num_qubits
        state = zero_state(x.shape[0], n)
        for q in range(n):
            state = apply_1q(state, H, q)
        for q in range(n):
            state = apply_1q(state, ry(x[:, q]), q)

        w = self.weight.reshape(self.reps, n)
        for r in range(self.reps):
            for q in range(n):
                state = apply_1q(state, ry(w[r, q]), q)
            for q in range(n - 1):
                state = apply_2q(state, CX, q, q + 1)

        out = expval_all_z(state)
        return out[0] if was_1d else out
