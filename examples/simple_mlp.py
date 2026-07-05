"""Train a hybrid classical-quantum MLP entirely in MLX (GPU, native autodiff).

    Linear -> tanh -> [QuantumLayer] -> Linear

    uv run python examples/simple_mlp.py
"""

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx_quantum import QuantumLayer

NUM_QUBITS = 16


class HybridMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.pre = nn.Linear(in_dim, NUM_QUBITS)
        self.qnn = QuantumLayer(NUM_QUBITS, reps=2)
        self.post = nn.Linear(NUM_QUBITS, out_dim)

    def __call__(self, x):
        x = mx.tanh(self.pre(x)) * mx.pi  # encode into rotation angles
        return self.post(self.qnn(x))


def main():
    mx.random.seed(0)
    np.random.seed(0)

    n, in_dim, out_dim = 128, 8, 3
    x = mx.array(np.random.randn(n, in_dim).astype(np.float32))
    y = mx.array(np.random.randint(0, out_dim, n))

    model = HybridMLP(in_dim, out_dim)
    optimizer = optim.Adam(learning_rate=0.05)
    loss_and_grad = nn.value_and_grad(model, lambda m, x, y: mx.mean(nn.losses.cross_entropy(m(x), y)))

    w0 = np.asarray(model.qnn.weight).copy()
    for epoch in range(20):
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        print(f"epoch {epoch + 1:2d} | loss {loss.item():.4f}")

    print(f"\nquantum weights moved by max|delta| = {np.abs(np.asarray(model.qnn.weight) - w0).max():.4f}")


if __name__ == "__main__":
    main()
