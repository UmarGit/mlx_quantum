import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from mlx_quantum import QuantumLayer


def test_forward_shapes():
    layer = QuantumLayer(num_qubits=3)
    assert layer(mx.zeros(3)).shape == (3,)
    assert layer(mx.zeros((5, 3))).shape == (5, 3)


def test_output_in_range():
    layer = QuantumLayer(num_qubits=4, reps=2)
    out = np.asarray(layer(mx.array(np.random.uniform(-3, 3, (8, 4)).astype(np.float32))))
    assert out.min() >= -1 - 1e-5 and out.max() <= 1 + 1e-5


def test_weight_is_trainable():
    layer = QuantumLayer(num_qubits=3, reps=2)
    params = dict(tree_flatten(layer.trainable_parameters()))
    assert "weight" in params
    assert params["weight"].shape == (6,)


def test_gradient_matches_finite_difference():
    np.random.seed(0)
    layer = QuantumLayer(num_qubits=3, reps=1)
    x = mx.array(np.random.uniform(-2, 2, (4, 3)).astype(np.float32))

    d_w = np.asarray(mx.grad(lambda l: mx.sum(l(x)))(layer)["weight"], dtype=np.float64)

    w0 = np.asarray(layer.weight, dtype=np.float64)
    eps, fd = 1e-3, np.zeros_like(w0)
    for i in range(len(w0)):
        for sign in (+1, -1):
            wp = w0.copy()
            wp[i] += sign * eps
            layer.weight = mx.array(wp.astype(np.float32))
            fd[i] += sign * float(mx.sum(layer(x)))
        fd[i] /= 2 * eps

    assert np.abs(fd).max() > 1e-6
    np.testing.assert_allclose(d_w, fd, atol=1e-3)


def test_training_moves_weights_and_reduces_loss():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre = nn.Linear(6, 4)
            self.qnn = QuantumLayer(num_qubits=4)
            self.post = nn.Linear(4, 2)

        def __call__(self, x):
            return self.post(self.qnn(mx.tanh(self.pre(x)) * mx.pi))

    np.random.seed(0)
    net = Net()
    x = mx.array(np.random.randn(16, 6).astype(np.float32))
    y = mx.array(np.random.randint(0, 2, 16))
    w0 = np.asarray(net.qnn.weight).copy()

    loss_and_grad = nn.value_and_grad(net, lambda m, x, y: mx.mean(nn.losses.cross_entropy(m(x), y)))
    opt = optim.Adam(learning_rate=0.1)
    first = last = None
    for _ in range(15):
        loss, grads = loss_and_grad(net, x, y)
        opt.update(net, grads)
        mx.eval(net.parameters(), opt.state)
        first = first if first is not None else loss.item()
        last = loss.item()

    assert np.abs(np.asarray(net.qnn.weight) - w0).max() > 1e-6
    assert last < first
