"""Phase 2 validation: correctness + performance, rendered as five graphs.

Scope for every measurement: noiseless statevector, float32.

Correctness reference: Qiskit Statevector (forward) and ReverseEstimatorGradient
(gradients). Performance baselines: qiskit-machine-learning's EstimatorQNN
(end-to-end, forward+backward from Python) and Aer's compiled statevector
estimator (kernel-level, forward only). MLX timings are taken after mx.eval, with
the first run discarded and the median +/- IQR reported over several runs.

    uv run --extra examples python benchmarks/validate.py

Writes benchmarks/*.png and prints a summary table.
"""

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import mlx.core as mx

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_algorithms.gradients import ReverseEstimatorGradient
from qiskit_aer.primitives import EstimatorV2 as AerEstimator

from mlx_quantum import QuantumLayer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import _circuits as C  # noqa: E402

HERE = os.path.dirname(__file__)
SCOPE = "noiseless statevector, float32"
BATCH = 16
REPS = 1
QUBITS_QK = [2, 3, 4, 5, 6, 7, 8]          # ranges that involve a Qiskit baseline
QUBITS_CLIFF = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]  # MLX-only, to the memory cliff


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #
def timed(fn, runs=7, warmup=2):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t)
    s = np.array(samples)
    return np.median(s), np.percentile(s, 25), np.percentile(s, 75)


# --------------------------------------------------------------------------- #
# the ansatz, built both ways
# --------------------------------------------------------------------------- #
def ansatz_qnn(n, reps=REPS):
    qc = QuantumCircuit(n)
    x = ParameterVector("x", n)
    w = ParameterVector("w", n * reps)
    qc.h(range(n))
    for i in range(n):
        qc.ry(x[i], i)
    k = 0
    for _ in range(reps):
        for i in range(n):
            qc.ry(w[k], i)
            k += 1
        for i in range(n - 1):
            qc.cx(i, i + 1)
    obs = [SparsePauliOp.from_list([("I" * (n - i - 1) + "Z" + "I" * i, 1.0)]) for i in range(n)]
    return EstimatorQNN(
        circuit=qc, input_params=list(x), weight_params=list(w),
        observables=obs, gradient=ReverseEstimatorGradient(), input_gradients=True,
    )


def aer_pub(n, reps=REPS):
    qc = QuantumCircuit(n)
    p = ParameterVector("p", n + n * reps)
    qc.h(range(n))
    for i in range(n):
        qc.ry(p[i], i)
    k = n
    for _ in range(reps):
        for i in range(n):
            qc.ry(p[k], i)
            k += 1
        for i in range(n - 1):
            qc.cx(i, i + 1)
    obs = SparsePauliOp.from_list([("I" * (n - i - 1) + "Z" + "I" * i, 1.0) for i in range(n)])
    return qc, obs


# --------------------------------------------------------------------------- #
# graph 1 + 2 + 3: accuracy, speedup, wall-time
# --------------------------------------------------------------------------- #
def bench_and_accuracy():
    rng = np.random.default_rng(0)
    rows = []
    for n in QUBITS_QK:
        w = rng.uniform(-1, 1, n * REPS).astype(np.float32)
        x = rng.standard_normal((BATCH, n)).astype(np.float32)
        layer = QuantumLayer(n, REPS, initial_weights=w)
        qnn = ansatz_qnn(n)
        qc, obs = aer_pub(n)
        aer = AerEstimator()
        params = np.concatenate([x, np.tile(w, (BATCH, 1))], axis=1)

        # accuracy: forward and weight-gradient vs Qiskit
        fwd_err = np.abs(np.asarray(layer(mx.array(x))) - qnn.forward(x, w)).max()
        _, wg = qnn.backward(x, w)
        qk_dw = np.einsum("bo,bow->w", np.ones((BATCH, n)), wg)
        mlx_dw = np.asarray(mx.grad(lambda l: mx.sum(l(mx.array(x))))(layer)["weight"])
        grad_err = np.abs(qk_dw - mlx_dw).max()

        # timing
        x_mx = mx.array(x)
        grad_fn = mx.grad(lambda l: mx.sum(l(x_mx)))
        t_mlx_fwd = timed(lambda: mx.eval(layer(x_mx)))
        t_mlx_grad = timed(lambda: mx.eval(grad_fn(layer)["weight"]))
        t_qnn = timed(lambda: (qnn.forward(x, w), qnn.backward(x, w)), runs=3, warmup=1)
        t_aer = timed(lambda: aer.run([(qc, obs, params)]).result(), runs=3, warmup=1)

        rows.append(dict(n=n, fwd_err=fwd_err, grad_err=grad_err,
                         mlx_fwd=t_mlx_fwd, mlx_grad=t_mlx_grad, qnn=t_qnn, aer=t_aer))
        print(f"n={n} | fwd {fwd_err:.1e} grad {grad_err:.1e} | "
              f"MLX fwd {t_mlx_fwd[0]*1e3:6.1f}ms grad {t_mlx_grad[0]*1e3:6.1f}ms | "
              f"QNN {t_qnn[0]:6.2f}s | Aer {t_aer[0]*1e3:6.1f}ms")
    return rows


def graph_accuracy(rows):
    n = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(n, [r["fwd_err"] for r in rows], "o-", label="forward error")
    ax.plot(n, [r["grad_err"] for r in rows], "s-", label="gradient error")
    ax.axhline(1e-6, ls=":", color="gray", label="1e-6 reference")
    ax.set(xlabel="qubits", ylabel="max abs error vs Qiskit",
           title=f"Accuracy vs Qiskit\n({SCOPE})")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    _save(fig, "accuracy_vs_qubits.png")


def graph_speedup(rows):
    n = [r["n"] for r in rows]
    end_to_end = [r["qnn"][0] / r["mlx_grad"][0] for r in rows]
    kernel = [r["aer"][0] / r["mlx_fwd"][0] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(n, end_to_end, "o-", label="vs EstimatorQNN (end-to-end, fwd+grad)")
    ax.plot(n, kernel, "s-", label="vs Aer statevector (kernel, fwd only)")
    ax.axhline(1.0, ls=":", color="gray", label="parity")
    ax.set(xlabel="qubits", ylabel="MLX speedup (x)", title=f"Speedup\n({SCOPE})")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    _save(fig, "speedup_vs_qubits.png")


def graph_walltime(rows):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    n_qk = [r["n"] for r in rows]
    ax.plot(n_qk, [r["qnn"][0] * 1e3 for r in rows], "o-", label="EstimatorQNN fwd+grad")
    ax.plot(n_qk, [r["aer"][0] * 1e3 for r in rows], "^-", label="Aer statevector fwd")

    cliff = cliff_walltime()
    ns = [c[0] for c in cliff]
    med = [c[1][0] * 1e3 for c in cliff]
    lo = [c[1][1] * 1e3 for c in cliff]
    hi = [c[1][2] * 1e3 for c in cliff]
    ax.plot(ns, med, "s-", label="MLX fwd (batch=1)")
    ax.fill_between(ns, lo, hi, alpha=0.2)

    ax.set(xlabel="qubits", ylabel="time (ms)", title=f"Wall-time to the memory cliff\n({SCOPE})")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    _save(fig, "walltime_vs_qubits.png")


def cliff_walltime():
    out = []
    for n in QUBITS_CLIFF:
        try:
            x = mx.array(np.random.standard_normal((1, n)).astype(np.float32))
            layer = QuantumLayer(n, REPS)
            stats = timed(lambda: mx.eval(layer(x)), runs=3, warmup=1)
        except Exception as e:  # memory cliff
            print(f"  cliff stopped at {n} qubits: {type(e).__name__}")
            break
        out.append((n, stats))
        print(f"  MLX forward n={n}: {stats[0]*1e3:.1f} ms")
        if stats[0] > 8.0:  # past this the trend is clear; don't hammer
            break
    return out


# --------------------------------------------------------------------------- #
# graph 4: error distribution over a random sweep
# --------------------------------------------------------------------------- #
def graph_error_distribution():
    fwd_errs, grad_errs = [], []
    for seed in range(150):
        rng = np.random.default_rng(5000 + seed)
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
        fwd_errs.append(abs(f_mlx - C.qiskit_forward(qc, obs, vals)))
        grad_errs.append(np.abs(g_mlx - C.qiskit_gradient(qc, obs, vals)).max())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.logspace(-9, -4, 30)
    ax.hist(fwd_errs, bins=bins, alpha=0.6, label=f"forward (n={len(fwd_errs)})")
    ax.hist(grad_errs, bins=bins, alpha=0.6, label=f"gradient (n={len(grad_errs)})")
    ax.axvline(1e-6, ls=":", color="gray", label="1e-6")
    ax.set(xlabel="max abs error vs Qiskit", ylabel="count",
           title=f"Error distribution over random circuits\n(all gates; {SCOPE})")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    _save(fig, "error_distribution.png")
    print(f"error distribution: forward max {max(fwd_errs):.1e}, gradient max {max(grad_errs):.1e}")


# --------------------------------------------------------------------------- #
# graph 5: training-curve overlay
# --------------------------------------------------------------------------- #
def graph_training_overlay(n=4, epochs=40, lr=2.0):
    # Plain SGD on both sides: with identical gradients the trajectories must
    # coincide, so any drift is purely the float32 gradient difference (~1e-6).
    rng = np.random.default_rng(0)
    w0 = rng.uniform(-1, 1, n).astype(np.float32)
    x = rng.standard_normal((BATCH, n)).astype(np.float32)
    target = rng.standard_normal((BATCH, n)).astype(np.float32)

    mlx_loss = _train_mlx(n, w0, x, target, epochs, lr)
    qk_loss = _train_qiskit(n, w0, x, target, epochs, lr)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ep = range(1, epochs + 1)
    ax.plot(ep, qk_loss, "o-", label="Qiskit QNN (qnn.backward + SGD)")
    ax.plot(ep, mlx_loss, "--", lw=2, label="MLX QuantumLayer (mx.grad + SGD)")
    ax.set(xlabel="epoch", ylabel="MSE loss",
           title=f"Same circuit, same init/data/optimizer\n({SCOPE})")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "training_overlay.png")
    print(f"training overlay: final loss MLX {mlx_loss[-1]:.5f}  Qiskit {qk_loss[-1]:.5f}  "
          f"max|gap| {np.abs(np.array(mlx_loss) - np.array(qk_loss)).max():.2e}")


def _train_mlx(n, w0, x, target, epochs, lr):
    import mlx.nn as nn
    import mlx.optimizers as optim

    layer = QuantumLayer(n, REPS, initial_weights=w0)
    opt = optim.SGD(learning_rate=lr)
    xm, tm = mx.array(x), mx.array(target)
    loss_and_grad = nn.value_and_grad(layer, lambda m, a, b: mx.mean((m(a) - b) ** 2))
    losses = []
    for _ in range(epochs):
        loss, grads = loss_and_grad(layer, xm, tm)
        opt.update(layer, grads)
        mx.eval(layer.parameters(), opt.state)
        losses.append(loss.item())
    return losses


def _train_qiskit(n, w0, x, target, epochs, lr):
    qnn = ansatz_qnn(n)
    w = w0.astype(np.float64).copy()
    losses = []
    for _ in range(epochs):
        out = qnn.forward(x, w)              # (B, n)
        diff = out - target
        losses.append(float(np.mean(diff ** 2)))
        cot = (2.0 / diff.size) * diff       # d loss / d out
        _, wg = qnn.backward(x, w)
        w -= lr * np.einsum("bo,bow->w", cot, wg)
    return losses


def _save(fig, name):
    fig.tight_layout()
    path = os.path.join(HERE, name)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"saved {path}")


def main():
    print("== benchmark + accuracy sweep ==")
    rows = bench_and_accuracy()
    graph_accuracy(rows)
    graph_speedup(rows)
    print("== memory-cliff wall-time ==")
    graph_walltime(rows)
    print("== error distribution ==")
    graph_error_distribution()
    print("== training overlay ==")
    graph_training_overlay()
    print("\ndone.")


if __name__ == "__main__":
    main()
