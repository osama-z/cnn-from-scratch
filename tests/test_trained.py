import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np

from trained.model import CNN, Adam, conv_forward, conv_backward, pool_forward, pool_backward
from trained.export import export_bundle
from trained.verify import ROOT, parse_trace, verify_bundle
from engine.model_format import write_model


def patterns():
    rng = np.random.default_rng(7)
    labels = np.repeat(np.arange(3), 12)
    x = rng.normal(0, 0.03, (len(labels), 1, 8, 8)).astype(np.float32)
    x[labels == 0, 0, 2:4, :] += 1
    x[labels == 1, 0, :, 2:4] += 1
    x[labels == 2, 0, 2:6, 2:6] += 1
    return x, labels


class Gradients(unittest.TestCase):
    def test_conv_weights_bias_and_input(self):
        rng = np.random.default_rng(5)
        # Non-square inputs and multiple input/output channels exercise indexing.
        x = rng.normal(size=(2, 2, 4, 5))
        w = rng.normal(size=(3, 2, 3, 3))
        b = rng.normal(size=3)
        output, cache = conv_forward(x, w, b)
        upstream = rng.normal(size=output.shape)
        dx, dw, db = conv_backward(upstream, cache)
        eps = 1e-5
        for array, grad in ((x, dx), (w, dw), (b, db)):
            for _ in range(8):
                direction = rng.normal(size=array.shape)
                original = array.copy()
                array[...] = original + eps * direction
                plus = np.sum(conv_forward(x, w, b)[0] * upstream)
                array[...] = original - eps * direction
                minus = np.sum(conv_forward(x, w, b)[0] * upstream)
                array[...] = original
                np.testing.assert_allclose((plus-minus)/(2*eps), np.sum(grad * direction), rtol=1e-6, atol=1e-7)

    def test_entire_network_gradients(self):
        rng = np.random.default_rng(8)
        net = CNN(4, 6, filters=2, classes=3, dtype=np.float64)
        x = rng.normal(size=(2, 1, 4, 6))
        labels = np.array([0, 2])
        _, grads, dx = net.loss_and_grads(x, labels)
        eps = 1e-6
        for array, grad in [(x, dx), *[(net.params[k], grads[k]) for k in grads]]:
            for index in list(np.ndindex(array.shape))[::max(1, array.size // 12)]:
                original = array[index]
                array[index] = original + eps
                plus = net.loss_and_grads(x, labels)[0]
                array[index] = original - eps
                minus = net.loss_and_grads(x, labels)[0]
                array[index] = original
                np.testing.assert_allclose((plus-minus)/(2*eps), grad[index], rtol=2e-4, atol=2e-6)

    def test_pool_ties_and_discarded_border(self):
        x = np.ones((1, 1, 3, 3))
        out, cache = pool_forward(x)
        dx = pool_backward(np.ones_like(out), cache)
        expected = np.zeros_like(x)
        expected[0, 0, 0, 0] = 1
        np.testing.assert_array_equal(dx, expected)


class Deployment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.labels = patterns()
        cls.model = CNN(8, 8, filters=3, classes=3, seed=12)
        optimizer = Adam(cls.model.params, lr=0.02)
        cls.initial = cls.model.loss_and_grads(cls.x, cls.labels)[0]
        for _ in range(60):
            _, grads, _ = cls.model.loss_and_grads(cls.x, cls.labels)
            optimizer.update(cls.model.params, grads)
        suffix = "_asan" if os.environ.get("CNN_TEST_SANITIZERS") else ""
        cls.binaries = [ROOT / "trained" / (name + suffix) for name in ("infer_c", "infer_cpp")]

    def test_learns_and_export_matches_both_engines(self):
        final = self.model.loss_and_grads(self.x, self.labels)[0]
        self.assertLess(final, self.initial * 0.05)
        self.assertGreaterEqual(np.mean(self.model.forward(self.x).argmax(axis=1) == self.labels), 0.99)
        with tempfile.TemporaryDirectory(prefix="cnn-trained-test-") as directory:
            chosen = np.array([0, 1, 12, 13, 24, 25])
            export_bundle(self.model, self.x[chosen], self.labels[chosen], directory)
            verify_bundle(directory, self.binaries)
            loaded = CNN.load(Path(directory) / "checkpoint.npz")
            np.testing.assert_array_equal(loaded.forward(self.x), self.model.forward(self.x))

    def test_non_square_input_and_random_weights(self):
        rng = np.random.default_rng(11)
        net = CNN(6, 10, filters=2, classes=4)
        x = rng.normal(size=(3, 1, 6, 10)).astype(np.float32)
        with tempfile.TemporaryDirectory(prefix="cnn-shape-test-") as directory:
            export_bundle(net, x, np.array([0, 1, 2]), directory)
            verify_bundle(directory, self.binaries)

    def test_bad_shapes_and_nonfinite_weights_rejected(self):
        with tempfile.TemporaryDirectory(prefix="cnn-invalid-test-") as directory:
            path = Path(directory)
            export_bundle(self.model, self.x[:2], self.labels[:2], path)
            cases = []
            even = dict(self.model.params)
            even["conv.weight"] = even["conv.weight"][:, :, :2, :2]
            cases.append(even)
            transposed = dict(self.model.params)
            transposed["dense.weight"] = transposed["dense.weight"].T.copy()
            cases.append(transposed)
            nonfinite = {k: v.copy() for k, v in self.model.params.items()}
            nonfinite["conv.bias"][0] = np.nan
            cases.append(nonfinite)
            for weights in cases:
                write_model(path / "weights.bin", weights)
                for binary in self.binaries:
                    result = subprocess.run([str(binary), str(path / "weights.bin"), str(path / "samples.bin")], capture_output=True)
                    self.assertEqual(result.returncode, 1, result.stderr)

    def test_trace_rejects_nan_and_duplicates(self):
        for text in ("TENSOR 0 conv 1\nnan\n", "TENSOR 0 conv 1\n1\nTENSOR 0 conv 1\n2\n"):
            with self.assertRaises(ValueError):
                parse_trace(text)


class FailureGates(unittest.TestCase):
    def test_synthesis_failure_propagates(self):
        # Skip analysis and replace the synthesizer with a guaranteed failure.
        with tempfile.TemporaryDirectory(prefix="cnn-synth-gate-") as directory:
            result = subprocess.run(["make", "-C", str(ROOT / "hardware/vhdl"), "-o", "analyze",
                                     "synth", "GHDL=false", f"WORKDIR={directory}"], capture_output=True)
            self.assertNotEqual(result.returncode, 0)

    def test_corruption_checker_rejects_always_successful_program(self):
        from lessons.day7.test_robust import run_checks
        with self.assertRaises(AssertionError):
            run_checks([Path("/usr/bin/true")])


if __name__ == "__main__":
    unittest.main()
