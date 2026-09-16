import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from trained.evaluate import classification_report
from trained.export import export_bundle
from trained.model import CNN
from trained.release import create_archive, validate_bundle


class Evaluation(unittest.TestCase):
    def test_confusion_orientation_and_absent_class(self):
        report = classification_report(np.array([0, 0, 1]), np.array([0, 1, 1]), classes=3)
        self.assertEqual(report["confusion_matrix"], [[1, 1, 0], [0, 1, 0], [0, 0, 0]])
        self.assertAlmostEqual(report["accuracy"], 2 / 3)
        self.assertEqual(report["per_class"][0]["recall"], 0.5)
        self.assertEqual(report["per_class"][2]["precision"], 0)

    def test_invalid_labels_rejected(self):
        with self.assertRaises(ValueError):
            classification_report(np.array([0]), np.array([-1]))


class Release(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cnn-release-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle = self.root / "model"
        self.model = CNN(4, 4, filters=2, classes=3, seed=3)
        self.images = np.arange(32, dtype=np.float32).reshape(2, 1, 4, 4) / 32
        export_bundle(self.model, self.images, np.array([0, 2]), self.bundle)
        weights = (self.bundle / "weights.bin").read_bytes()
        (self.bundle / "metrics.json").write_text(json.dumps({"weights_sha256": hashlib.sha256(weights).hexdigest()}))
        (self.bundle / "demo.html").write_text("<!doctype html><title>Fixture</title>")

    def test_mixed_checkpoint_rejected(self):
        self.model.params["conv.bias"][0] += 1
        self.model.save(self.bundle / "checkpoint.npz")
        with self.assertRaisesRegex(ValueError, "weights.bin"):
            validate_bundle(self.bundle)

    def test_mixed_samples_rejected(self):
        np.savez(self.bundle / "samples.npz", images=self.images + 1, labels=np.array([0, 2]))
        with self.assertRaisesRegex(ValueError, "samples.bin"):
            validate_bundle(self.bundle)

    def test_archive_hashes_reproducibility_and_file_allowlist(self):
        (self.bundle / "unrelated-private-note.txt").write_text("must not be packaged")
        first = create_archive(self.bundle, self.root / "a", "v0.1.0", {"git_commit": "fixture"}, {})
        second = create_archive(self.bundle, self.root / "b", "v0.1.0", {"git_commit": "fixture"}, {})
        self.assertEqual(first.read_bytes(), second.read_bytes())
        with zipfile.ZipFile(first) as archive:
            manifest = json.loads(archive.read("MANIFEST.json"))
            self.assertNotIn("unrelated-private-note.txt", archive.namelist())
            self.assertEqual(set(archive.namelist()), set(manifest["files"]) | {"MANIFEST.json"})
            for name, expected in manifest["files"].items():
                data = archive.read(name)
                self.assertEqual(hashlib.sha256(data).hexdigest(), expected["sha256"])
                self.assertEqual(len(data), expected["bytes"])
        with self.assertRaises(FileExistsError):
            create_archive(self.bundle, self.root / "a", "v0.1.0", {}, {})


if __name__ == "__main__":
    unittest.main()
