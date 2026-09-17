"""Package a verified model and offline demo, with checksums and provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile

import numpy as np

from engine.model_format import write_model
from .model import CNN
from .verify import ROOT, verify_bundle

BUNDLE_FILES = ("checkpoint.npz", "weights.bin", "samples.bin", "samples.npz", "reference.npz", "metrics.json")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_bundle(bundle):
    """Reject mixed checkpoints, inputs, references, and recorded metrics."""
    bundle = Path(bundle)
    metrics = json.loads((bundle / "metrics.json").read_text())
    if digest((bundle / "weights.bin").read_bytes()) != metrics["weights_sha256"]:
        raise ValueError("weights hash does not match metrics.json")
    model = CNN.load(bundle / "checkpoint.npz")
    with np.load(bundle / "samples.npz", allow_pickle=False) as samples:
        images, labels = samples["images"], samples["labels"]
    if labels.shape != (len(images),) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("invalid sample labels")
    classes = model.params["dense.bias"].size
    if not np.isfinite(images).all() or np.any((labels < 0) | (labels >= classes)):
        raise ValueError("invalid sample values or classes")
    with tempfile.TemporaryDirectory(prefix="cnn-release-validate-") as temporary:
        path = Path(temporary)
        write_model(path / "weights.bin", model.params)
        write_model(path / "samples.bin", {"input": images})
        for name in ("weights.bin", "samples.bin"):
            if (path / name).read_bytes() != (bundle / name).read_bytes():
                raise ValueError(f"checkpoint/sample data do not match {name}")
    actual = model.forward(images, trace=True)
    with np.load(bundle / "reference.npz", allow_pickle=False) as reference:
        if set(reference.files) != set(actual):
            raise ValueError("reference layer names do not match the model")
        for key, values in actual.items():
            expected = reference[key]
            if not np.isfinite(expected).all() or expected.shape != values.shape:
                raise ValueError(f"invalid reference tensor: {key}")
            np.testing.assert_allclose(values, expected, rtol=2e-4, atol=5e-5, err_msg=key)
    return metrics


def create_archive(bundle, destination, version, provenance, verification):
    """Use an explicit file list so datasets, local paths, and credentials stay out."""
    if not re.fullmatch(r"v\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?", version):
        raise ValueError("version must look like v0.1.0")
    bundle, destination = Path(bundle), Path(destination)
    metrics = validate_bundle(bundle)
    payload = {name: (bundle / name).read_bytes() for name in (*BUNDLE_FILES, "demo.html")}
    payload["LICENSE"] = (ROOT / "LICENSE").read_bytes()
    payload["DATA_SOURCES.md"] = (ROOT / "DATA_SOURCES.md").read_bytes()
    evaluation = bundle / "evaluation/evaluation.json"
    if evaluation.exists():
        summary = json.loads(evaluation.read_text())
        if summary["weights_sha256"] != metrics["weights_sha256"] or summary["checkpoint_sha256"] != digest(payload["checkpoint.npz"]):
            raise ValueError("evaluation belongs to a different checkpoint")
        for name in ("evaluation.json", "training-curves.png", "confusion-matrix.png", "predictions.png"):
            payload["evaluation/" + name] = (bundle / "evaluation" / name).read_bytes()
    payload["README.md"] = f"""# CNN from scratch {version}: model and demo

Open demo.html directly in a browser. It works offline and displays precomputed
NumPy/C/C++ predictions. The model's measured test accuracy is {metrics.get('test_accuracy', 0):.2%}.

To verify native inference, download the matching source release from:
https://github.com/osama-z/cnn-from-scratch/releases/tag/{version}

From the source repository root, install requirements.txt, then extract this
archive into build/trained/ and run:

    make all
    make verify-trained
    make demo

Python 3.10+, NumPy, GCC/G++, and Make are required for native verification.
No dataset download or retraining is needed for these commands.

MANIFEST.json records per-file SHA-256 checksums and source revision information.
DATA_SOURCES.md credits the external MNIST examples. LICENSE covers project code.
The full MNIST archives and platform-specific executables are not bundled.
""".encode()
    manifest = {"version": version, "source": provenance,
                "weights_sha256": metrics["weights_sha256"], "verification": verification,
                "files": {name: {"sha256": digest(raw), "bytes": len(raw)} for name, raw in sorted(payload.items())}}
    payload["MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"cnn-from-scratch-{version}-model.zip"
    checksum = destination / (archive.name + ".sha256")
    if archive.exists() or checksum.exists():
        raise FileExistsError("release archive already exists; choose a fresh --dist directory")
    # Stable ZIP timestamps and permissions keep identical inputs reproducible.
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as output:
        for name, raw in sorted(payload.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            output.writestr(info, raw)
    checksum.write_text(f"{digest(archive.read_bytes())}  {archive.name}\n")
    print(f"Release archive: {archive}")
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "build/trained")
    parser.add_argument("--dist", type=Path, default=ROOT / "build/release")
    parser.add_argument("--version", default="v0.1.0")
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if dirty:
        parser.error("commit the source changes first so the archive has an exact reproducible source revision")
    validate_bundle(args.out)
    verification = verify_bundle(args.out)
    from .demo import generate_demo
    generate_demo(args.out)
    create_archive(args.out, args.dist, args.version, {"git_commit": revision}, verification)


if __name__ == "__main__":
    main()
