"""Compare every exported layer against both deployed float engines."""
import argparse
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_trace(text):
    lines = iter(text.splitlines())
    result = {}
    for line in lines:
        fields = line.split()
        if len(fields) != 4 or fields[0] != "TENSOR":
            raise ValueError(f"invalid trace header: {line}")
        key = (int(fields[1]), fields[2])
        count = int(fields[3])
        if count <= 0 or key in result:
            raise ValueError("invalid size or duplicate trace tensor")
        result[key] = np.array([float(next(lines)) for _ in range(count)])
        if not np.isfinite(result[key]).all():
            raise ValueError(f"non-finite output: {key}")
    return result


def verify_bundle(out, binaries=None):
    out = Path(out).resolve()
    binaries = binaries or [ROOT / "trained/infer_c", ROOT / "trained/infer_cpp"]
    with np.load(out / "reference.npz", allow_pickle=False) as data:
        reference = {k: data[k] for k in data.files}
    with np.load(out / "samples.npz", allow_pickle=False) as data:
        labels = data["labels"]
    keys = {(i, layer) for layer, values in reference.items() for i in range(len(values))}
    predictions = reference["softmax"].argmax(axis=1)
    results = {}
    for binary in binaries:
        run = subprocess.run([str(binary), str(out / "weights.bin"), str(out / "samples.bin"), "--trace"],
                             capture_output=True, text=True, check=True)
        actual = parse_trace(run.stdout)
        if actual.keys() != keys:
            raise AssertionError(f"{binary}: missing/extra tensors")
        worst = 0.0
        for image, layer in sorted(keys):
            expected = reference[layer][image].reshape(-1)
            got = actual[(image, layer)]
            # Vectorized NumPy and scalar float32 loops use different sum orders.
            np.testing.assert_allclose(got, expected, rtol=2e-4, atol=5e-5,
                                       err_msg=f"{binary.name}: image {image}, layer {layer}")
            worst = max(worst, float(np.max(np.abs(got - expected))))
        got_labels = np.array([actual[(i, "softmax")].argmax() for i in range(len(labels))])
        np.testing.assert_array_equal(got_labels, predictions)
        results[binary.name] = {"max_abs_error": worst, "images": len(labels),
                                "accuracy": float(np.mean(got_labels == labels))}
        print(f"{binary.name}: all {len(keys)} layer tensors match; max error={worst:.3g}; "
              f"predictions agree on {len(labels)}/{len(labels)} images")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "build/trained")
    verify_bundle(parser.parse_args().out)
