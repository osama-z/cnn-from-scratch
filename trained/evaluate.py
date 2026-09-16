"""Evaluate a saved checkpoint on MNIST and render reproducible result figures."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .model import CNN
from .train import ROOT, load_idx


def classification_report(labels, predicted, classes=10):
    labels, predicted = np.asarray(labels), np.asarray(predicted)
    if labels.ndim != 1 or labels.shape != predicted.shape or not len(labels):
        raise ValueError("nonempty paired label/prediction vectors required")
    for values in (labels, predicted):
        if not np.issubdtype(values.dtype, np.integer) or np.any((values < 0) | (values >= classes)):
            raise ValueError("class indices must be integers in range")
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (labels, predicted), 1)
    rows = []
    for digit in range(classes):
        correct = int(matrix[digit, digit])
        support, selected = int(matrix[digit].sum()), int(matrix[:, digit].sum())
        rows.append({"class": digit, "support": support,
                     "precision": correct / selected if selected else 0.0,
                     "recall": correct / support if support else 0.0})
    return {"images": len(labels), "correct": int(np.trace(matrix)),
            "accuracy": float(np.mean(labels == predicted)),
            "confusion_matrix": matrix.tolist(), "per_class": rows}


def render_figures(destination, metrics, report, images, labels, predicted):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "figure.dpi": 140})
    history = metrics["history"]
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    axes[0].plot(epochs, [row["train_loss"] for row in history], "o-", color="#2563eb")
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].plot(epochs, [100 * row["validation_accuracy"] for row in history], "o-", color="#059669")
    axes[1].axvline(metrics["best_epoch"], color="#475569", linestyle="--", label="Selected checkpoint")
    axes[1].set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    axes[1].legend(fontsize=9)
    for axis in axes:
        axis.set_xticks(epochs)
        axis.grid(alpha=0.2)
    fig.savefig(destination / "training-curves.png")
    plt.close(fig)

    matrix = np.asarray(report["confusion_matrix"])
    fig, axis = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    heat = axis.imshow(matrix, cmap="Blues")
    axis.set(xlabel="Predicted digit", ylabel="Actual digit", xticks=range(10), yticks=range(10),
             title=f"MNIST test set: {report['accuracy']:.2%} on {report['images']:,} images")
    for row in range(10):
        for column in range(10):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=8,
                      color="white" if matrix[row, column] > matrix.max() / 2 else "#0f172a")
    fig.colorbar(heat, ax=axis, label="Images")
    fig.savefig(destination / "confusion-matrix.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 6, figsize=(10, 4.3), constrained_layout=True)
    for row, (matches, title) in enumerate(((True, "Correct"), (False, "Incorrect"))):
        chosen = np.flatnonzero((labels == predicted) == matches)[:6]
        for column, axis in enumerate(axes[row]):
            axis.axis("off")
            if column < len(chosen):
                index = int(chosen[column])
                axis.imshow(images[index, 0], cmap="gray", vmin=0, vmax=1)
                axis.set_title(f"True {labels[index]} / predicted {predicted[index]}\nTest index {index}", fontsize=9)
        axes[row, 0].text(-0.15, 0.5, title, transform=axes[row, 0].transAxes,
                          rotation=90, ha="right", va="center")
    fig.suptitle("First six correct and first six incorrect test predictions")
    fig.savefig(destination / "predictions.png")
    plt.close(fig)


def evaluate(bundle, data, destination):
    bundle, data, destination = Path(bundle), Path(data), Path(destination)
    metrics = json.loads((bundle / "metrics.json").read_text())
    weights_hash = hashlib.sha256((bundle / "weights.bin").read_bytes()).hexdigest()
    if weights_hash != metrics["weights_sha256"]:
        raise ValueError("weights do not match the recorded training run")
    # Verify the checkpoint and native parameters belong to the same export.
    from .release import validate_bundle
    validate_bundle(bundle)
    for name in ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"):
        if hashlib.sha256((data / name).read_bytes()).hexdigest() != metrics["dataset_sha256"][name]:
            raise ValueError(f"test archive hash mismatch: {name}")
    images = load_idx(data / "t10k-images-idx3-ubyte.gz", True)
    labels = load_idx(data / "t10k-labels-idx1-ubyte.gz", False)
    count = metrics["test_size"]
    if len(images) != len(labels) or not 0 < count <= len(labels):
        raise ValueError("test split exceeds the paired archives")
    images, labels = images[:count], labels[:count]
    model = CNN.load(bundle / "checkpoint.npz")
    predicted = np.concatenate([model.forward(images[i:i+128]).argmax(axis=1)
                                for i in range(0, count, 128)])
    report = classification_report(labels, predicted)
    if not np.isclose(report["accuracy"], metrics["test_accuracy"], rtol=0, atol=1e-12):
        raise ValueError("saved checkpoint accuracy differs from the recorded run")
    report["weights_sha256"] = weights_hash
    report["checkpoint_sha256"] = hashlib.sha256((bundle / "checkpoint.npz").read_bytes()).hexdigest()
    report["example_selection"] = "First six correct and first six incorrect examples in test archive order"
    render_figures(destination, metrics, report, images, labels, predicted)
    (destination / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Evaluation: {report['correct']}/{count} correct ({report['accuracy']:.2%})")
    print(f"Figures and confusion matrix: {destination}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "build/trained")
    parser.add_argument("--data", type=Path, default=ROOT / "lessons/day3/mnist_data")
    parser.add_argument("--report-dir", type=Path, help="Default: OUT/evaluation")
    args = parser.parse_args()
    evaluate(args.out, args.data, args.report_dir or args.out / "evaluation")


if __name__ == "__main__":
    main()
