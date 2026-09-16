"""Train a real CNN on local MNIST IDX archives, then export the best checkpoint."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import time

# Small matrices are faster and more reproducible with one BLAS thread.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

from .model import Adam, CNN
from .export import export_bundle

ROOT = Path(__file__).resolve().parents[1]


def load_idx(path, images):
    with gzip.open(path, "rb") as stream:
        raw = stream.read()
    header_size = 16 if images else 8
    if len(raw) < header_size:
        raise ValueError(f"truncated IDX header: {path}")
    header = struct.unpack(">IIII" if images else ">II", raw[:header_size])
    if header[0] != (2051 if images else 2049) or header[1] == 0:
        raise ValueError(f"invalid IDX header: {path}")
    if images and header[2:] != (28, 28):
        raise ValueError("expected 28x28 MNIST images")
    count = header[1] * (784 if images else 1)
    data = np.frombuffer(raw, dtype=np.uint8, offset=header_size)
    if len(data) != count:
        raise ValueError(f"IDX payload size mismatch: {path}")
    if images:
        return data.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    if np.any(data > 9):
        raise ValueError("MNIST labels must be 0..9")
    return data.astype(np.int64)


def accuracy(model, x, labels, batch=128):
    correct = 0
    for start in range(0, len(x), batch):
        predicted = model.forward(x[start:start+batch]).argmax(axis=1)
        correct += int(np.sum(predicted == labels[start:start+batch]))
    return correct / len(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "lessons/day3/mnist_data")
    parser.add_argument("--out", type=Path, default=ROOT / "build/trained")
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--validation-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--filters", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-images", type=int, default=32)
    args = parser.parse_args()
    if min(args.train_size, args.validation_size, args.test_size, args.epochs,
           args.batch_size, args.filters, args.export_images) <= 0 or not args.learning_rate > 0:
        parser.error("sizes, epochs, filters, and learning rate must be positive")
    files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    if any(not (args.data / name).is_file() for name in files):
        parser.error(f"MNIST archives missing in {args.data}; see lessons/day3/README.md")
    x = load_idx(args.data / files[0], True)
    y = load_idx(args.data / files[1], False)
    if len(x) != len(y) or args.train_size + args.validation_size > len(x):
        parser.error("training/validation sizes exceed the paired training archive")
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(x))
    train_idx = order[:args.train_size]
    val_idx = order[args.train_size:args.train_size+args.validation_size]
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    model = CNN(filters=args.filters, seed=args.seed)
    optimizer = Adam(model.params, args.learning_rate)
    args.out.mkdir(parents=True, exist_ok=True)
    best, best_epoch, history = -1.0, 0, []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        indices = rng.permutation(len(x_train))
        for start in range(0, len(indices), args.batch_size):
            batch = indices[start:start+args.batch_size]
            loss, grads, _ = model.loss_and_grads(x_train[batch], y_train[batch])
            if not np.isfinite(loss) or any(not np.isfinite(g).all() for g in grads.values()):
                raise RuntimeError("training produced non-finite loss/gradients")
            optimizer.update(model.params, grads)
            total_loss += loss * len(batch)
        val_acc = accuracy(model, x_val, y_val)
        history.append({"epoch": epoch, "train_loss": total_loss / len(x_train),
                        "validation_accuracy": val_acc})
        print(f"epoch {epoch}: loss={history[-1]['train_loss']:.4f}, validation={val_acc:.2%}", flush=True)
        if val_acc > best:
            best, best_epoch = val_acc, epoch
            model.save(args.out / "checkpoint.npz")
    # The test archive is first evaluated AFTER model selection is complete.
    model = CNN.load(args.out / "checkpoint.npz")
    x_test = load_idx(args.data / files[2], True)
    y_test = load_idx(args.data / files[3], False)
    if len(x_test) != len(y_test) or args.test_size > len(x_test):
        parser.error("test size exceeds the paired test archive")
    x_test, y_test = x_test[:args.test_size], y_test[:args.test_size]
    test_acc = accuracy(model, x_test, y_test)
    export_bundle(model, x_test[:args.export_images], y_test[:args.export_images], args.out)
    report = {
        "architecture": f"Conv(1->{args.filters},3x3,same)-ReLU-Pool(2)-Dense(10)-Softmax",
        "seed": args.seed, "train_size": len(x_train), "validation_size": len(x_val),
        "test_size": len(x_test), "epochs": args.epochs, "best_epoch": best_epoch,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "validation_accuracy": best, "test_accuracy": test_acc, "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "numpy": np.__version__, "python": platform.python_version(),
        "dataset_sha256": {name: hashlib.sha256((args.data / name).read_bytes()).hexdigest() for name in files},
        "weights_sha256": hashlib.sha256((args.out / "weights.bin").read_bytes()).hexdigest(),
    }
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Selected epoch {best_epoch}; final test accuracy {test_acc:.2%} on {len(y_test)} images")
    print(f"Artifacts: {args.out}")


if __name__ == "__main__":
    main()
