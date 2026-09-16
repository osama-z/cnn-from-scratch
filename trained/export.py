"""Export learned weights and NCHW inputs using the existing Day 7 format."""
from pathlib import Path
import numpy as np
from engine.model_format import write_model


def export_bundle(model, images, labels, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    images = np.asarray(images, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != (len(images),):
        raise ValueError("one label per exported image required")
    reference = model.forward(images, trace=True)
    model.save(out / "checkpoint.npz")
    write_model(out / "weights.bin", model.params)
    write_model(out / "samples.bin", {"input": images})
    np.savez(out / "samples.npz", images=images, labels=labels)
    np.savez(out / "reference.npz", **reference)
