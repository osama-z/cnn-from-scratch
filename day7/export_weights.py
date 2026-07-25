#!/usr/bin/env python3
"""
DAY 7, STEP 1: THE GOLDEN MODEL EXPORTER
=========================================

THE PROBLEM THIS SOLVES
-----------------------
Day 4 (C) and Day 6 (C++) implement the same architecture, both seed srand(42),
and still produce different predictions:

    day4/cnn_forward.c:384    ((rand()%100)/100.0f - 0.5f) * 0.1f   bias {0.1,-0.1,0}
    day6/cnn_forward.cpp:321  (rand()/RAND_MAX - 0.5f)*2.0f*scale   bias {0,0,0}

Seeding the same PRNG does NOT give you the same numbers if you feed it into
different formulas. The seed only makes each program reproducible against
ITSELF. So "verify all implementations match" is impossible today: they are
not running the same model.

THE FIX: stop generating weights. Start loading them.

    weights.bin   <-- one file, one source of truth
         |
    +----+----+----+
    |    |    |    |
  NumPy  C   C++  VHDL      every implementation LOADS these exact bytes

Once the weights live in a file, the algorithm that produced them stops
mattering. NumPy can use a Mersenne Twister that C could never reproduce,
because C never reproduces it -- C reads it.

WHAT YOU ARE ACTUALLY BUILDING HERE
-----------------------------------
A model file format. That is what ONNX and TFLite ARE: a header, tensor
metadata, and a blob of weights. You will meet those tools in Month 3, and
having hand-rolled one first means they will look obvious instead of magical.

See MODEL_FORMAT.md for the byte-level specification.
"""

import struct
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent

# =====================================================================
# FORMAT CONSTANTS
# =====================================================================

MAGIC   = 0x574E4E43        # bytes 'C','N','N','W' read as little-endian u32
VERSION = 1
DTYPE_FLOAT32 = 0

# Every field is little-endian. x86 and ARM are both little-endian, so this
# costs nothing today -- but writing it down is the difference between a
# format and a habit. A format that does not state its endianness is a bug
# waiting for the first big-endian target.
U32 = "<I"


def _pad4(n: int) -> int:
    """Bytes needed to round n up to a 4-byte boundary."""
    return (-n) & 3


def write_model(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """
    Serialize named tensors to our format.

    WHY 4-BYTE ALIGNMENT MATTERS (a CE detail most formats get right silently):
    a loader wants to point a `const float*` straight at the mapped bytes with
    no copy. On x86 an unaligned float load is merely slow; on some ARM cores
    it faults outright. Padding each record to 4 bytes keeps every float array
    naturally aligned, so zero-copy loading is safe.
    """
    body = bytearray()

    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr, dtype="<f4")   # force LE float32
        raw = arr.tobytes()
        nb  = name.encode("ascii")

        body += struct.pack(U32, len(nb))
        body += nb
        body += b"\x00" * _pad4(len(nb))               # realign after the name

        body += struct.pack(U32, DTYPE_FLOAT32)
        body += struct.pack(U32, arr.ndim)
        for d in arr.shape:
            body += struct.pack(U32, d)

        body += struct.pack(U32, len(raw))
        body += raw
        body += b"\x00" * _pad4(len(raw))              # float32 is already 4-aligned

    # A checksum over the body. Cheap, and it catches the failure mode that
    # actually happens in embedded work: a truncated or partially-flashed file
    # that still parses and then produces silently wrong predictions.
    checksum = sum(body) & 0xFFFFFFFF

    header = struct.pack(U32, MAGIC)
    header += struct.pack(U32, VERSION)
    header += struct.pack(U32, len(tensors))
    header += struct.pack(U32, checksum)               # header is exactly 16 bytes

    path.write_bytes(header + bytes(body))
    print(f"  wrote {path.name}: {len(header) + len(body)} bytes, "
          f"{len(tensors)} tensors, checksum 0x{checksum:08X}")


# =====================================================================
# THE MODEL
# =====================================================================
# Architecture, unchanged since Day 1:
#   Conv2D(1->2, 3x3) -> ReLU -> MaxPool(2) -> Flatten -> Dense(32->3) -> Softmax
#
# LAYOUT CONVENTIONS -- these are the contract every implementation obeys:
#   conv.weight   (out_ch, in_ch, kh, kw)   index: oc*(ic*9) + ic*9 + ky*3 + kx
#   dense.weight  (in_size, out_size)       index: i*out_size + j
#
# The dense layout is row-major (in, out) because that is what day4's
# `weights[i * out_size + j]` already assumed. The golden model's job is to
# freeze a convention, not to invent a new one.

IN_CH, OUT_CH, K = 1, 2, 3
IMG, POOL, NCLS = 8, 2, 3
FLAT = OUT_CH * (IMG // POOL) * (IMG // POOL)      # 2*4*4 = 32


def build_weights() -> dict[str, np.ndarray]:
    # Conv kernels: the two hand-designed edge detectors from Day 1.
    # These are NOT random -- they are the reason the network detects edges.
    conv_w = np.zeros((OUT_CH, IN_CH, K, K), dtype=np.float32)
    conv_w[0, 0] = [[-1, -1, -1],
                    [ 0,  0,  0],
                    [ 1,  1,  1]]        # horizontal edge
    conv_w[1, 0] = [[-1,  0,  1],
                    [-1,  0,  1],
                    [-1,  0,  1]]        # vertical edge
    conv_b = np.zeros(OUT_CH, dtype=np.float32)

    # Dense: pseudo-random, but generated ONCE here and then frozen in the file.
    # Note we can use NumPy's PCG64 freely -- C could never reproduce this
    # generator, and it does not need to. That is the whole point.
    rng = np.random.default_rng(42)
    scale = 1.0 / np.sqrt(FLAT)
    dense_w = (rng.random((FLAT, NCLS), dtype=np.float32) - 0.5) * 2.0 * scale
    dense_b = np.array([0.1, -0.1, 0.0], dtype=np.float32)

    return {
        "conv.weight":  conv_w,
        "conv.bias":    conv_b,
        "dense.weight": dense_w.astype(np.float32),
        "dense.bias":   dense_b,
    }


def build_images() -> tuple[np.ndarray, list[str]]:
    """
    The three 8x8 test images used since Day 1, plus one stability probe.

    WHY THE FOURTH IMAGE EXISTS
    ---------------------------
    Mutation testing found a hole in this suite. Deleting the max-subtraction
    trick from softmax -- i.e. computing exp(x) instead of exp(x - max) --
    still PASSED every check, because the original three images only drive the
    logits to ~16.8, and exp(16.8) = 2e7 is nowhere near the float32 ceiling of
    3.4e38. The guard was never exercised, so the suite could not tell whether
    it was there.

    A test that cannot fail is not a test. Image 3 is a horizontal edge at
    amplitude 60, which drives the largest logit to ~100. exp(100) overflows
    float32 to +inf, and the subsequent inf/inf produces NaN. Any
    implementation that drops the stability trick now fails loudly here.

    This is the reason to mutation-test a verification suite rather than trust
    it: the code was correct, but the evidence for it was not.
    """
    imgs = np.zeros((4, IN_CH, IMG, IMG), dtype=np.float32)
    imgs[0, 0, 4:, :] = 10.0      # bottom half bright -> horizontal edge
    imgs[1, 0, :, 4:] = 10.0      # right half bright  -> vertical edge
    imgs[2, 0, :, :]  = 5.0       # flat               -> uniform
    imgs[3, 0, 4:, :] = 60.0      # same edge, 6x amplitude -> softmax overflow probe
    return imgs, ["horizontal_edge", "vertical_edge", "uniform", "saturated_edge"]


# =====================================================================
# THE REFERENCE IMPLEMENTATION (NumPy)
# =====================================================================
# This is the definition of "correct". Every other implementation is checked
# against these numbers. It is written for clarity, not speed -- the loops
# mirror the C exactly so a mismatch is easy to localize.

def conv2d(x, w, b):
    out_ch, in_ch, kh, kw = w.shape
    _, H, W = x.shape
    kh2 = kh // 2
    out = np.zeros((out_ch, H, W), dtype=np.float32)
    for oc in range(out_ch):
        for y in range(H):
            for x_ in range(W):
                acc = np.float32(b[oc])
                for ic in range(in_ch):
                    for ky in range(-kh2, kh2 + 1):
                        for kx in range(-kh2, kh2 + 1):
                            iy, ix = y + ky, x_ + kx
                            if 0 <= iy < H and 0 <= ix < W:      # zero padding
                                acc += x[ic, iy, ix] * w[oc, ic, ky + kh2, kx + kh2]
                out[oc, y, x_] = acc
    return out


def relu(x):
    return np.maximum(x, 0.0).astype(np.float32)


def maxpool2d(x, p):
    c, H, W = x.shape
    out = np.zeros((c, H // p, W // p), dtype=np.float32)
    for ch in range(c):
        for y in range(H // p):
            for x_ in range(W // p):
                out[ch, y, x_] = x[ch, y*p:(y+1)*p, x_*p:(x_+1)*p].max()
    return out


def dense(x, w, b):
    # w is (in_size, out_size); matches C's weights[i * out_size + j]
    out = np.zeros(w.shape[1], dtype=np.float32)
    for j in range(w.shape[1]):
        acc = np.float32(b[j])
        for i in range(w.shape[0]):
            acc += x[i] * w[i, j]
        out[j] = acc
    return out


def softmax(x):
    e = np.exp(x - x.max())        # the Day 1 stability trick
    return (e / e.sum()).astype(np.float32)


def forward(img, W):
    """Returns every intermediate tensor, so a mismatch can be traced to a layer."""
    a = conv2d(img, W["conv.weight"], W["conv.bias"])
    b = relu(a)
    c = maxpool2d(b, POOL)
    d = c.reshape(-1)
    e = dense(d, W["dense.weight"], W["dense.bias"])
    f = softmax(e)
    return {"conv": a, "relu": b, "pool": c, "flatten": d, "dense": e, "softmax": f}


# =====================================================================
# REFERENCE VECTOR DUMPS
# =====================================================================

def write_reference(path: Path, imgs, names, W):
    """
    Human- and machine-readable expected outputs, layer by layer.

    Per-layer (not just final) output matters: if the C version disagrees only
    at 'dense', you know conv/relu/pool are fine and you have one function to
    debug instead of six.
    """
    lines = [
        "# Golden model reference outputs",
        "# Generated by export_weights.py -- do not edit by hand.",
        "# Format:  TENSOR <image_index> <layer_name> <count>",
        "#          then <count> float values, one per line, %.9g",
        f"# Architecture: Conv2D({IN_CH}->{OUT_CH},{K}x{K}) ReLU MaxPool({POOL}) "
        f"Flatten Dense({FLAT}->{NCLS}) Softmax",
        "",
    ]
    for i, name in enumerate(names):
        lines.append(f"# ---- image {i}: {name} ----")
        flat_img = imgs[i].reshape(-1)
        lines.append(f"TENSOR {i} input {flat_img.size}")
        lines += [f"{v:.9g}" for v in flat_img]

        for layer, t in forward(imgs[i], W).items():
            t = np.asarray(t).reshape(-1)
            lines.append(f"TENSOR {i} {layer} {t.size}")
            lines += [f"{v:.9g}" for v in t]
        lines.append("")

    path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {path.name}: {len(lines)} lines")


def write_vhdl_vectors(outdir: Path, imgs, W):
    """
    Phase 1 (VHDL) testbench vectors.

    A VHDL testbench reads plain text with textio, one value per line, so these
    are deliberately dumber than the file above: no headers, no sections.

    Symmetric int8 quantization (zero_point = 0) is used here, because the first
    thing the VHDL builds is a MAC unit and a conv engine -- the accumulator
    path. Asymmetric zero-points and requantization come later, in the layer
    wrapper. Expected output is the raw int32 accumulator: exactly what a
    conv3x3 datapath produces before any rescaling.
    """
    outdir.mkdir(exist_ok=True)

    cw = W["conv.weight"]
    w_scale = float(np.abs(cw).max()) / 127.0
    i_scale = 10.0 / 127.0                     # images span 0..10, symmetric

    q_w = np.clip(np.round(cw / w_scale), -128, 127).astype(np.int32)
    (outdir / "conv_weights_int8.txt").write_text(
        "\n".join(str(int(v)) for v in q_w.reshape(-1)) + "\n")

    # Only the first three images. Image 3 is a float-path stability probe with
    # amplitude 60, which would simply saturate to 127 under this 0..10 scale
    # and tell the hardware nothing it does not already learn from image 0.
    for i in range(3):
        q_i = np.clip(np.round(imgs[i] / i_scale), -128, 127).astype(np.int32)
        (outdir / f"image{i}_int8.txt").write_text(
            "\n".join(str(int(v)) for v in q_i.reshape(-1)) + "\n")

        # int32 accumulator, zero padding, no bias, no requantization
        acc = conv2d(q_i.astype(np.float32),
                     q_w.astype(np.float32),
                     np.zeros(q_w.shape[0], dtype=np.float32)).astype(np.int64)
        (outdir / f"image{i}_conv_acc_int32.txt").write_text(
            "\n".join(str(int(v)) for v in acc.reshape(-1)) + "\n")

    peak = int(np.abs(acc).max())
    (outdir / "quant_params.txt").write_text(
        f"# symmetric quantization, zero_point = 0\n"
        f"input_scale  {i_scale:.9g}\n"
        f"weight_scale {w_scale:.9g}\n")

    print(f"  wrote {outdir.name}/: int8 vectors for the Phase 1 VHDL testbench")
    print(f"    peak |int32 accumulator| = {peak}  "
          f"({'fits' if peak < 2**31 else 'OVERFLOWS'} int32; "
          f"would {'NOT fit' if peak > 127 else 'fit'} int8 -- Day 5's rule, measured)")


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    print("Building golden model")
    print("=" * 60)

    W = build_weights()
    imgs, names = build_images()

    total = sum(a.size for a in W.values())
    print(f"  {len(W)} tensors, {total} parameters, "
          f"{total * 4} bytes as float32, {total} bytes as int8")

    write_model(HERE / "weights.bin", W)
    write_reference(HERE / "reference_io.txt", imgs, names, W)
    write_vhdl_vectors(HERE / "vhdl_vectors", imgs, W)

    print("=" * 60)
    print("Predictions from the NumPy reference:")
    for i, n in enumerate(names):
        p = forward(imgs[i], W)["softmax"]
        print(f"  image {i} ({n:16s}) -> "
              f"[{', '.join(f'{v*100:5.1f}%' for v in p)}]  class {int(p.argmax())}")
    print()
    print("These numbers are now the definition of correct.")
    print("Run ./verify.py after building the C and C++ implementations.")
