#!/usr/bin/env python3
"""
DAY 7, STEP 5: THE VERIFIER
============================

Golden-model testing. Runs the C and C++ binaries, parses their TENSOR output,
and compares every layer of every image against the NumPy reference produced by
export_weights.py.

WHY COMPARE EVERY LAYER, NOT JUST THE FINAL PREDICTION
------------------------------------------------------
If two implementations disagree only at the softmax, the bug could be in any of
the six layers. Comparing layer by layer localizes it: the FIRST layer that
diverges is where the bug lives, and everything before it is proven correct.
This is exactly how you debug an RTL pipeline against a C reference model in
Phase 1 -- you probe intermediate signals, not just the output port.

TOLERANCE
---------
Not exact equality. C uses expf(); NumPy uses its own exp(); the compiler may
contract a*b+c into a fused multiply-add with different rounding. These produce
differences at the 1e-6 level that are correctness-preserving. We assert
closeness, not bit-identity. (int8 is a separate story -- that lives in Day 6's
templated_cnn, where the differences are large and expected.)
"""

import subprocess
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
ATOL = 1e-4      # absolute tolerance
RTOL = 1e-4      # relative tolerance

LAYERS = ["input", "conv", "relu", "pool", "flatten", "dense", "softmax"]


def parse_tensors(text: str) -> dict[tuple[int, str], np.ndarray]:
    """Parse 'TENSOR <img> <layer> <n>' blocks into {(img, layer): array}."""
    out, lines, i = {}, text.splitlines(), 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("TENSOR"):
            _, img, layer, n = line.split()
            img, n = int(img), int(n)
            vals = [float(lines[i + 1 + k]) for k in range(n)]
            out[(img, layer)] = np.array(vals, dtype=np.float64)
            i += n + 1
        else:
            i += 1
    return out


def load_reference() -> dict[tuple[int, str], np.ndarray]:
    return parse_tensors((HERE / "reference_io.txt").read_text())


def run(binary: str) -> dict[tuple[int, str], np.ndarray]:
    exe = HERE / binary
    if not exe.exists():
        print(f"  ERROR: {binary} not built. Run `make` first.")
        sys.exit(2)
    res = subprocess.run([str(exe), str(HERE / "weights.bin")],
                         capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  ERROR: {binary} exited {res.returncode}\n{res.stderr}")
        sys.exit(2)
    return parse_tensors(res.stdout)


def n_images(ref) -> int:
    """Derive the image count from the reference rather than hardcoding it,
    so adding a test case means editing export_weights.py only."""
    return max(img for img, _ in ref) + 1


def compare(name: str, ref, got) -> bool:
    ok = True
    worst = 0.0
    worst_where = None
    for img in range(n_images(ref)):
        for layer in LAYERS:
            key = (img, layer)
            if key not in got:
                print(f"    [{name}] image {img} missing layer '{layer}'")
                ok = False
                continue
            a, b = ref[key], got[key]
            if a.shape != b.shape:
                print(f"    [{name}] image {img} '{layer}' shape {b.shape} != {a.shape}")
                ok = False
                continue
            diff = np.abs(a - b)
            m = float(diff.max()) if diff.size else 0.0
            if m > worst:
                worst, worst_where = m, f"image {img} '{layer}'"
            if not np.allclose(a, b, atol=ATOL, rtol=RTOL):
                bad = int(diff.argmax())
                print(f"    [{name}] DIVERGES at image {img} '{layer}' elem {bad}: "
                      f"ref={a[bad]:.9g} got={b[bad]:.9g} (|d|={m:.2e})")
                ok = False
    status = "PASS" if ok else "FAIL"
    print(f"    [{name}] {status}  (largest deviation {worst:.2e} at {worst_where})")
    return ok


def main() -> int:
    if not (HERE / "reference_io.txt").exists():
        print("  reference_io.txt missing. Run `python export_weights.py` first.")
        return 2

    print("Golden-model verification")
    print("=" * 60)
    ref = load_reference()
    n = n_images(ref)
    print(f"  tolerance: atol={ATOL}, rtol={RTOL}")
    print(f"  comparing every layer of {n} images against the NumPy reference\n")

    results = {
        "C   (golden_c)":   compare("C  ", ref, run("golden_c")),
        "C++ (golden_cpp)": compare("C++", ref, run("golden_cpp")),
    }

    print()
    print("=" * 60)
    all_ok = all(results.values())
    for name, ok in results.items():
        print(f"  {name:20s} {'MATCHES golden model' if ok else 'DIVERGED'}")

    # The final predictions, side by side, as the human-readable payoff.
    print("\n  Predictions (all implementations, being identical, agree):")
    for img in range(n):
        p = ref[(img, "softmax")]
        print(f"    image {img}: [{', '.join(f'{v*100:5.1f}%' for v in p)}]  "
              f"class {int(p.argmax())}")

    print()
    if all_ok:
        print("  RESULT: golden model verified. NumPy = C = C++, bit-close.")
        print("  This is the reference every later implementation checks against,")
        print("  including the Phase 1 VHDL testbench (see vhdl_vectors/).")
        return 0
    else:
        print("  RESULT: at least one implementation diverged. See the first")
        print("  diverging layer above -- that is where the bug lives.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
