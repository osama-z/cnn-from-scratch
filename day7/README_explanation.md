# 🥇 Day 7: The Golden Model (One Truth, Many Implementations)

> **Core Concept:** You now have the same CNN in NumPy, C, and C++. Are they
> actually computing the *same thing*? Until today the honest answer was **no** —
> and you could not even have known, because nothing checked. This day builds the
> thing that checks: a single frozen set of weights every implementation loads,
> and a verifier that proves they agree, layer by layer.

---

## 1. The Bug That Was Hiding in Plain Sight

Day 4 (C) and Day 6 (C++) run the same architecture. Both seed `srand(42)`.
Both looked correct. But they never produced the same predictions:

```text
    day4/cnn_forward.c:384    dense_w = ((rand()%100)/100.0f - 0.5f) * 0.1f
                              dense_b = {0.1, -0.1, 0.0}

    day6/cnn_forward.cpp:321  dense_w = (rand()/RAND_MAX - 0.5f) * 2.0f * scale
                              dense_b = {0, 0, 0}
```

**Seeding the same PRNG does not give you the same numbers if you feed it into
different formulas.** `srand(42)` only makes each program reproducible against
*itself*. Two programs, two formulas, two different networks that happen to share
a shape.

This is the single most dangerous class of bug in numerical code: **no crash, no
warning, plausible-looking output, silently wrong.** On a drone it is the
detector that quietly stops seeing people.

---

## 2. The Fix: Stop Generating, Start Loading

```text
                    ┌─────────────────┐
                    │   weights.bin   │   one file, one source of truth
                    └────────┬────────┘
                             │  every implementation LOADS these exact bytes
          ┌──────────┬───────┼───────┬──────────┐
          ▼          ▼       ▼       ▼          ▼
       NumPy        C       C++    (int8)     VHDL
     reference   golden_c golden_cpp Day 6   Phase 1
```

Once the weights live in a file, **the algorithm that produced them stops
mattering.** NumPy's `export_weights.py` uses a PCG64 generator that C could
never reproduce — and does not need to, because C never reproduces it. C reads
the file.

This inverts the dependency. Before: each language had to re-derive the weights
identically (impossible across `rand()` implementations). After: one language
writes, all languages read.

---

## 3. What You Actually Built: A Model File Format

`weights.bin` is not an ad-hoc dump. It has a header, a magic number, tensor
metadata, alignment rules, and a checksum — see `MODEL_FORMAT.md`.

```text
  magic 0x574E4E43   ← reject a JPEG or a truncated file on byte 0
  version            ← let a future loader refuse an old file
  n_tensors
  checksum           ← catch a brownout-truncated flash write
  ─────────
  per tensor: name, dtype, shape, 4-byte-aligned data
```

**This is what ONNX and TFLite ARE.** ONNX is protobuf + an external weight
blob; TFLite is a FlatBuffer. Both start with a magic number, both are alignment-
strict so they can be `mmap`'d on embedded targets, both separate graph metadata
from the weight bytes. You will meet them in Month 3 — and because you built one
by hand first, they will look like your `.bin` with more features, not like magic.

### The alignment detail (a CE payoff)

Every field before a tensor's `data` is a multiple of 4 bytes, so `data` always
begins 4-byte aligned. The loader then points a `const float*` **straight into
the loaded buffer** — no copy, no second allocation:

```c
mt->data = (const float*)(const void*)(m->blob + off);   // zero-copy
```

On x86 an unaligned float load is merely slow. On several ARM cores it **faults**.
This is precisely why real formats are so fussy about alignment — they are
designed to be memory-mapped and read in place on hardware exactly like your
Month 4 targets.

---

## 4. One Loader, Two Languages

`model_io.h` is header-only C that is also valid C++. `golden_c.c` and
`golden_cpp.cpp` both `#include` it.

That matters for what the verification actually *proves*. If C and C++ each had
their own parser, "C and C++ agree" would partly be a claim about two parsers
being equivalent. Sharing one loader removes that variable: the comparison is
between two **inference engines**, reading provably identical bytes.

---

## 5. Verify Every Layer, Not Just the Answer

`verify.py` does not just compare final predictions. Each implementation emits
every intermediate tensor:

```text
    TENSOR 0 input   64
    TENSOR 0 conv    128
    TENSOR 0 relu    128
    TENSOR 0 pool    32
    TENSOR 0 flatten 32
    TENSOR 0 dense   3
    TENSOR 0 softmax 3
```

and the verifier checks all of them against NumPy.

**Why:** if two implementations disagree only at `softmax`, the bug could be in
any of six layers. Comparing layer by layer localizes it — the *first* layer that
diverges is where the bug lives, and everything upstream is proven correct.

This is not a Python habit. It is **exactly how you debug RTL against a C
reference model** in Phase 1: you probe intermediate signals, not just the output
port. Day 7 is rehearsal for hardware verification.

### Closeness, not bit-identity

```text
    [C  ] PASS  (largest deviation 1.20e-10 at image 1 'softmax')
    [C++] PASS  (largest deviation 1.20e-10 at image 1 'softmax')
```

The tolerance is `1e-4`, and the real deviation is `1e-10`. Not exactly zero,
because C's `expf()`, NumPy's `exp()`, and compiler fused-multiply-add all round
slightly differently. Those differences are correctness-preserving. Demanding
bit-identity across languages would be wrong — `1e-10` is agreement.

*(int8 is a different regime: its deviations are large and expected, which is why
that comparison lives in Day 6's `templated_cnn.cpp`, not here.)*

---

## 6. The VHDL Bridge (Why This Feeds Phase 1)

`export_weights.py` also writes `vhdl_vectors/`: the same weights and images
quantized to int8, plus the **expected int32 accumulator** for the convolution.

```text
    peak |int32 accumulator| = 24,384
```

That number is **Day 5's overflow rule, measured on real data.** 24,384 does not
fit in int8 (max 127); it needs int32. When your VHDL `conv3x3` engine runs in
August, its accumulator register must hold this, and the testbench will compare
against `image*_conv_acc_int32.txt` — these exact files.

The golden model is not just Day 7's deliverable. It is the reference vector for
every implementation you build for the next four months.

---

## 7. Files

| File | Role |
|---|---|
| `export_weights.py` | Writes `weights.bin`, the NumPy reference, and VHDL vectors. Contains the reference implementation. |
| `model_io.h` | Header-only loader, shared by C and C++. |
| `golden_c.c` | Day 4's C, loading weights instead of generating them. |
| `golden_cpp.cpp` | Day 6's C++ classes, loading weights. |
| `verify.py` | Runs both binaries, compares every layer to NumPy. |
| `MODEL_FORMAT.md` | Byte-level format spec. |
| `weights.bin` | The golden model. 636 bytes. |

---

## 8. How to Run

```bash
make model        # regenerate weights.bin + reference vectors from NumPy
make verify       # build C and C++, prove NumPy = C = C++ layer by layer
make asan         # prove the loader has no leaks and no buffer overruns
make test-robust  # prove a corrupted file is rejected, not trusted
```

Expected: `golden model verified. NumPy = C = C++, bit-close.`

---

## ✅ Self-Test

1. Two programs both call `srand(42)`. Why might they still compute different
   networks?
2. Why does the format need a magic number *and* a version *and* a checksum —
   what distinct failure does each catch?
3. Why is `data` forced to a 4-byte boundary? What breaks on ARM if it isn't?
4. Why compare every layer instead of just the final prediction?
5. The C and C++ outputs differ by `1e-10`. Is that a bug? Why or why not?
