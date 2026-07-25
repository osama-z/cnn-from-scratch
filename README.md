# CNN From Scratch — NumPy → C → C++ → int8 → VHDL

**One convolutional neural network, implemented five ways, every output verified identical.**

No frameworks. Every operation — convolution, backpropagation, quantization, memory
layout — written from first principles and checked against a single reference model.

```text
        NumPy   ──►    C    ──►   C++   ──►   int8   ──►   VHDL
      days 1-3       day 4       day 6        day 5       in progress
          │            │           │            │            │
          └────────────┴───────────┴────────────┴────────────┘
                                  │
                            weights.bin
                    one frozen set of numbers that
                     every implementation loads
```

```console
$ cd day7 && make verify

Golden-model verification
  comparing every layer of 4 images against the NumPy reference
    [C  ] PASS  (largest deviation 1.20e-10)
    [C++] PASS  (largest deviation 1.20e-10)

  RESULT: golden model verified. NumPy = C = C++, bit-close.
```

**Author:** Osama Z.
**Focus:** embedded AI · quantization · hardware-software co-design

---

## Results

| | Measured |
|---|---|
| MNIST test accuracy (NumPy, from scratch) | **95.0%** on 10,000 unseen digits |
| Cross-implementation agreement | **1.2e-10** across every layer of every image |
| C inference throughput | **16,000+ FPS**, 1.4 KB working memory |
| int8 vs float32 | **4× smaller** — all 119 parameters: 476 B → 119 B; ~1.3 pp probability error |
| Peak int32 conv accumulator | **48,387** — needs 17 bits signed, so int8 *and* int16 overflow |
| Memory safety | ASan + UBSan clean on all binaries, zero `free()` in the C++ |

---

## Why this exists

A framework can train a network in ten lines. It cannot tell you *why* `int8 × int8`
overflows, why an unaligned float load faults on ARM, or why your C and C++ builds
silently disagreed. This project answers those questions by building the thing.

The single thread running through all of it:

```text
        int8 × int8 overflows → you MUST accumulate in int32

Day 5    a comment you have to remember             (C)
Day 6    a type the compiler enforces               (C++ AccumTraits)
Day 7    48,387 — the peak, measured on real data   (golden model)
VHDL     a 17-bit-minimum accumulator register      (hardware)
```

One rule, four levels of abstraction, each enforcing it more strongly than the last.

---

## Where to start

**New here? Read in this order:**

| # | File | What you get |
|---|---|---|
| 1 | **[UNDERSTAND.md](UNDERSTAND.md)** | Every design decision and why the alternative was rejected. One number traced through all six layers with real output. Start here. |
| 2 | [day7/README_explanation.md](day7/README_explanation.md) | The verification story — and the bug that was hiding in the repo |
| 3 | [day6/README_explanation.md](day6/README_explanation.md) | RAII, virtual dispatch, and why `Tensor<int8_t>` alone recreates an overflow bug |
| 4 | [notes_week1.md](notes_week1.md) | Formula cheat sheet — every equation on one page |

**Want the code?** `day7/cnn.hpp` is the cleanest expression of the engine.
`day4/cnn_forward.c` is the one that teaches pointers.

---

## What's in each day

| Day | Language | Builds | Deep dive |
|---|---|---|---|
| **1** | Python/NumPy | Convolution, ReLU, MaxPool, Dense, Softmax from nothing | [explanation](day1/README_explanation.md) |
| **2** | Python/NumPy | Cross-entropy, backprop by chain rule, SGD/Momentum/Adam | [explanation](day2/README_explanation.md) |
| **3** | Python/NumPy | MNIST at 95%, IDX parsing, train/test discipline | [explanation](day3/README_explanation.md) |
| **4** | C | Same network, `float*` and 1D indexing, manual `malloc`/`free` | [explanation](day4/README_explanation.md) |
| **5** | C | int8 quantization, scale/zero-point, the int32 accumulator rule | [explanation](day5/README_explanation.md) |
| **6** | C++ | Classes, RAII, templates — float and int8 from one source | [explanation](day6/README_explanation.md) |
| **7** | Python + C + C++ | The golden model, a hand-rolled binary format, layer-by-layer verification | [explanation](day7/README_explanation.md) · [format spec](day7/MODEL_FORMAT.md) |
| **next** | VHDL | MAC unit and conv engine, verified against the golden model | — |

---

## Run it

```bash
# Verify all implementations agree (the headline claim)
cd day7 && make verify

# Prove the loader has no leaks and no buffer overruns
cd day7 && make asan

# Prove a corrupted model file is rejected, not trusted
cd day7 && make test-robust

# C++: object-oriented version, then float-vs-int8 from one templated source
cd day6 && make run
cd day6 && make asan     # RAII proof: zero leaks, zero free() in the source
cd day6 && make asm      # `if constexpr` proof: the two builds share no arithmetic

# Earlier days
cd day4 && make && ./cnn_forward       # C, ~16k FPS
cd day5 && make && ./fixed_point       # int8 quantization
cd day1 && python part1_convolution.py # the math, visualized
```

MNIST archives are not in the repo (11.5 MB). `day3/README_explanation.md` covers
re-downloading them.

---

## Two things worth knowing before you read the output

**The class labels are meaningless, deliberately.** The convolution kernels are
hand-designed edge detectors, but the Dense layer is generated and never trained — so it
maps good features to arbitrary classes. This project verifies that five implementations
*agree*, not that the network is accurate. Training is Day 2's subject.

**"Verified" means bit-close, not bit-identical.** The measured deviation is 1.2e-10
against a 1e-4 tolerance. `expf()`, NumPy's `exp()`, and fused multiply-add round
differently; demanding exactness across languages and compilers would be wrong.

---

## Verification approach

The interesting part isn't that the tests pass — it's that they were checked for the
ability to fail. Bugs were deliberately injected to see whether the suite would notice:

| Injected bug | Caught? |
|---|---|
| Off-by-one in convolution padding | ✅ first divergence at `conv` |
| Dense weight layout transposed | ✅ first divergence at `dense` |
| Softmax loses its numerical-stability guard | ❌ **passed — a hole in the test** |

The third exposed a gap: the original images only drove logits to ~16.8, and `exp(16.8)`
never overflows, so the guard was never exercised. A fourth test image at amplitude 60
(logit ≈ 100 → `exp` → `inf` → `NaN`) closed it.

**A passing suite is evidence only once you have checked that it can fail.** The same
reasoning applies to the RTL testbench coming next.

---

## Repo layout

```text
UNDERSTAND.md      every decision, and why — start here
ROADMAP.md         the plan and its revisions
notes_week1.md     formula cheat sheet
daily_log.md       what was learned, day by day

day1/ day2/ day3/  Python — the math
day4/ day5/        C — memory and fixed-point
day6/              C++ — structure without cost
day7/              the golden model and verifier
  cnn.hpp            the reusable float engine
  model_io.h         header-only loader, shared by C and C++
  export_weights.py  writes weights.bin + reference outputs + VHDL vectors
  verify.py          compares every layer against NumPy
  MODEL_FORMAT.md    byte-level format spec
  vhdl_vectors/      int8 test vectors for the hardware testbench
```

Each day folder holds heavily commented source (23–32% comments) plus a
`README_explanation.md` deriving the concepts from first principles.

---

## License

MIT — see [LICENSE](LICENSE).
