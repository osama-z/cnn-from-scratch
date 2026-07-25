# 🧠 Understanding This Project — Every Decision and Why

> **Purpose:** not *what* the code does — the per-day `README_explanation.md` files
> cover that. This is *why* it was built this way, and why the alternatives were rejected.
>
> **How to use it:** read Part 3 (the trace) with the code open. Answer Part 7 out loud
> without looking. If you can do Part 8 from memory, you own this project.

---

# PART 0 — What you actually built

One convolutional neural network, implemented five times, all proven to compute
the same thing.

```text
        NumPy  ──►   C   ──►  C++  ──►  int8  ──►  VHDL
        Day 1-3     Day 4    Day 6     Day 5     Phase 1
          │           │        │         │          │
          └───────────┴────────┴─────────┴──────────┘
                            │
                     weights.bin
                  ONE set of numbers
              every version loads, verified
                 to agree to 1.2e-10
```

The architecture, unchanged throughout:

```text
8×8 image → Conv2D(1→2, 3×3) → ReLU → MaxPool(2×2) → Flatten → Dense(32→3) → Softmax
   64 px         128 values     128        32           32          3          3
```

119 parameters. 476 bytes as float32, 119 bytes as int8.

---

# PART 1 — Why build from scratch at all?

**The honest answer: because `model.fit()` teaches you nothing you can defend.**

Anyone can train a network in 10 lines of PyTorch. What that leaves you unable to do:

| Question | Framework user | You |
|---|---|---|
| Why does conv detect edges? | "It learns features" | Can show the kernel arithmetic |
| Why int8 and not int4? | — | Range, overflow, and accuracy trade-off |
| Why is your model 4× smaller? | "I quantized it" | Can derive scale and zero-point |
| Why does it run on an MCU? | — | Knows the cycle cost of a float multiply |

The second column is a *hardware engineer's* answer, and it's the only thing that
separates you from the thousands of CS graduates applying to the same programs.

**This is also the reason it stops.** Building from scratch is for *understanding*,
not production. Nobody deploys hand-written conv loops. Once you know what the
framework hides, you use the framework.

---

# PART 2 — The math, and why each piece exists

## 2.1 Convolution — why sliding multiply-add?

An image is a grid of numbers. A *feature* is a local pattern — an edge, a corner.
Local patterns are found by comparing each small patch against a template.

"Compare a patch to a template" = multiply elementwise and sum. Do it at every
position = convolution.

```text
Why a 3×3 kernel and not the whole image?
  Whole image → one weight per pixel per output → millions of parameters,
                and a feature learned at the top-left is not recognised elsewhere.
  3×3 sliding → 9 weights, reused at every position.

  This is WEIGHT SHARING, and it is why CNNs beat dense nets on images:
  an edge is an edge no matter where it appears.
```

**Why does the kernel sum tell you what it does?**

| Kernel sum | Effect | Reason |
|---|---|---|
| 0 | edge detector | flat regions cancel to zero; only *changes* survive |
| 1 | blur / feature preserver | preserves average brightness |
| > 1 | amplifier | scales brightness up |

Your kernel `[-1,-1,-1, 0,0,0, 1,1,1]` sums to zero — bottom row minus top row.
On a flat patch: `10+10+10 − 10−10−10 = 0`. On a horizontal edge: large. **That's
why it detects horizontal edges** — nothing was learned, the arithmetic simply does it.

## 2.2 ReLU — why `max(0,x)`?

Without a non-linearity, stacked layers collapse:

```text
Layer2(Layer1(x)) = W₂(W₁x) = (W₂W₁)x = W₃x     ← one layer, not two
```

Ten linear layers = one linear layer. Depth would be pointless.

**Why ReLU rather than sigmoid or tanh?**
- Gradient is exactly 1 for x>0 — no vanishing gradient
- One comparison, no `exp()` — critical on an MCU
- Sparsity: ~half the outputs become exactly 0, and zeros are cheap

## 2.3 MaxPool — why throw information away?

Deliberate. Three reasons:
1. **Translation tolerance** — an edge shifted by one pixel gives the same pooled result
2. **Compression** — 2×2 pooling cuts data 4×, so later layers are 4× cheaper
3. **Receptive field growth** — after pooling, a 3×3 kernel effectively sees 6×6 of the original

**Why *max* and not average?** You're asking "was this feature present anywhere in
this region?" — that's a max, not a mean. Average dilutes a strong local response.

> **Detail worth knowing:** max pooling needs no special int8 handling. Quantization
> is monotonic (scale is always positive), so the largest integer is the largest real
> value. The same comparison is correct in both worlds — which is why real engines pool
> directly on quantized data.

## 2.4 Softmax — why exponentiate?

You need probabilities: all positive, summing to 1. `exp()` makes everything positive;
dividing by the sum makes it total 1.

**Why not just `x / sum(x)`?** Negative logits would give negative "probabilities".

**Why subtract the max first?**

```c
exp(x - max)   instead of   exp(x)
```

Mathematically identical — the constant cancels in the ratio. Numerically essential:
`exp(100)` overflows float32 (max ≈ 3.4e38), giving `inf`, then `inf/inf = NaN`.

Your Day 7 mutation test proved this matters *and* that your test suite originally
couldn't detect its removal. See Part 5.4.

---

# PART 3 — Trace one number all the way through

**This is the section that produces real understanding.** These are actual outputs
from your code, not illustrations.

Input image 0 — a horizontal edge:

```text
 0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0     ← the edge is here
10 10 10 10 10 10 10 10
10 10 10 10 10 10 10 10
10 10 10 10 10 10 10 10
10 10 10 10 10 10 10 10
```

**Step 1 — Conv at position (y=4, x=4), filter 0.**

The 3×3 window (input rows 3,4,5 × cols 3,4,5), and the kernel:

```text
window          kernel          product
 0  0  0       -1 -1 -1         0  0  0
10 10 10   ×    0  0  0    =    0  0  0
10 10 10        1  1  1        10 10 10      sum = 30
```

Your code outputs exactly **30.0** at `conv(0,4,4)`. Bias is 0, so nothing is added.

**Step 2 — The whole conv output, filter 0:**

```text
row 2:   0  0  0  0  0  0  0  0
row 3:  20 30 30 30 30 30 30 20     ← note the 20s at the edges
row 4:  20 30 30 30 30 30 30 20
row 5:   0  0  0  0  0  0  0  0
```

**Why 20 at the edges and 30 in the middle?** *Zero padding.* At x=0 the kernel's
left column sits outside the image and contributes nothing, so only 2 of 3 columns
count: `10+10 = 20`. **You are looking at the boundary condition in the numbers.**

**Why zeros in rows 0–2 and 5–7?** Flat regions. Top minus bottom = 0. The filter
fires *only* where brightness changes — which is what an edge detector should do.

**Step 3 — ReLU:** 30.0 is positive → unchanged, `30.0`.

**Step 4 — MaxPool 2×2:** the window containing (4,4) has max **30.0** →
`pool(0,2,2) = 30.0`.

**Step 5 — Flatten:** 2×4×4 → a flat vector of 32. No arithmetic, just reinterpretation.

**Step 6 — Dense (32→3):** each output is a weighted sum of all 32 inputs plus bias:

```text
logits = [16.808,  1.210, -10.461]
```

**Step 7 — Softmax:**

```text
max = 16.808
exp(16.808 - 16.808) = 1.000
exp( 1.210 - 16.808) = 1.7e-07
exp(-10.461 - 16.808) = 1.4e-12
                        ─────────
probabilities        = [1.000, 0.000, 0.000]  → class 0
```

> ⚠️ **The class labels are meaningless and that is expected.** The conv kernels are
> hand-designed, but the Dense layer was never trained — it maps good features to
> arbitrary classes. This project verifies that *five implementations agree*, not that
> the network is accurate. Training was Day 2's lesson.

---

# PART 4 — Why each language, in this order

| Day | Language | The one thing it teaches |
|---|---|---|
| 1–3 | Python/NumPy | The math, with nothing in the way |
| 4 | C | Memory is a flat tape; you manage it |
| 5 | C (int8) | Hardware hates floats |
| 6 | C++ | Structure without cost |
| Phase 1 | VHDL | The arithmetic *is* physical |

**Why Python first?** You cannot debug math and memory at the same time. NumPy
removes memory from the problem so you can be sure the math is right — then C
removes NumPy so you can be sure the memory is right. **One unknown at a time.**

**Why C after Python?**

```c
image[c][y][x]   →   ptr[c*H*W + y*W + x]
```

RAM is a single line of bytes. There is no "2D". Writing that formula by hand
teaches you what a tensor *is* — and it's the same formula you'll write in VHDL for
address generation.

**Why int8 (Day 5)?**

| | float32 | int8 |
|---|---|---|
| Multiply on Cortex-M4 | ~20 cycles | **1 cycle** |
| Bytes per weight | 4 | **1** |
| Accuracy cost | — | ~1.5% |

A drone has a battery and 256 KB of SRAM. 20× faster and 4× smaller for 1.5%
accuracy is not a close call.

**Why C++ after C?** Every production inference engine — TFLite, ONNX Runtime,
LibTorch, TensorRT — is C++. To read or extend them you need classes, RAII and
templates. *(CMSIS-NN is the exception, and it's plain C precisely because bare-metal
Cortex-M has no heap and no exceptions.)*

**Why VHDL last?** Because it's the only step a CS graduate cannot take, and because
it needs everything before it: you cannot design a datapath until you know exactly
what arithmetic must happen.

---

# PART 5 — The engineering decisions, and the rejected alternatives

## 5.1 Why RAII — what C actually gets wrong

```c
float* buf = malloc(n * sizeof(float));
if (error) return -1;          /* ← LEAK: free() skipped */
free(buf);
```

```cpp
std::vector<float> buf(n);
if (error) return -1;          /* ← destructor runs. Cannot leak. */
```

The key isn't convenience — it's **every exit path**: normal return, early return,
`break`, and the one C cannot handle at all, a thrown exception. The compiler emits
the destructor call on all of them.

**Proven, not asserted:** `make asan` runs both binaries under LeakSanitizer. They
allocate constantly, contain zero `free()` and zero `delete`, and report nothing.

## 5.2 Why templates — and why `Tensor<int8_t>` alone is a trap

The obvious approach recreates Day 5's bug:

```cpp
template <typename T>
T acc = 0;                  // T is int8_t → OVERFLOW
acc += input[i] * w[i];     // 127×127 = 16,129, and 9 of those = 145,161
```

The accumulator must be a **different type** from the storage:

```cpp
template <typename T> struct AccumTraits         { using type = float;   };
template <>           struct AccumTraits<int8_t> { using type = int32_t; };
```

**Rejected alternative:** two separate files, one float and one int8. Rejected
because they drift — fix a bug in one and the other silently keeps it. That is
precisely the failure the golden model exists to catch.

## 5.3 Why `forward()` can't be a virtual template

```cpp
template <typename T>
virtual Tensor<T> forward(const Tensor<T>&) = 0;   // ILLEGAL C++
```

A vtable is a **fixed-size array** of function pointers, laid out at compile time.
Templates instantiate on demand, so a virtual template would need one slot per type
anyone ever uses — including types in files not yet written. Unbounded size, so the
compiler cannot lay it out. Hence: template the *class*, not the method.

## 5.4 Why a golden model — the bug that was actually there

Day 4 and Day 6 both called `srand(42)` and computed **different networks**:

```c
day4:  ((rand()%100)/100.0f - 0.5f) * 0.1f     bias {0.1, -0.1, 0}
day6:  (rand()/RAND_MAX - 0.5f) * 2.0f * scale bias {0, 0, 0}
```

**Same seed ≠ same numbers when the formula differs.** A seed only makes a program
reproducible against *itself*.

The fix inverts the dependency: **stop generating, start loading.** Once weights live
in a file, the generator stops mattering — NumPy uses a PCG64 that C could never
reproduce, and doesn't need to, because C *reads* it.

**Why verify every layer, not just the prediction?** If only softmax disagrees, the
bug is in one of six layers. Layer-by-layer localizes it: **the first diverging layer
is where the bug lives.** This is exactly how RTL is debugged against a C reference
model — you probe intermediate signals, not just the output port.

**Why mutation testing?** Because a suite that always passes proves nothing. Injecting
bugs found that removing the softmax stability guard *still passed* — the logits only
reached 16.8, and `exp(16.8)` never overflows. The code was right; the evidence wasn't.
Adding a 4th image at amplitude 60 (logit ~100, `exp` → `inf` → `NaN`) closed it.

> **A passing test is evidence only once you've checked that it can fail.**

## 5.5 Why the file format has a magic number, version, checksum and alignment

| Field | The specific failure it catches |
|---|---|
| magic `0x574E4E43` | A wrong file fails on byte 0, instead of being parsed as garbage dimensions |
| version | An old loader refuses a newer file rather than misreading it |
| checksum | A brownout-truncated flash write that still parses and predicts nonsense |
| 4-byte alignment | Lets a `const float*` point straight into the buffer — **zero copy** |

That last one is a hardware detail: an unaligned float load is *slow* on x86 but
**faults** on some ARM cores. It's why TFLite (FlatBuffers) is alignment-strict —
it's built to be `mmap`'d on embedded targets.

**What you actually built is a model file format.** That's what ONNX and TFLite *are*.

---

# PART 6 — The single thread through everything

If you remember one thing, remember this:

```text
        int8 × int8 overflows → you MUST accumulate in int32

Day 5   a comment you have to remember              (C)
Day 6   a type the compiler enforces                (C++ AccumTraits)
Day 7   48,387 — the peak, measured on real data    (golden model)
VHDL    a 32-bit register, 384/384 verified        (hardware)
```

One rule, four levels of abstraction, each enforcing it more strongly than the last.
Day 5 hopes you remember. Day 6 makes forgetting a compile error. Phase 1 makes it
a wire.

**That progression is your entire project in four lines**, and it is the paragraph
that goes in your application.

---

# PART 7 — Self-test (answer out loud, no notes)

**Math**
1. Why does a kernel summing to zero detect edges?
2. Why do stacked layers collapse without a non-linearity?
3. Why max pooling and not average?
4. Why subtract the max before `exp()` if it cancels out anyway?

**C / memory**
5. Why is `image[c][y][x]` written `ptr[c*H*W + y*W + x]`?
6. Where exactly does C leak that C++ cannot?

**Quantization**
7. Why does `int8 × int8` need an `int32` accumulator? Give the number.
8. In int8, why is `relu(x) = max(0, x)` **wrong**?
9. Why does max pooling need no int8 special case?

**C++**
10. Why doesn't `Tensor<int8_t>` alone fix the overflow?
11. Why can't `forward()` be a virtual template function?
12. What does `if constexpr` cost at run time? How did you prove it?

**Verification**
13. Two programs both call `srand(42)`. Why might they still differ?
14. Why compare every layer instead of the final prediction?
15. Your suite passed after a guard was deleted. What was wrong — code or test?

<details>
<summary>Answers</summary>

1. Flat regions cancel to zero; only *changes* survive.
2. `W₂(W₁x) = (W₂W₁)x` — the composition is still a single matrix.
3. You're asking "was the feature present anywhere here?" — that's a max. Averaging dilutes a strong local response.
4. Mathematically it cancels; numerically `exp(100)` overflows float32 to `inf`, then `inf/inf = NaN`.
5. RAM is one flat line of bytes. There is no 2D — the formula is the addressing.
6. Any early exit path: `return` on error, `break`, or a thrown exception. The destructor runs on all of them; `free()` doesn't.
7. `127×127 = 16,129`, and a 3×3 kernel sums nine of them → `145,161`. Far past int8's 127. Measured peak in your data: **48,387** — needing 17 bits signed, so even int16 would overflow.
8. Integer 0 doesn't represent real 0.0 — `zero_point` does. Correct form is `max(zero_point, x)`.
9. Quantization is monotonic (positive scale), so the largest integer is the largest real value.
10. The accumulator would follow `T` and be int8 — recreating the overflow automatically. Storage and accumulator must be different types.
11. The vtable is fixed-size and built at compile time; templates instantiate on demand, so it would need unbounded slots.
12. Nothing — the discarded branch is never emitted. Proved with `make asm`: the float build has `mulss`, the int8 build has `movsbl`/`imull`, sharing no arithmetic instructions.
13. The seed only makes a program reproducible against itself. Feed it into different formulas and you get different weights.
14. The first diverging layer localizes the bug; everything upstream is proven correct.
15. The **test**. The guard was correct but never exercised, because the logits never got large enough to overflow.

</details>

---

# PART 8 — Explain it in 60 seconds

> *"I implemented a convolutional neural network from first principles — NumPy, then
> C with manual memory management, then C++ with templates, then int8 fixed-point,
> and now VHDL. All five load one shared weights file and I verify them layer by layer;
> they agree to 1.2e-10.*
>
> *Building it that way taught me what a framework hides. `int8 × int8` overflows
> unless you accumulate in int32 — I first wrote that as a comment I had to remember,
> then encoded it in C++'s type system so the compiler enforces it, and in the FPGA
> version it becomes a 32-bit accumulator register. Same rule, three levels of abstraction.*
>
> *I also mutation-tested my own verification suite and found it couldn't detect the
> removal of a numerical-stability guard — the code was correct but the evidence wasn't,
> so I added a test case that forces the overflow."*

Engineers respect the second paragraph. Everyone understands the first. The third is
what makes you sound like someone who has actually shipped something.

---

## Where to look in the code

| To understand | Read |
|---|---|
| The math, visually | `day1/README_explanation.md` |
| Backprop and optimizers | `day2/README_explanation.md` |
| Pointers and memory | `day4/cnn_forward.c` |
| Quantization | `day5/README_explanation.md` |
| RAII, templates, `if constexpr` | `day6/README_explanation.md` |
| Verification and the file format | `day7/README_explanation.md`, `day7/MODEL_FORMAT.md` |
| Formula cheat sheet | `notes_week1.md` |
