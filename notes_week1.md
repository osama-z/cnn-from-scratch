# 📝 AI Engineer Notes — Month 1

> Quick-reference formulas and key insights from every day.
> Open this file in VS Code while coding as your cheat sheet.

---

## WEEK 1: Python — Build the Math

### Day 1: Convolution & CNN Forward Pass

| Operation | Formula | Purpose |
|-----------|---------|---------|
| Convolution | `out[y][x] = Σ input[y+ky][x+kx] * kernel[ky][kx]` | Feature detection |
| ReLU | `max(0, x)` | Non-linearity (curves) |
| Max Pool | `max(2x2 region)` | Spatial compression |
| Dense | `x @ W + b` | Classification |
| Softmax | `exp(x) / Σ exp(x)` | Probabilities |

**Kernel Sum Rule:**
- Sum = 0 → Edge detector
- Sum = 1 → Feature extractor
- Sum > 1 → Amplifier

**Multi-channel:** Each filter spans ALL input channels → 1 output channel.

---

### Day 2: Loss, Backprop, Training

**Loss Functions:**
- MSE: `(1/n) × Σ(pred - target)²` — weak for classification
- **Cross-Entropy:** `-Σ target × log(pred)` — screams when wrong, whispers when right

**⭐ THE KEY FORMULA:** `Softmax + CE gradient = predicted - target`

**Backprop (Chain Rule):** `dz/dx = dz/dy × dy/dx`

| Layer | Backward |
|-------|----------|
| Dense | `dW = input^T × d_out`, `db = d_out`, `dx = d_out @ W^T` |
| ReLU | `dx = d_out × (input > 0)` — gate: pass or block |
| Softmax+CE | `d_logits = probs - target` |

**Optimizers:**
- SGD: `W -= lr × grad`
- Momentum: `v = 0.9v - lr×grad; W += v`
- **Adam:** adaptive lr per weight (default choice, 90% of industry)

**Vocabulary:**
- Epoch = one pass through ALL data
- Batch = subset of data
- Iteration = one weight update

---

### Day 3: MNIST Real Data

**Data Pipeline:**
1. Load binary IDX files → `(60000, 28, 28)` uint8
2. Normalize: `/ 255.0` → range `[0, 1]` — **critical!**
3. Flatten: `(28, 28)` → `(784,)`
4. One-hot: label `7` → `[0,0,0,0,0,0,0,1,0,0]`
5. Shuffle every epoch (prevent order memorization)

**Train/Test Split:** the network trains on one set and is scored on another it
never saw. If train acc >> test acc → **Overfitting** (memorized, not learned).

> ⚠️ The IDX files hold 60K train / 10K test, but `part1_mnist_classifier.py`
> deliberately uses a **10,000 / 2,000 subset** — the full set takes hours in pure
> NumPy, which is exactly the problem PyTorch solves. Quote the subset, not the
> file size.

**Result:** 784→128→64→10 dense net, **8.1% → 95.0% test accuracy**.

```text
train 100.0%  vs  test 95.0%   →  5-point GENERALIZATION GAP
```

That gap *is* overfitting, and it's the whole reason a test set exists. Aggregate
accuracy also hides an 8-point spread: digit 1 hits 99.1%, digit 7 only 91.2%.
**Always break accuracy down per class.**

---

## WEEK 2: C — Rewrite for Hardware

### Day 4: CNN Forward Pass in C

**1D Pointer Indexing (THE key skill):**
```c
// 2D image
index = row * width + col

// 3D tensor (channel, height, width)
index = ch * (H * W) + row * W + col
```

**Memory Management:**
```c
float* buf = (float*)malloc(size * sizeof(float));  // allocate
// ... use it ...
free(buf);  // MANDATORY: every malloc needs a free!
```

**In-place operations** save memory on drones (no extra buffers).

**Performance:** C version runs at **16,000+ FPS** vs Python ~100 FPS.

---

### Day 5: Fixed-Point Arithmetic (int8)

**Why kill float?**
- float multiply: ~20 clock cycles on ARM Cortex-M4
- int8 multiply: ~1 clock cycle
- int8 uses 4x less memory (1 byte vs 4 bytes)

**⭐ THE QUANTIZATION FORMULA:**
```text
QUANTIZE:   int8  = round(float / scale) + zero_point
DEQUANTIZE: float = (int8 - zero_point) × scale

scale      = (max - min) / 255.0
zero_point = round(-min / scale) - 128
```

**Int8 Convolution Rule:**
- `int8 × int8` → accumulate into `int32` (prevent overflow!)
- After all 9 kernel multiplies: scale `int32` back to `int8`

**ReLU Trap:**
- Float: `relu(x) = max(0.0, x)`
- Int8: `relu(x) = max(zero_point, x)` ← NOT `max(0, x)`!

**Trade-off:** 4x smaller, 20x faster, -1.5% accuracy loss.

**Golden Rule:** Train in float. Deploy in int8.

---

### Day 6: C++ — RAII, Polymorphism, Templates ✅

**Why C++?** TFLite, ONNX Runtime, LibTorch, TensorRT are all C++.
CMSIS-NN is the exception (plain C — bare-metal MCU, no heap, no STL).

**RAII = Resource Acquisition Is Initialization**
```cpp
std::vector<float> buf(n);   // constructor allocates
if (failed) return -1;       // destructor runs. No leak.
```
The point is *every exit path*: early return, break, **and thrown exception**.
In C, an early `return` skips your `free()`. In C++ it cannot.
> Verified, not assumed: `make asan` → zero leaks, zero `free()` in the file.

**Virtual functions:** one interface, six layers, one loop.
```cpp
virtual Tensor forward(const Tensor&) = 0;   // "= 0" = pure virtual = MUST implement
virtual ~Layer() = default;                  // forget this → derived part leaks
for (auto& L : layers) x = L->forward(x);
```

**⭐ THE TEMPLATE TRAP (the whole point of Day 6):**
```cpp
template <typename T>
T acc = 0;                  // ← T is int8_t. OVERFLOW. Day 5's bug, auto-generated.
```
Templating the *storage* type recreates the overflow. The accumulator must be a
**separate type**, chosen by a trait:
```cpp
template <typename T> struct AccumTraits         { using type = float;   };
template <>           struct AccumTraits<int8_t> { using type = int32_t; };
```

| Where | How the int32 rule lives |
|---|---|
| Day 5 (C) | a comment you must remember |
| Day 6 (C++) | `Accum<int8_t> = int32_t` — compiler enforces it |
| Phase 1 (VHDL) | a 32-bit accumulator register — physical wire |

**`if constexpr`** = resolved at COMPILE time. Discarded branch is never emitted.
Proof (`make asm`): float build has `mulss`, int8 build has `movsbl`+`imull`.
**Zero shared arithmetic instructions.** One code path costs the reader nothing,
the CPU nothing.

**Why can't `forward()` be a virtual template?** The vtable is a fixed-size array
built at compile time. Templates instantiate on demand → unbounded slots → the
compiler can't lay it out. So template the *class*, not the method.

**Result:** float vs int8 agreed 3/3, max error 1.26 pp, weights 456 B → 114 B.

---

### Day 7: The Golden Model ✅ 🔑

**The bug that was hiding:** day4 (C) and day6 (C++) both called `srand(42)` —
and still computed different networks.
```c
day4:  ((rand()%100)/100.0f - 0.5f) * 0.1f     bias {0.1, -0.1, 0}
day6:  (rand()/RAND_MAX - 0.5f) * 2.0f * scale bias {0, 0, 0}
```
> **Same seed ≠ same numbers if the formula differs.**
> `srand(42)` only makes a program reproducible against *itself*.

**The fix: stop generating, start loading.**
```text
        weights.bin  ← one file, one truth
             │
   NumPy ─── C ─── C++ ─── int8 ─── VHDL     everyone LOADS these bytes
```
Once weights live in a file, the generator stops mattering. NumPy can use PCG64
that C could never reproduce — because C never reproduces it. C *reads* it.

**You built a model file format.** That's what ONNX/TFLite ARE.

| Field | Catches |
|---|---|
| magic `0x574E4E43` | a JPEG / wrong file — fails on byte 0, not later |
| version | a future file fed to an old loader |
| checksum | brownout-truncated flash write (parses fine, predicts garbage) |
| 4-byte alignment | **zero-copy** `const float*` into the buffer |

> Unaligned float load: *slow* on x86, **faults** on some ARM cores.
> That's why TFLite (FlatBuffers) is alignment-strict — it's built to be mmap'd.

**One loader, two languages:** `model_io.h` is header-only C that's also valid
C++. Both engines include it → the comparison is between *inference engines*,
not between two parsers that might differ.

**⭐ VERIFY EVERY LAYER, not just the prediction:**
```text
TENSOR 0 conv 128 → TENSOR 0 relu 128 → ... → TENSOR 0 softmax 3
```
If only `softmax` disagrees, the bug could be in any of 6 layers. Layer-by-layer
localizes it: **the first diverging layer is where the bug lives.**
> This is exactly how you debug RTL against a C reference model. Day 7 is
> rehearsal for Phase 1 hardware verification.

**Tolerance, not equality:** deviation was `1e-10` (tolerance `1e-4`).
`expf()` vs NumPy `exp()` vs fused multiply-add round differently. Demanding
bit-identity across languages would be *wrong*. 1e-10 = agreement.

**VHDL bridge:** `vhdl_vectors/` holds int8 images, weights, and expected int32
conv accumulators. **Peak accumulator = 48,387** — Day 5's overflow rule
*measured*: needs 17 bits signed, so int8 (127) and even int16 overflow. int32 it is. Your August testbench
compares against these exact files.

---

## The Complete Algorithm (5 Steps)

```text
1. FORWARD:  input → layers → prediction      (Day 1)
2. LOSS:     prediction vs target → number     (Day 2)
3. BACKWARD: loss → gradients for all weights  (Day 2)
4. UPDATE:   weight -= lr × gradient           (Day 2)
5. REPEAT until loss is small                  (Day 3)

THAT'S ALL OF DEEP LEARNING.
Everything else is optimization.
```

## The Deployment Pipeline (5 Steps)

```text
1. TRAIN in Python (float32, big GPU)          Phase 2
2. COMPRESS: quantize (int8)                   Day 5 ✅ / Phase 2
3. EXPORT: weights → a file format             Day 7 ✅ (ONNX/TFLite = same idea)
4. DEPLOY: C / C++ / ARM NEON                  Days 4,6 ✅ / Phase 2
5. ACCELERATE: VHDL on FPGA fabric             Phase 1 ← the differentiator
```

> The technical focus is cross-implementation verification and hardware design.

---

## The technical goal

```text
ONE CNN. SIX IMPLEMENTATIONS. ONE GOLDEN MODEL. EVERY OUTPUT VERIFIED.

  NumPy → C → C++ → int8 → ARM NEON → VHDL
     └──────── all bit-verified against weights.bin ────────┘
```

Every applicant has trained a model. Almost none can show the same network hand-built
at six levels of abstraction with bit-exact agreement between Python and hand-written RTL.

**The single thread running through all of it:**

```text
int8 × int8 overflows → you must accumulate in int32

  Day 5   a comment you remember          (C)
  Day 6   a type the compiler enforces    (C++ traits)
  Day 7   48,387 — measured on real data  (golden model)
  Phase 1 a 32-bit register               (VHDL)
```
