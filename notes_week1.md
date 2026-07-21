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

**Train/Test Split:** Train on 60K, test on 10K unseen images.
If train acc >> test acc → **Overfitting** (memorized, not learned).

**Result:** 784→128→64→10 dense net, **95% test accuracy**.

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

### Day 6-7: Benchmark Project ⬜

**Goal:** Run same convolution in 3 ways, measure everything:

| Version | Size | Speed | Accuracy |
|---------|------|-------|----------|
| Python float32 | ~50MB | baseline | 95.0% |
| C float32 | ~20KB | 10-50x faster | 95.0% |
| C int8 | ~5KB | 50-100x faster | ~93.5% |

This table goes in your research paper.

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
1. TRAIN in Python (float32, big GPU)          (Months 1-2)
2. COMPRESS: quantize + prune + distill        (Month 3)
3. EXPORT: PyTorch → ONNX → TFLite            (Month 3)
4. DEPLOY: TFLite on ARM with CMSIS-NN         (Month 4)
5. FLY: ARM on drone with camera               (Month 5)
```
