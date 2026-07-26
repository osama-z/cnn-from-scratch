# 👁️ Day 1: The Visual Cortex (Forward Pass)

> **Core Concept:** an image is just numbers, and "seeing" is just multiply-and-add.
> Data flows one direction only: Input → Output. No learning yet.
>
> Every number in this file was produced by running the code in this folder.

---

## 0. The files

| File | Lines | What it builds | Run |
|---|---|---|---|
| `part1_convolution.py` | 355 | 2D convolution from scratch + a kernel zoo | `python part1_convolution.py` |
| `part2_cnn_building_blocks.py` | 590 | ReLU, pooling, stride, multi-channel conv | `python part2_cnn_building_blocks.py` |
| `part3_full_cnn_forward.py` | 504 | The complete chain, end to end | `python part3_full_cnn_forward.py` |

Each writes a `*_results.png` beside itself.

---

## 1. The architecture

```text
[Input Image]  8×8 = 64 numbers
      │
      ▼
[Convolution] ───► slides a 3×3 kernel over every pixel to detect edges
      │            1 channel in → 2 channels out = 128 values
      ▼
[ReLU]        ───► f(x) = max(0, x)
      │            blocks negatives; without it, stacked layers collapse into one
      ▼
[Max Pooling] ───► keeps the largest value in each 2×2 block
      │            halves the size: 128 → 32 values
      ▼
[Flatten]     ───► reinterprets the 2×4×4 grid as a flat vector of 32
      │            no arithmetic — just a change of view
      ▼
[Dense]       ───► (inputs · weights) + bias, fully connected
      │            32 → 3 scores
      ▼
[Softmax]     ───► eˣ / Σeˣ  → 3 probabilities summing to 1.0
```

---

## 2. Convolution — the one operation

```python
for y in range(ih):
    for x in range(iw):
        region = padded[y:y+kh, x:x+kw]          # the patch under the kernel
        output[y, x] = np.sum(region * kernel)   # ← this line IS convolution
```

At every pixel: take the neighbourhood, multiply element-by-element with the
kernel, sum. One number out.

### The test image (`part1`)

```text
 10  10  10  10  10  10  10
 10  10  10  10  10  10  10
 10  10 255 255 255  10  10
 10  10 255 255 255  10  10      10 = dark, 255 = bright
 10  10 255 255 255  10  10
 10  10  10  10  10  10  10
 10  10  10  10  10  10  10
```

### Worked example — Sobel X at (2,2)

The window comes from the **padded** image, so the row above the bright square is
still all-dark:

```text
window            kernel            row totals
 10  10  10      -1   0   1        -10 +   0 +  10 =   0   ← all dark: cancels
 10 255 255   ×  -2   0   2   =    -20 +   0 + 510 = 490
 10 255 255      -1   0   1        -10 +   0 + 255 = 245
                                                     ─────
                                             total =  735
```

Run the code: `conv[2,2]` is exactly **735.0**, and the script's self-check prints
`Match: YES ✓`.

> **Why the first row contributes 0.** Sobel X computes *right minus left*. A row
> that is uniformly dark cancels itself. Only rows that actually **change** across
> the kernel contribute anything.

### The full Sobel-X output — read it, it teaches three things at once

```text
  30    0    0    0    0    0  -30
  40  245  245    0 -245 -245  -40
  40  735  735    0 -735 -735  -40
  40  980  980    0 -980 -980  -40      ← strongest response
  40  735  735    0 -735 -735  -40
  40  245  245    0 -245 -245  -40
  30    0    0    0    0    0  -30
```

1. **Positive left, negative right.** The **sign carries the direction** of the
   edge. Dark→bright gives `+`, bright→dark gives `−`.
2. **The middle column is exactly 0.** At x=3 the window is symmetric — equally
   bright both sides — so left and right cancel. **No change, no response.**
3. **±40 in columns 0 and 6 are fake.** Zero-padding invents a dark border, which
   looks like an edge. Every CNN has this artifact; it's why padding mode matters.

**Why 980 is the peak:** at row 3 all three kernel rows straddle the edge. At row 2
only two of them do, which is why it's 735.

---

## 3. The kernel zoo — values determine behaviour

| Kernel | Values | Sum | Detects |
|---|---|---|---|
| Identity | `0 0 0 / 0 1 0 / 0 0 0` | 1 | nothing — copies the input |
| Box blur | all `1/9` | 1 | smooths, kills noise |
| Sobel X | `-1 0 1 / -2 0 2 / -1 0 1` | **0** | vertical edges |
| Sobel Y | `-1 -2 -1 / 0 0 0 / 1 2 1` | **0** | horizontal edges |
| Sharpen | `0 -1 0 / -1 5 -1 / 0 -1 0` | 1 | boosts local contrast |

### ⭐ The kernel-sum rule

```text
sum = 0   →  edge detector      flat regions cancel; only CHANGE survives
sum = 1   →  feature preserver  average brightness unchanged
sum > 1   →  amplifier          image gets brighter
```

Take the flat corner, all `10`:

```text
(10×-1) + (10×0) + (10×1) = 0
(10×-2) + (10×0) + (10×2) = 0     total = 0  ← nothing happening here
(10×-1) + (10×0) + (10×1) = 0
```

The weights cancel **because they sum to zero**. Nothing is learned — the
arithmetic simply does it.

> **In real CNNs these kernels are not hardcoded.** They start random and the
> network *learns* the values during training (Day 2). Hand-designing them here
> proves the mechanism before adding learning on top.

**Why the middle row of Sobel is `-2 0 2`, not `-1 0 1`:** the centre row is
closest to the pixel being computed, so it gets double weight. It's a weighted
edge detector, which is less noise-sensitive than a plain difference.

---

## 4. Padding — why the output stays 7×7

```python
pad_h = kh // 2                      # 3 // 2 = 1
padded = np.pad(image, ((1,1),(1,1)), constant_values=0)
```

Without padding, a 3×3 kernel on 7×7 gives 5×5 — you lose a ring every layer.
Ten layers and a 32×32 image is gone.

| Mode | Border becomes | Cost |
|---|---|---|
| `constant` (zero) | black | invents a fake edge — the ±40 column |
| `reflect` | mirrored | no fake edge, slightly more work |
| `edge` | nearest pixel repeated | no fake edge |

> **This convention must match everywhere.** Day 4 (C), Day 6 (C++) and Day 7
> (golden model) all use zero padding — which is *why* they agree to 1.2e-10.

---

## 5. `part2` — the remaining building blocks

| Function | Formula | Why it exists |
|---|---|---|
| `relu(x)` | `max(0, x)` | without non-linearity, stacked layers collapse |
| `leaky_relu(x, α)` | `x if x>0 else αx` | fixes "dying ReLU" — dead units never recover |
| `max_pool_2d` | max of each 2×2 | "was the feature anywhere here?" |
| `avg_pool_2d` | mean of each 2×2 | smoother, dilutes strong responses |
| `convolution_2d_stride` | step by `s` | downsample *while* convolving |
| `conv2d_multichannel` | sum over input channels | each filter spans **all** channels |

**Why stacking linear layers is pointless without ReLU:**

```text
Layer2(Layer1(x)) = W₂(W₁x) = (W₂W₁)x = W₃x     ← one layer, not two
```

Ten linear layers collapse algebraically into one matrix. Depth would buy nothing.

**Why ReLU and not sigmoid?** Gradient is exactly 1 for x>0 (no vanishing), it's a
single compare with no `exp()` (critical on an MCU), and ~half the outputs become
exactly 0, which is cheap.

**Why max and not average pooling?** You're asking *"did this feature appear
anywhere in this region?"* — that's a max. Averaging dilutes one strong response
among three weak ones.

**Multi-channel — the part people get wrong:**

```text
Each filter spans ALL input channels and produces ONE output channel.

  input (3, H, W)  +  one filter (3, 3, 3)  →  ONE output channel
  8 filters                                 →  output (8, H, W)
```

Filter count sets output depth. Input depth is consumed entirely by each filter.

---

## 6. `part3` — the full chain

```text
 8×8 image → Conv2D(1→2) → ReLU → MaxPool(2) → Flatten → Dense(32→3) → Softmax
   64 vals     128 vals    128        32          32          3           3
```

Real output:

```text
  Predicted class: Horizontal Edge (100.0%)
  This tiny model: 915 parameters = 3.6 KB (float32)
  Our tiny model:  14,944 FLOPs per image
```

### Softmax — and the one detail that matters

```python
e = np.exp(x - x.max())      # ← subtract the max
return e / e.sum()
```

The `- max` cancels mathematically in the ratio. Numerically it is essential:
`exp(100)` overflows float32 (max 3.4e38) → `inf` → `inf/inf` = **NaN**.

> This exact line becomes a finding on Day 7: the verification suite could not
> detect its removal, because the test images never produced logits large enough
> to overflow. See `day7/README_explanation.md` §6b.

### ⚠️ The predictions are meaningless — and that is correct

The script says so itself: *"The predictions are RANDOM because we haven't trained
the network."* The conv kernels are hand-designed, but `Dense` is randomly
initialized, so it maps good features to arbitrary classes.

**Day 1 proves the shapes and arithmetic flow correctly.** Making the answers
*right* is Day 2's job.

### Scale check — why the rest of the roadmap exists

| Model | Params | FLOPs/frame |
|---|---|---|
| **This one** | 915 | 14,944 |
| YOLO-Nano | ~1.9 M | ~4 **billion** |

A factor of ~270,000× in compute. **This single comparison justifies Days 5–7 and
the whole FPGA phase** — real detection does not fit on a drone without
quantization and hardware acceleration.

---

## 7. Where this gets used, and why

* **Phase:** inference (production)
* **Hardware:** edge devices — drones, phones, Raspberry Pi, MCUs
* **Why it's the deployable part:** the forward pass needs no gradients and no
  history. It is pure multiply-and-add, which is why it runs 60×/second on a
  drone — and why it is the only part that maps cleanly onto FPGA logic.

> When you deploy, you **strip away Days 2 and 3** and ship only Day 1's math.
> Training happens once, on a big machine. Inference happens forever, on a small one.

---

## ✅ Self-test

1. Why does a kernel whose values sum to zero detect edges?
2. In the Sobel-X output, why is the middle column exactly 0?
3. What are the ±40 values in columns 0 and 6 — real edges or not?
4. Why is the response 735 at (2,2) but 980 at (3,1)?
5. Why pad at all? What breaks after 10 unpadded layers?
6. Input is `(3, H, W)` and you apply 8 filters — what is the output shape?
7. Why does softmax subtract the max if it cancels out mathematically?
8. Why are `part3`'s predicted classes meaningless?

<details><summary>Answers</summary>

1. Flat regions cancel to zero; only *changes* survive.
2. The window is symmetric there — equally bright both sides — so left and right cancel.
3. Not real. Zero-padding invents a dark border that looks like an edge.
4. At (3,1) all three kernel rows straddle the edge; at (2,2) only two do — the top row is uniformly dark and contributes 0.
5. A 3×3 kernel loses one ring per layer; 10 layers costs 20 pixels of width and height.
6. `(8, H, W)` — each filter consumes all 3 input channels and produces 1 output channel.
7. It cancels mathematically but not numerically: `exp(100)` = `inf`, then `inf/inf` = `NaN`.
8. `Dense` is randomly initialized and never trained, so good features map to arbitrary classes.

</details>

---

**Next:** `day2/` — loss functions, backpropagation, and the optimizers that turn
those random Dense weights into something that actually works.
