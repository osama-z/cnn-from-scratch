# 📘 Day 1 — Convolution & the CNN Forward Pass

> **What this day proves:** an image is just numbers, and "seeing" is just
> multiply-and-add. Every number below was produced by running the code in this
> folder — nothing here is illustrative.

---

## The files

| File | Lines | What it builds | Run it |
|---|---|---|---|
| `part1_convolution.py` | 341 | 2D convolution from scratch + a kernel zoo | `python part1_convolution.py` |
| `part2_cnn_building_blocks.py` | 590 | ReLU, pooling, stride, multi-channel conv | `python part2_cnn_building_blocks.py` |
| `part3_full_cnn_forward.py` | 504 | The complete chain, end to end | `python part3_full_cnn_forward.py` |

Each writes a `*_results.png` next to itself.

---

## 1. An image is a matrix

The test image in `part1` — a 7×7 grid with a bright 3×3 square in the middle:

```text
 10  10  10  10  10  10  10
 10  10  10  10  10  10  10
 10  10 255 255 255  10  10
 10  10 255 255 255  10  10
 10  10 255 255 255  10  10
 10  10  10  10  10  10  10
 10  10  10  10  10  10  10
```

`10` = dark, `255` = bright. That's the whole idea. There is no "image" in memory —
only intensities.

---

## 2. Convolution — the one operation

```python
for y in range(ih):
    for x in range(iw):
        region = padded[y:y+kh, x:x+kw]     # the patch under the kernel
        output[y, x] = np.sum(region * kernel)   # ← this line IS convolution
```

**In words:** at every pixel, take the small neighbourhood around it, multiply
element-by-element with the kernel, and add it all up. One number out.

### Worked example — Sobel X at position (2,2)

The 3×3 window at that position (after zero-padding), and the kernel:

```text
window            kernel            product
 10  10  10      -1   0   1        -10   0  10     →   0
 10 255 255   ×  -2   0   2   =    -20   0 510     → 490
 10 255 255      -1   0   1        -10   0 255     → 245
                                                     ─────
                                            total =   735
```

Run the code and `conv[2,2]` is exactly **735.0**.

> ⚠️ **Bug in this file — see §7.** The printed diagram claims 980 and the
> self-check prints `Match: NO ✗`. The code is right; the explanation is wrong.

### The full Sobel-X output

```text
  30    0    0    0    0    0  -30
  40  245  245    0 -245 -245  -40
  40  735  735    0 -735 -735  -40
  40  980  980    0 -980 -980  -40      ← strongest response
  40  735  735    0 -735 -735  -40
  40  245  245    0 -245 -245  -40
  30    0    0    0    0    0  -30
```

**Read this output — it teaches three things at once:**

1. **Positive on the left edge, negative on the right.** Sobel X computes
   *right minus left*. Dark→bright gives `+`, bright→dark gives `−`. The **sign
   carries the direction** of the edge.
2. **Zero down the middle column.** At x=3 the window is symmetric — bright on
   both sides — so left and right cancel. **No change means no response.**
3. **±40 in column 0 and 6.** These are padding artifacts, not real edges. The
   zero-pad invents a dark border, which looks like an edge to the filter. Every
   CNN has this; it's why very deep networks care about padding mode.

---

## 3. The kernel zoo — why values determine behaviour

| Kernel | Values | Sum | Detects |
|---|---|---|---|
| Identity | `0 0 0 / 0 1 0 / 0 0 0` | 1 | nothing — copies input |
| Box blur | all `1/9` | 1 | smooths, kills noise |
| Sobel X | `-1 0 1 / -2 0 2 / -1 0 1` | **0** | vertical edges |
| Sobel Y | `-1 -2 -1 / 0 0 0 / 1 2 1` | **0** | horizontal edges |
| Sharpen | `0 -1 0 / -1 5 -1 / 0 -1 0` | 1 | boosts local contrast |

### ⭐ The kernel-sum rule

```text
sum = 0   →  edge detector      flat regions cancel to zero; only CHANGE survives
sum = 1   →  feature preserver  average brightness unchanged
sum > 1   →  amplifier          image gets brighter
```

**Why sum = 0 detects edges** — take the flat corner of the image, all `10`:

```text
(10×-1)+(10×0)+(10×1) = 0
(10×-2)+(10×0)+(10×2) = 0
(10×-1)+(10×0)+(10×1) = 0     total = 0   ← nothing happening here
```

The weights cancel *because they sum to zero*. Only where the numbers differ does
anything survive. **Nothing is learned here — the arithmetic simply does it.**

---

## 4. Padding — why the output stays 7×7

```python
pad_h = kh // 2          # 3//2 = 1
padded = np.pad(image, ((1,1),(1,1)), constant_values=0)
```

Without padding, a 3×3 kernel on a 7×7 image gives 5×5 — you lose a ring every
layer. Stack 10 layers and a 32×32 image is gone.

| Mode | Border becomes | Cost |
|---|---|---|
| `constant` (zero) | black | invents a fake edge — the ±40 column |
| `reflect` | mirrored | no fake edge, slightly more work |
| `edge` | nearest pixel copied | no fake edge |

Your Day 4 C, Day 6 C++ and Day 7 golden model all use **zero padding**, which is
why they agree — the convention must match everywhere.

---

## 5. `part2` — the rest of the building blocks

| Function | Formula | Why it exists |
|---|---|---|
| `relu(x)` | `max(0, x)` | without a non-linearity, stacked layers collapse to one matrix |
| `leaky_relu(x, α)` | `x if x>0 else αx` | fixes "dying ReLU" — dead units never recover |
| `max_pool_2d` | max of each 2×2 | "was the feature anywhere here?" |
| `avg_pool_2d` | mean of each 2×2 | smoother, dilutes strong responses |
| `convolution_2d_stride` | step by `s` | downsample *while* convolving |
| `conv2d_multichannel` | sum over input channels | each filter spans **all** channels |

**Why ReLU and not sigmoid?** Gradient is exactly 1 for x>0 (no vanishing), it's a
single compare (no `exp()` — critical on an MCU), and ~half the outputs become
exactly 0, which is cheap.

**Why max and not average pooling?** You're asking "did this feature appear
anywhere in this region?" That's a max. Averaging dilutes one strong response
among three weak ones.

**Multi-channel is the part people get wrong:**

```text
Each filter spans ALL input channels and produces ONE output channel.

  input (3, H, W)  +  filter (3, 3, 3)  →  ONE output channel
  8 filters                             →  output (8, H, W)
```

The filter count sets output depth; input depth is consumed entirely by each filter.

---

## 6. `part3` — the full chain

```text
 8×8 image → Conv2D(1→2) → ReLU → MaxPool(2) → Flatten → Dense(32→3) → Softmax
   64 vals     128 vals     128       32          32          3           3
```

Real output from the script:

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

Mathematically the `- max` cancels in the ratio. Numerically it is essential:
`exp(100)` overflows float32 (max 3.4e38) → `inf` → `inf/inf` = **NaN**.

> This exact line becomes a finding on Day 7: the verification suite could not
> detect its removal, because the test images never produced logits large enough
> to overflow. See `day7/README_explanation.md` §6b.

### ⚠️ The predictions are meaningless — and that's correct

The script says so itself: *"The predictions are RANDOM because we haven't trained
the network."* The conv kernels are hand-designed, but `Dense` is random, so it
maps good features to arbitrary classes. Day 1 proves the **shapes and arithmetic**
flow correctly. Training is Day 2.

### Scale check — why this matters later

| Model | Params | FLOPs/frame |
|---|---|---|
| **This one** | 915 | 14,944 |
| YOLO-Nano | ~1.9 M | ~4 **billion** |

That's a factor of ~270,000× in compute. **This is the entire reason Days 5–7 and
the FPGA phase exist** — real detection does not fit on a drone without
quantization and hardware acceleration.

---

## 7. 🐛 Known issue in `part1_convolution.py`

**The hand-trace is wrong, and the file's own self-check reports it:**

```text
Our code computed: 735.0
Hand calculation:  980.0
Match: NO ✗
```

**Cause (lines 213–227):** the printed diagram shows this window at (2,2)—

```text
 10  10 255
 10 255 255
 10 255 255      → 245 + 490 + 245 = 980
```

—but that window **does not occur anywhere in the image**. I searched all 49
positions. The real window at (2,2) has a top row of `10 10 10`, giving
`0 + 490 + 245 = 735`.

**Fix:** redraw the diagram with the true window and change 980 → 735. The
arithmetic in the file is correct; only the explanation is wrong.

*(Minor: lines 79–82 contain a duplicated two-line comment — a copy-paste artifact.)*

**Why this is worth fixing rather than ignoring:** the file prints `NO ✗` every
run. Anyone reviewing your repo sees a failing self-check on the first script.

---

## ✅ Self-test

1. Why does a kernel whose values sum to zero detect edges?
2. In the Sobel-X output, why is the middle column exactly 0?
3. What are the ±40 values in columns 0 and 6, and are they real edges?
4. Why pad at all? What breaks after 10 unpadded layers?
5. If input is `(3, H, W)` and you apply 8 filters, what is the output shape?
6. Why does `softmax` subtract the max if it cancels out mathematically?
7. Why are the predicted classes meaningless in `part3`?

<details><summary>Answers</summary>

1. Flat regions cancel to zero; only *changes* survive.
2. The window is symmetric there — equally bright both sides, so left and right cancel.
3. Padding artifacts. Zero-padding invents a dark border that looks like an edge. Not real.
4. A 3×3 kernel loses one ring per layer; 10 layers costs 20 pixels of width and height.
5. `(8, H, W)` — each filter consumes all 3 input channels and produces 1 output channel.
6. It cancels mathematically but not numerically: `exp(100)` = `inf`, then `inf/inf` = `NaN`.
7. `Dense` is randomly initialized and never trained, so good features map to arbitrary classes.

</details>

---

**Next:** `day2/` — loss functions, backpropagation, and the optimizers that turn
those random Dense weights into something meaningful.
