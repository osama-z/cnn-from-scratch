# Lesson 1: follow a CNN forward pass

[All lessons](../README.md) · [Next: learning](../day2/README.md)

Goal: understand how an image becomes class scores before introducing training.
You need Python loops and NumPy arrays.

## Read and run in order

Commands run from the repository root with the lesson dependencies installed.

| File | What to look for |
|---|---|
| [part1_convolution.py](part1_convolution.py) | A sliding 3×3 window, padding, and different filters |
| [part2_cnn_building_blocks.py](part2_cnn_building_blocks.py) | Stride, ReLU, pooling, and multiple channels |
| [part3_full_cnn_forward.py](part3_full_cnn_forward.py) | Connect the operations in a small CNN |

```bash
(cd lessons/day1 && python part1_convolution.py)
(cd lessons/day1 && python part2_cnn_building_blocks.py)
(cd lessons/day1 && python part3_full_cnn_forward.py)
```

The programs print intermediate results and save plots in this folder. Existing
plots show [filters](part1_results.png), [building blocks](part2_results.png),
and [the full pass](part3_results.png).

## Work through one value

For a window filled with ones and a 3×3 filter also filled with ones, the output
is nine plus the bias. The nine products use the same filter weights at every
image position. At a zero-padded boundary, fewer real pixels contribute.

The next operations have different jobs:

```text
convolution → ReLU → pooling → flatten → Dense → softmax
local features  │    smaller maps   │     scores   probabilities
               │                  └─ preserve values, change shape
               └─ replace negative values with zero
```

For a 2×2 pooling window containing `[1, 4; 2, 3]`, max pooling returns 4.
Flattening only changes layout; it does not learn weights.

## What the result means

A complete forward pass can run with arbitrary weights. Class probabilities
alone do not prove that it has learned to classify images. These programs teach
the operations; the learned convolutional classifier lives in
[trained/](../../trained/README.md).

Checkpoint: with two 8×8 feature maps and 2×2 pooling at stride 2, how many values
reach Dense? Answer: `2 × 4 × 4 = 32`.

For the 28×28 trained model's shapes and a worked convolution, read the
[main guide](../../docs/guide.md#2-follow-one-image).
