# Lesson 4: implement the forward pass in C

[Previous: MNIST](../day3/README.md) · [All lessons](../README.md) ·
[Next: integer arithmetic](../day5/README.md)

Goal: translate tensor operations into loops and explicit memory buffers.
You need basic C pointers, arrays, and allocation.

## Files and commands

| File | Purpose |
|---|---|
| [conv2d.c](conv2d.c) | Start with one convolution operation |
| [cnn_forward.c](cnn_forward.c) | Connect convolution, ReLU, pooling, Dense, and softmax |
| [Makefile](Makefile) | Build both programs |

From the repository root:

```bash
make -C lessons/day4
./lessons/day4/conv2d
./lessons/day4/cnn_forward
```

Expect printed feature maps and forward-pass results. This is an inference
exercise, not training on MNIST.

## Understand the indexing

C stores the tensor in a flat array. For channel-first data:

```text
index(c, y, x) = c × height × width + y × width + x
```

For a tensor with height 4 and width 5, channel 1, row 2, column 3 is at
`1×4×5 + 2×5 + 3 = 33`, using zero-based indices.

The allocation must hold every element the loops can access. Track the shape
alongside each buffer, and match each owned allocation with a release.

## What to compare

Use a small hand-computable input to compare indices and padding with Lesson 1.
Do not infer exact cross-language agreement from similar-looking probabilities:
earlier experiments can use different weights and random initialization.

Checkpoint: how many float32 bytes are required for a `2×4×5` tensor?
Answer: `2×4×5×4 = 160` bytes, excluding any metadata.

The reusable C kernels now live in [engine/cnn_c.h](../../engine/cnn_c.h).
[Lesson 7](../day7/README.md) supplies shared weights and automatic comparisons.
