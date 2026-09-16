# Understand the CNN, one step at a time

[Documentation index](README.md) · [Setup and commands](setup.md) ·
[Formula reference](reference/formulas.md)

This guide explains the network that actually trains in this repository.
You need basic Python, arrays, multiplication, and addition. Derivatives help
with the training section, but you can understand the forward pass first.

## Contents

1. [Identify the model](#1-identify-the-model)
2. [Follow one image](#2-follow-one-image)
3. [Understand training](#3-understand-training)
4. [Understand deployment](#4-understand-deployment)
5. [Understand the hardware work](#5-understand-the-hardware-work)
6. [Check your understanding](#6-check-your-understanding)

## 1. Identify the model

There are several experiments here. Do not mix their results.

| Location | What it is | Main purpose |
|---|---|---|
| [Lesson 3](../lessons/day3/README.md) | Dense network, 784 → 128 → 64 → 10 | Learn training on MNIST |
| [Lesson 7](../lessons/day7/README.md) | Small, fixed 8×8-image CNN | Compare implementations using known weights |
| [trained/](../trained/README.md) | Learned 28×28-image CNN | Train and deploy a useful classifier |
| [hardware/vhdl/](../hardware/vhdl/README.md) | Integer arithmetic and convolution blocks | Verify hardware building blocks |

The rest of the software explanation refers to `trained/`. Its model is:

```text
image → convolution → ReLU → max pooling → flatten → Dense → softmax
```

The implementation is in [model.py](../trained/model.py). During inference the
weights stay fixed. During training both convolution and Dense weights change.

## 2. Follow one image

An MNIST image has 28 rows and 28 columns, with one grayscale channel. The loader
divides pixel bytes by 255, giving input values between 0 and 1.

The code uses **NCHW** order: batch size, channels, height, width.
A batch of 64 images therefore has shape `(64, 1, 28, 28)`.

For one image, the default architecture has these shapes:

| Stage | Output shape, excluding batch | What changed? |
|---|---|---|
| Input | 1 × 28 × 28 | Normalized grayscale pixels |
| Convolution | 4 × 28 × 28 | Four learned feature maps |
| ReLU | 4 × 28 × 28 | Negative values become zero |
| Max pooling | 4 × 14 × 14 | Each 2×2 region becomes one value |
| Flatten | 784 | Same values, viewed as a vector |
| Dense | 10 | One raw score per digit |
| Softmax | 10 | Scores become probabilities |

### Convolution: reuse a small set of weights

A 3×3 filter looks at one local window. Multiply matching entries, add the
products, then add a bias. Slide the filter to the next position and repeat.

Here is an illustrative window and filter, not learned MNIST weights:

```text
window           filter
0  1  1         -1  0  1
0  1  1         -1  0  1
0  1  1         -1  0  1

output = 0×(-1) + 1×0 + 1×1   ← first row
       + 0×(-1) + 1×0 + 1×1   ← second row
       + 0×(-1) + 1×0 + 1×1   ← third row
       + bias
       = 3 + bias
```

Each filter produces one feature map. With four filters, there are four output
channels. The same filter weights are reused at every position; this is why a
convolution needs relatively few parameters.

Here stride is 1 and a one-pixel zero border supplies **same padding**, preserving
28×28 spatial dimensions. At a boundary, some window entries are padding zeros.
The operation commonly called convolution in neural networks is technically
cross-correlation: this implementation does not flip the filter.

Start with [Lesson 1](../lessons/day1/README.md), then find `conv_forward` in
[model.py](../trained/model.py).

### ReLU: keep positive activations

ReLU means `max(0, x)`:

```text
[-2, 0, 3] → [0, 0, 3]
```

It adds nonlinearity. Without nonlinear operations, stacked affine layers could
be combined into a single affine transformation. ReLU does not guarantee that
gradients never vanish: its negative side has zero gradient.

### Pooling: reduce spatial size

Max pooling keeps the largest value in each 2×2 window, using stride 2:

```text
1  4
2  3    → 4
```

Thus 28×28 becomes 14×14. It reduces the amount of data passed onward and can
make features less sensitive to small position changes, but does not guarantee
translation invariance.

Training remembers which entry won. Only that entry receives the window's
backward gradient. This implementation selects the first maximum in a tie.

### Flatten: change the view, not the values

The pooled tensor contains `4 × 14 × 14 = 784` numbers. Flattening turns it into
a vector of length 784 without learning or discarding anything.

The order matters: NumPy, C, and C++ must flatten channels, rows, and columns in
the same order, or Dense weights will be paired with the wrong inputs.

### Dense: combine the detected features

For class `j`, the raw score is:

```text
score[j] = bias[j] + sum(flattened[i] × weight[i, j])
```

The Dense weight shape is `(784, 10)`. Its ten output scores are called
**logits**. They need not be positive or sum to one.

### Softmax: normalize the scores

Softmax exponentiates the logits and divides by their total. The code first
subtracts the largest logit to avoid unnecessarily large exponentials.

For a three-class illustration:

```text
logits:            [2,     1,     0]
subtract maximum:  [0,    -1,    -2]
exponentiate:      [1,     0.368, 0.135]
normalize:         [0.665, 0.245, 0.090]
```

The largest probability determines the predicted class. A value of 0.90 is not
proof that the model is correct or that its confidence is well calibrated.

### Count the learned parameters

| Parameter | Count |
|---|---:|
| Convolution weights: 4 × 1 × 3 × 3 | 36 |
| Convolution biases | 4 |
| Dense weights: 784 × 10 | 7,840 |
| Dense biases | 10 |
| Total | 7,890 |

At four bytes per float32, the parameter payload is 31,560 bytes. The exported
file is 31,720 bytes because it also contains names, shapes, and a header.

## 3. Understand training

Inference asks, “What digit does this image look like?”
Training asks, “How should the weights change so labeled images get better scores?”

```text
images + labels
      ↓
forward pass → loss → backward pass → Adam update
      ↑                                    │
      └──────── next batch, new weights ────┘
```

### Loss: one number describing the current mistake

Cross-entropy penalizes a low probability for the correct label:

```text
loss = -log(probability assigned to the correct class)
```

If the correct class receives probability 0.8, the loss is about 0.223.
At probability 0.1, it is about 2.303. Training minimizes the average across the
batch. The implementation computes this stably from logits.

Loss and accuracy are different: accuracy only counts whether the largest score
belongs to the correct class; loss also changes when the scores change without
changing that winning class.

### Gradients: local sensitivity

A gradient tells us how a small change to a value would affect the loss.
It is a derivative, not a “percentage of blame.”

Basic gradient descent uses:

```text
new_weight = old_weight - learning_rate × gradient
```

This repository uses Adam, which maintains moving averages of gradients and
squared gradients to scale updates. Its update logic is explicit in
[model.py](../trained/model.py); there is no autograd engine.

### Backpropagation: work backward through the same layers

For averaged softmax cross-entropy, the starting derivative is:

```text
d_logits = (probabilities - one_hot_labels) / batch_size
```

Then:

1. Dense computes weight/bias gradients and sends gradients to its input.
2. Flatten restores the pooled tensor shape.
3. Pooling routes each gradient to the winning input position.
4. ReLU blocks gradients where its forward input was not positive.
5. Convolution accumulates gradients for its weights, biases, and inputs.

The forward pass caches values needed by the backward pass. The convolution
weights really learn; this is not just a trained classifier on fixed filters.

The [tests](../tests/test_trained.py) compare analytical gradients with numerical
changes in loss after small parameter perturbations. This checks the backward
math independently of whether a training curve looks plausible.

### Batches, epochs, and data splits

A **batch** is one group used for a gradient update. An **epoch** is one pass
through the chosen training set.

The default experiment separates three roles:

| Split | Default size | Used for |
|---|---:|---|
| Training | 10,000 | Computing gradients and changing weights |
| Validation | 2,000 | Selecting the best epoch's checkpoint |
| Test | 10,000 | Evaluating the selected checkpoint |

Training and validation are disjoint subsets of the training archive. Test
images come from the separate official test archive. Choosing settings repeatedly
based on test results would undermine that separation.

See [train.py](../trained/train.py) for the orchestration and
[setup](setup.md#3-download-mnist-and-train) for commands.

## 4. Understand deployment

Training needs gradients and an optimizer. Inference only needs the forward
operations and the learned parameters.

```text
trained/model.py + trained/export.py
                  ↓
        engine/model_format.py
                  ↓
              weights.bin
                  ↓
          engine/model_io.h
             ↙          ↘
    engine/cnn_c.h     engine/cnn.hpp
         C                 C++
```

The Python writer stores tensor names, shapes, and float32 values. The shared
native reader validates the file. C and C++ then use separate arithmetic
implementations to run the same architecture.

The [format specification](reference/model-format.md) describes the bytes.
The file is a parameter container, not a general graph format: the runners still
define the layer sequence.

### Agreement is not accuracy

Two implementations can agree perfectly and both classify badly. Therefore:

- **Accuracy** compares predictions with ground-truth labels.
- **Deployment agreement** compares intermediate outputs and predictions between
  implementations.
- **Robustness** checks that invalid inputs fail safely.

The trained verification bundle contains 32 exported test examples by default.
Agreement on these examples does not claim numerical verification over every
possible input. Full test-set accuracy is reported separately.

Floating-point accumulation order can differ between NumPy, C, and C++.
Verification uses stated absolute and relative tolerances rather than requiring
bit-for-bit equality. Read [verification](verification.md) before interpreting a
“PASS.”

## 5. Understand the hardware work

The VHDL folder currently explores integer arithmetic and convolution. It is
separate from the trained float32 deployment.

### Integer arithmetic needs a contract

Quantization approximates real values using integers plus scale information.
For example, signed int8 values range from -128 to 127, but multiplying and
summing them requires more bits:

```text
127 × 127 = 16,129
9 × 16,129 = 145,161
```

An int8 accumulator clearly cannot hold this result. Even this nine-product
example does not bound a larger convolution with multiple input channels or
bias. Width must be chosen from a range analysis, not just a few observed images.

The small fixture's observed maximum absolute accumulator, 48,387, fits in a
**17-bit signed value in total**: the range is -65,536 through 65,535.
That observation is not a worst-case proof.

Requantization reduces a wide accumulator back to a narrower representation.
Software and hardware must agree on scale, rounding, shifting, and saturation;
“both use int8” is not enough.

### What exists, and what does not

The [hardware guide](../hardware/vhdl/README.md) explains the MAC, ReLU,
requantizer, convolution, and line buffer. Testbenches compare integer
convolution outputs with Python-generated vectors.

GHDL simulation checks the tested behavior. GHDL synthesis checks whether the
blocks can be translated into hardware. Neither establishes FPGA clock speed,
resource use, power, or board throughput.

The trained CNN is not yet integrated end to end in RTL. That requires a trained
integer reference, additional datapath/control integration, and implementation
and measurement on a chosen device.

## 6. Check your understanding

Try answering before reading the explanations.

1. **Why does convolution output four channels?**
   The default model has four filters. Each filter produces one feature map.
2. **Why is the flattened vector length 784?**
   Pooling leaves four 14×14 maps: `4 × 14 × 14 = 784`.
3. **Does flattening learn weights?**
   No. It only changes how the same values are arranged/viewed.
4. **Can C and NumPy agree while accuracy is poor?**
   Yes. Agreement checks implementation consistency, not the quality of learning.
5. **Why not select the best checkpoint using test accuracy?**
   The test set should assess a choice made without using those test results.

Next: [run the project](setup.md), or follow the smaller
[lessons in order](../lessons/README.md).
