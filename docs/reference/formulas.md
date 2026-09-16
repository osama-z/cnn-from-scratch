# CNN formula reference

[Main guide](../guide.md) · [Documentation index](../README.md)

These equations describe the conventions used by the trained float model.
Read the guide first if the symbols are unfamiliar.

## Shapes and indexing

| Symbol | Meaning |
|---|---|
| N | Batch size |
| C | Number of channels |
| H, W | Image height and width |
| K | Kernel size |
| P | Zero padding on each side |
| S | Stride |

Inputs use `(N, C, H, W)`. A single image's flat offset is
`c×H×W + y×W + x`.

For dilation 1:

```text
output_height = floor((H + 2P - K) / S) + 1
output_width  = floor((W + 2P - K) / S) + 1
```

For K=3, P=1, S=1, the spatial size is unchanged.

## Forward pass

The operation convention is cross-correlation, with no kernel reversal:

```text
conv[n,o,y,x] = bias[o]
             + sum over c,ky,kx:
               padded_input[n,c,y+ky,x+kx] × weight[o,c,ky,kx]

relu(x) = max(0, x)
pool[n,c,y,x] = max of input[n,c,2y:2y+2,2x:2x+2]

logits = flattened_input @ dense_weight + dense_bias
shifted = logits - max(logits)
probability[j] = exp(shifted[j]) / sum(exp(shifted))
```

Pooling uses stride 2; an unmatched last row or column is discarded for odd sizes.

## Loss and backward pass

For integer labels y and averaged cross-entropy:

```text
loss = -(1/N) × sum_n log(probability[n, y[n]])
d_logits = (probability - one_hot(y)) / N

d_dense_weight = flattened_input.T @ d_logits
d_dense_bias = sum over batch of d_logits
d_flattened_input = d_logits @ dense_weight.T
```

Pool backward sends each gradient to the first maximum in its forward window.
ReLU backward multiplies by `(input > 0)`, choosing zero derivative at zero.
Convolution backward sums the contributions from every window sharing a weight
or input value. See [model.py](../../trained/model.py) for explicit loops/einsums.

Numerical derivative, for a small perturbation epsilon:

```text
d_loss/d_weight ≈ (loss(weight + epsilon) - loss(weight - epsilon)) / (2×epsilon)
```

Finite-difference checks need care near ReLU zeros and max-pool ties because
these operations are not differentiable at those boundaries.

## Parameter count

```text
convolution = output_channels × input_channels × K × K + output_channels
Dense       = input_features × output_features + output_features
```

The default CNN has `4×1×3×3 + 4 + 784×10 + 10 = 7,890` parameters.

## Integer reference

```text
q = clamp(round(real/scale) + zero_point, -128, 127)
real_approx = scale × (q - zero_point)
quantized_relu(q) = max(q, zero_point)
```

For a sum of quantized products, the accumulator scale is the input scale times
the weight scale. Rescaling must account for the intended output scale and zero
point. Choose accumulator width from the sum's range, including biases.

Rounding rules must be specified explicitly: different libraries can break ties
differently. See [VHDL requantization](../../hardware/vhdl/README.md#understand-the-arithmetic).
