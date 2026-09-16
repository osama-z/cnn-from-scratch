# Lesson 5: represent activations with integers

[Previous: C](../day4/README.md) · [All lessons](../README.md) ·
[Next: C++](../day6/README.md)

Goal: understand quantization scales, zero points, wide accumulators, and
saturation. Read Lesson 4 first.

## Run the experiment

[fixed_point.c](fixed_point.c) contains the arithmetic examples and float/integer
comparison. From the repository root:

```bash
make -C lessons/day5
./lessons/day5/fixed_point
```

Read the printed scales, intermediate values, and differences. These are teaching
experiments, separate from the learned CNN's deployment path.

## Work through quantization

An affine representation uses:

```text
q = clamp(round(real / scale) + zero_point, -128, 127)
real_approx = scale × (q - zero_point)
```

For scale 0.1 and zero point 0, a real value of 1.23 becomes integer 12 and
reconstructs as 1.2. The lost 0.03 is quantization error.

If zero point is -5, real zero is stored as -5. Therefore ReLU must clamp at
the zero point, and padding must also use that zero point.

## Why the accumulator is wider

Two int8 operands can produce a product much larger than 127:
`127×127 = 16,129`. Summing nine such products gives 145,161. Accumulation
therefore needs a wider integer type. Bias and extra input channels can enlarge
the required range further.

After accumulation, rescale and saturate before storing int8. Saturation clamps
an out-of-range result; integer wrapping would change its meaning.

Reducing parameter storage does not by itself prove faster inference. Speed
depends on the processor, instructions, memory access, and conversion costs.

Checkpoint: with zero point -5, what is quantized ReLU of -9?
Answer: -5, the integer representation of real zero.

Next, see [C++ templates](../day6/README.md) and the
[VHDL arithmetic guide](../../hardware/vhdl/README.md).
