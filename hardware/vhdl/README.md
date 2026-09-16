# Integer CNN building blocks in VHDL

[Project home](../../README.md) · [CNN guide](../../docs/guide.md) ·
[Integer arithmetic lesson](../../lessons/day5/README.md)

Goal: implement and verify the arithmetic and window formation needed for integer
convolution. These blocks are not yet a complete implementation of the trained
MNIST classifier.

## Find each block

| Source | Role |
|---|---|
| [cnn_types.vhd](cnn_types.vhd) | Shared signed int8 array types |
| [mac_unit.vhd](mac_unit.vhd) | Clocked multiply-accumulate |
| [relu_int8.vhd](relu_int8.vhd) | ReLU using the quantized zero point |
| [requantize.vhd](requantize.vhd) | Wide integer multiplication, shift, rounding, and saturation |
| [conv3x3.vhd](conv3x3.vhd) | Combinational sum of nine int8 products |
| [line_buffer.vhd](line_buffer.vhd) | Form padded 3×3 windows from a raster pixel stream |

The streaming convolution test connects the final two blocks:

```text
pixel stream → line buffer → nine-pixel window → conv3x3 → wide accumulator
                                               ↑
                                         nine weights
```

The convolution output is an accumulator, before a complete trained network's
bias/requantization/activation/pooling/classification path.

## Run the checks

Install GHDL using [setup](../../docs/setup.md). From the repository root:

```bash
make test-hardware
# Individual tasks:
make -C hardware/vhdl test
make -C hardware/vhdl synth
# Optional waveform:
make -C hardware/vhdl wave
gtkwave hardware/vhdl/build/mac.ghw
```

The Makefile uses VHDL-2008 and the standard `numeric_std` package.
Generated work libraries and waveforms stay in the ignored build directory.

## What each testbench checks

| Testbench | Coverage |
|---|---|
| [tb_mac_unit.vhd](tb_mac_unit.vhd) | Reset, enabled products, and accumulated results |
| [tb_relu_requant.vhd](tb_relu_requant.vhd) | ReLU and requantization cases |
| [tb_conv3x3.vhd](tb_conv3x3.vhd) | Convolution with windows supplied by the bench |
| [tb_stream_conv.vhd](tb_stream_conv.vhd) | Window generation plus convolution from a pixel stream |

Each convolution bench compares 384 accumulator values: three 8×8 images times
two filters. Its reference files come from
[the frozen Lesson 7 fixture](../../lessons/day7/vhdl_vectors/).
They can be regenerated with `make -C lessons/day7 model`.

## Understand the arithmetic

With zero points of zero, convolution adds products of signed input and weight
integers. Nine products of `127×127` already sum to 145,161, so an int8
accumulator would overflow. The blocks use a wider accumulator.

To convert the accumulator back to int8, approximate the real scale ratio with
an integer multiplier and a power-of-two divisor:

```text
rescaled = round(accumulator × multiplier / 2^shift)
output   = clamp(rescaled + output_zero_point, -128, 127)
```

The requantizer handles the sign explicitly for round-half-away-from-zero, then
saturates. Scale approximation can still change results relative to a float
reference, so the integer contract must be tested explicitly.

ReLU and padding must use the integer representation of real zero. For these
convolution vectors, the zero point is 0.

## Understand the stream

The line buffer retains `2×image_width + 3` pixel positions. It tracks the
window center and substitutes zero for taps outside the frame.

The window center follows the incoming pixel by `image_width + 1` positions.
The caller must supply the documented flush cycles after the last pixel to
emit the remaining windows. See [line_buffer.vhd](line_buffer.vhd) and the
[stream testbench](tb_stream_conv.vhd) for the protocol.

A combinational convolution has propagation delay, not an independently measured
clock rate. Its achievable timing depends on synthesis, placement, routing, and
any added pipeline registers.

## What passing proves

Simulation proves agreement for the tested cases. The synthesis target checks
that GHDL can synthesize five blocks and fails if any block fails synthesis.
Neither test establishes FPGA resource usage, clock frequency, power, or board
throughput.

Next steps are a calibrated trained integer reference, integration of the
remaining CNN operations, stronger stream/stall/frame tests, and measurement on
a chosen FPGA. See [verification](../../docs/verification.md#hardware).
