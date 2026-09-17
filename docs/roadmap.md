# Technical roadmap

[Project home](../README.md) · [Verification](verification.md) · [Recorded baseline](../reports/trained-baseline.md)

The next goal is to take the verified trained CNN from float32 CPU inference to
integer inference and then to a measured hardware implementation. Each milestone
should produce reproducible evidence before the next one begins.

## Current baseline

- NumPy trains the convolution and Dense layers with explicit derivatives.
- The recorded checkpoint reaches 91.87% accuracy on 10,000 MNIST test images.
- C and C++ load the exported weights and match NumPy at every checked layer
  across 32 exported examples.
- VHDL MAC, activation, requantization, convolution, and line-buffer blocks pass
  simulation and GHDL synthesis checks using a separate integer fixture.
- The model release includes an offline demo, evaluation figures, and a verified
  archive with checksums and source provenance.

The existing VHDL tests do not constitute deployment of the complete trained CNN.

## 1. Quantize the trained CNN

Build an integer reference for the saved checkpoint, including weight and
activation scales, zero points, int32 accumulation, rounding, and saturation.
Use training data for calibration and validation data to select the settings;
reserve the test set for the final comparison.

Deliverables:

- A documented integer model format and reproducible calibration command.
- NumPy and native integer implementations checked at every layer.
- Tests for accumulator bounds, saturation, rounding, padding, and pooling.
- A report of accuracy change, parameter storage, peak working memory, and CPU
  latency relative to the existing float32 model.

## 2. Integrate the complete network in VHDL

Connect the verified integer arithmetic to the streaming convolution blocks,
then implement pooling, the Dense layer, control, and model loading. Define the
input/output protocol and test stalls, reset behavior, and multiple images.

Deliverables:

- Test vectors exported from the trained integer reference.
- Agreement at layer boundaries and final class decisions.
- A reproducible simulation of the complete network.
- A documented buffer layout and cycle counts for the chosen configuration.

## 3. Measure a hardware target

Choose one FPGA board and toolchain, then record the device, clock constraints,
tool versions, and build settings. Report synthesis and implementation estimates
separately from measurements made on the board.

Deliverables:

- LUT, flip-flop, block RAM, and DSP use, plus timing reports.
- Measured latency and throughput with the workload and transfer boundaries stated.
- Power or energy measurements only when suitable measurement equipment is available.
- A comparison of resource use and throughput for a small set of parallelism and
  buffering choices, with numerical agreement checked after each change.

## 4. Extend to a constrained application

After the inference path is validated, select one task and one target, such as
keyword spotting on a microcontroller or visual sensing on an FPGA. Establish a
baseline and explicit accuracy, memory, and response-time requirements before
optimizing. Keep MNIST as the small reproducible integration example.

Each completed milestone should update the verification guide and reports with
commands, measured results, and remaining limitations.
