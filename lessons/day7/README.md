# Lesson 7: share weights and verify every layer

[Previous: C++](../day6/README.md) · [All lessons](../README.md) ·
[Next: train the CNN](../../trained/README.md)

Goal: export one model and prove that independent implementations compute
matching intermediate results. You need NumPy plus the C/C++ concepts from
Lessons 4 and 6.

## The frozen verification model

```text
1×8×8 image → Conv 2×8×8 → ReLU → Pool 2×4×4 → Flatten 32 → Dense 3 → Softmax
```

The model has 119 float32 parameters: 18 convolution weights, 2 convolution
biases, 96 Dense weights, and 3 Dense biases. Its file is 636 bytes including
metadata. The filters and untrained classifier make a small reproducible
fixture; the probabilities are not a measure of learned classification quality.

## Files to read

| File | Role |
|---|---|
| [export_weights.py](export_weights.py) | Define the fixture and export float references and integer convolution vectors |
| [weights.bin](weights.bin) | Frozen binary parameters |
| [reference_io.txt](reference_io.txt) | NumPy reference inputs and layer outputs |
| [golden_c.c](golden_c.c), [golden_cpp.cpp](golden_cpp.cpp) | Load the same file and run separate arithmetic implementations |
| [verify.py](verify.py) | Compare the layer traces |
| [test_robust.py](test_robust.py) | Reject malformed parameter files |
| [test_engine.cpp](test_engine.cpp) | Validate the reusable C++ tensor/layer interfaces |
| [vhdl_vectors/](vhdl_vectors/) | Integer inputs, weights, and accumulator references for hardware |

Shared layers and the binary reader/writer are in [engine/](../../engine/README.md).
The file contract is described in the [format specification](../../docs/reference/model-format.md).

## Run it

From the repository root:

```bash
make verify-golden
make -C lessons/day7 test-robust test-engine
# Reproduce the tracked fixture from its source:
make -C lessons/day7 model
# Check native memory behavior:
make -C lessons/day7 asan
```

To choose a Python environment, append `PYTHON=/absolute/path/to/python`.
Run the exporter as a module from the root: `python -m lessons.day7.export_weights`.

The verifier checks four images. The recorded maximum absolute float difference
is about `1.2e-10`; the acceptance tolerances are `atol=1e-4, rtol=1e-4`.

## Diagnose the first difference

If the convolution differs, check layout, padding, filter order, and bias first.
If convolution agrees but Dense differs, check flattening order and Dense weight
layout. Intermediate outputs make this much easier than comparing only the final
class.

Checkpoint: why do matching final predictions not prove matching implementations?
Answer: different logits or intermediate tensors can still have the same largest
output.

The [trained pipeline](../../trained/README.md) applies this method to learned
weights. The [hardware tests](../../hardware/vhdl/README.md) use the separate
integer convolution vectors from this fixture.
