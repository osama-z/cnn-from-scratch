# Train and deploy the CNN

[Project home](../README.md) · [Setup](../docs/setup.md) ·
[How the model works](../docs/guide.md) · [Verification](../docs/verification.md)

This is the current training and deployment path. Both convolution and Dense
weights learn from MNIST; C and C++ run the exported weights.

```text
1×28×28 → Conv(4 filters, 3×3, same) → ReLU → Pool(2×2)
        → Flatten(784) → Dense(10) → Softmax
```

## Run from the repository root

After following [setup](../docs/setup.md):

```bash
make test-software
make download
make train
make verify-trained
make benchmark
make demo
```

The default experiment trains for five epochs on 10,000 images, selects the best
checkpoint on 2,000 disjoint validation images, and evaluates on 10,000 test
images. It uses batch size 64, Adam at 0.001, and seed 42.

Outputs live in `build/trained/`. Use `--out build/another-experiment` with the
Python commands to preserve multiple runs. See the
[artifact table](../docs/setup.md#generated-files) for each file's meaning.

## Find the implementation

| File | Read it to understand |
|---|---|
| [model.py](model.py) | Forward/backward operations, gradients, Adam, checkpoints |
| [train.py](train.py) | Data splits, training loop, checkpoint selection, metrics |
| [download.py](download.py) | Explicit MNIST downloads and archive hash verification |
| [export.py](export.py) | Export weights, test samples, and NumPy layer references |
| [run_common.h](run_common.h) | Native input/model validation and runner utilities |
| [infer_c.c](infer_c.c), [infer_cpp.cpp](infer_cpp.cpp) | C and C++ inference programs |
| [verify.py](verify.py) | Layer-by-layer agreement and prediction checks |
| [benchmark.py](benchmark.py) | Rebuild, verify, warm up, and time native inference |
| [demo.py](demo.py) | Generate an offline gallery of exported predictions |
| [evaluate.py](evaluate.py) | Re-evaluate the saved checkpoint and generate figures |
| [release.py](release.py) | Validate and package a checksummed model/demo bundle |

The reusable kernels and file format code live in [engine/](../engine/README.md).

## Inspect one native run

```bash
make -C trained
trained/infer_c build/trained/weights.bin build/trained/samples.bin
trained/infer_cpp build/trained/weights.bin build/trained/samples.bin --trace
python -m trained.verify --out build/trained
```

The verifier checks input, convolution, ReLU, pooling, flatten, Dense, and softmax
tensors. It requires matching predictions and finite outputs, with floating-point
tolerances of `atol=5e-5` and `rtol=2e-4`.

## Interpret the outputs

The [recorded baseline](../reports/trained-baseline.md) reached 91.87% test
accuracy. Its 32 exported examples agree across NumPy/C/C++ at all checked layers.

The gallery contains precomputed native predictions on held-out examples.
Benchmarks measure host CPU inference. The current VHDL tests use a different,
small integer fixture; deploying this trained network in integer RTL remains
future work.
