# Trained CNN baseline — September 15, 2026

This run trains an actual convolutional digit classifier. It is separate from
the Day 3 dense-only classifier and Day 7's untrained 8×8 teaching model.

| Setting | Value |
|---|---|
| Architecture | Conv(1→4, 3×3 same) → ReLU → MaxPool(2) → Dense(784→10) → Softmax |
| Learned parameters | 7,890 float32 values |
| Training | 10,000 shuffled MNIST training examples |
| Validation | 2,000 disjoint examples from the training archive |
| Test | All 10,000 official test examples |
| Optimizer | Mini-batch Adam, learning rate 0.001, batch 64 |
| Seed / epochs | 42 / 5 |
| Selected checkpoint | Epoch 5, selected by validation accuracy |
| Validation accuracy | 92.30% |
| Test accuracy | **91.87%** |
| Python / NumPy | 3.12.3 / 2.4.2 |

The exported `weights.bin` is 31,720 bytes including tensor metadata. Its SHA-256:

```text
0faae314b694a47245e5126c0db8a5c7de779dca66a4f03dbf5e89df55a5a4b7
```

## Deployment verification

The 32 exported test images produce 224 layer tensors per engine. C and C++ both
match the NumPy reference at every layer, with maximum absolute error
`9.57e-6`, and agree on all 32 predictions. Agreement does not imply every
prediction is correct; classification accuracy is measured separately above.

The original Day 7 fixture still matches to `1.20e-10`. Both existing VHDL
convolution testbenches still pass 384 comparisons each. Those hardware checks
use the original integer convolution vectors, not the newly trained classifier.

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m trained.download
OPENBLAS_NUM_THREADS=1 python -m trained.train --epochs 5 --seed 42
make verify-trained
make benchmark
python -m trained.demo
make test
make asan
```

Exact floating-point results can vary with NumPy/BLAS versions and compiler
choices. Split sizes, hashes, validation history, and final accuracy are recorded
in `build/trained/metrics.json`. Reproducing an accuracy number and verifying
cross-language agreement are separate checks.

## Host performance

The benchmark records five trials of 2,000 inferences after ten warm-ups per
trial. It times inference with `CLOCK_MONOTONIC` and excludes file loading and
printing. C uses preallocated buffers; C++ includes Tensor allocations and input
copies. See the generated `build/trained/benchmark.json` for machine/compiler
details, trial timings, model and binary hashes, and build commands.

These results are measured in WSL2 on the host CPU. They do not describe an ARM
target or an FPGA. No vendor timing/resource report or physical board
measurement was produced in this release.

Measured on an Intel Core i7-10750H with GCC/G++ 13.3.0 and `-O2`:

| Engine | Median latency per image | Throughput |
|---|---:|---:|
| C, preallocated buffers | 63.81 µs | 15,671 images/s |
| C++, Tensor allocations included | 90.10 µs | 11,099 images/s |

The timing scopes differ as described above; this is not an isolated comparison
of the C and C++ languages. Raw records are saved in
[`host-benchmark.json`](host-benchmark.json) and
[`trained-metrics.json`](trained-metrics.json).
