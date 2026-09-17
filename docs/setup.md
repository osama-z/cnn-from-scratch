# Setup and commands

[Documentation index](README.md) · [How the CNN works](guide.md)

All commands below run from the **repository root**, unless a subshell explicitly
changes directory. Activate your Python environment in each new terminal.

## 1. Install the tools

For the software path:

- Python 3.10 or newer, with NumPy.
- GCC, G++, and Make.

For the hardware path, also install GHDL. GTKWave is optional for viewing
waveforms. The plotting scripts in Lessons 1–3 additionally need Matplotlib.

On Ubuntu or Debian:

```bash
sudo apt-get update
sudo apt-get install build-essential python3-venv
# Optional: needed for the VHDL tests.
sudo apt-get install ghdl gtkwave
```

Create a Python environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
# Optional: plots in the original lessons.
python -m pip install -r lessons/requirements.txt
```

If you already have an environment, use it instead. Make accepts an explicit
interpreter, for example:

```bash
make test-software PYTHON=/absolute/path/to/venv/bin/python
```

## 2. Check the installation

```bash
make test-software  # Golden model, malformed files, gradients, training, deployment
make docs-check     # Local Markdown links
make test-hardware  # VHDL simulations and synthesis checks; requires GHDL
make test           # All three groups
```

Tests use fixed fixtures and small generated images. No MNIST download is needed.
A passing run is explained in the [verification guide](verification.md).

## 3. Download MNIST and train

```bash
make download
make train
```

Downloading is explicit and verifies archive SHA-256 hashes. Valid existing
archives are reused from `lessons/day3/mnist_data/`.

Default training uses four convolution filters, five epochs, batches of 64,
Adam with learning rate 0.001, and seed 42. It uses 10,000 training images and
2,000 disjoint validation images from the training archive. Validation selects
the checkpoint; the selected checkpoint is evaluated on 10,000 test images.

The [recorded run](../reports/trained-baseline.md) reached 91.87% test accuracy.
That is a baseline, not a promised result for every configuration.

To preserve another experiment, choose a different output directory:

```bash
python -m trained.train --epochs 5 --train-size 10000 --validation-size 2000 \
  --test-size 10000 --seed 42 --out build/experiment-01
```

Reusing an output directory overwrites the artifacts produced by that command.
The examples below use the default `build/trained/` directory.

## 4. Verify, benchmark, and view predictions

```bash
make verify-trained
make benchmark
make demo
```

To create test-set figures and a distributable model bundle, follow the
[release guide](release.md). The pre-trained bundle can also be used to verify
native inference without downloading the dataset or training again.

Open `build/trained/demo.html` in a browser. The page displays precomputed
predictions from actual NumPy, C, and C++ runs on exported test images. It needs
no external web assets. It does not accept hand-drawn input.

For a non-default experiment:

```bash
python -m trained.verify --out build/experiment-01
python -m trained.benchmark --out build/experiment-01
python -m trained.demo --out build/experiment-01
```

The benchmark verifies the implementations before timing them. It measures CPU
inference, not FPGA performance. See [measurement boundaries](verification.md#benchmarks).

## Generated files

These artifacts live under `build/trained/`, which Git ignores.

| File | Meaning | Produced by |
|---|---|---|
| `checkpoint.npz` | Learned weights and model dimensions | Training |
| `weights.bin` | The weights in the native binary format | Training/export |
| `samples.bin` | Exported float32 test inputs | Training/export |
| `samples.npz` | Test images and labels | Training/export |
| `reference.npz` | NumPy outputs at every checked layer | Training/export |
| `metrics.json` | Accuracy, split sizes, history, and hashes | Training |
| `benchmark.json` | Timings and machine/compiler details | Benchmark |
| `demo.html` | Offline prediction gallery | Demo |

Tracked files in [reports/](../reports/trained-baseline.md) describe an earlier
recorded run. They do not update automatically when you run a new experiment.

## Useful commands

| Task | Command from the repository root |
|---|---|
| List common commands | `make help` |
| Build native inference programs | `make all` |
| Check the small frozen model | `make verify-golden` |
| Regenerate that model and its vectors | `make -C lessons/day7 model` |
| Exercise malformed model files | `make -C lessons/day7 test-robust` |
| Run native memory/undefined-behavior checks | `make asan` |
| Generate a MAC waveform | `make -C hardware/vhdl wave` |
| See training options | `python -m trained.train --help` |

Run plotting lessons inside their own folder so generated images stay there:

```bash
(cd lessons/day1 && python part1_convolution.py)
```

## Troubleshooting

- **No NumPy or Matplotlib:** activate the intended environment, then install
  the matching requirements file with that environment's `python -m pip`.
- **MNIST files missing:** run `make download` before training. Downloading
  requires network access; tests do not.
- **GHDL missing:** use `make test-software` until GHDL is installed.
- **No trained weights:** run `make train` before verification, benchmarking,
  or the demo, and check that all commands use the same output directory.
- **LeakSanitizer reports a ptrace limitation:** run `make asan` in an ordinary
  terminal outside debugger/tracing restrictions. This environment error is not
  evidence of a model memory leak.
- **An old path no longer exists:** use `lessons/dayN/`, `hardware/vhdl/`,
  or the [documentation index](README.md). Run Python modules from the root;
  for example, `python -m lessons.day7.export_weights`.
