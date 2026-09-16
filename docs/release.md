# Run the v0.1.0 model release

[Project home](../README.md) · [Setup](setup.md) · [Verification](verification.md)

The first release covers NumPy training, C/C++ float inference, reproducible
evaluation, and the existing integer VHDL building blocks. Trained-network int8
deployment and FPGA board measurements remain separate future milestones.

## Try the pre-trained model

Download `cnn-from-scratch-v0.1.0-model.zip` and its `.sha256` file from the
[GitHub releases page](https://github.com/osama-z/cnn-from-scratch/releases).
A draft release is visible only to repository collaborators until published.

Verify the download from the directory containing both files:

```bash
sha256sum --check cnn-from-scratch-v0.1.0-model.zip.sha256
```

Extract the ZIP into any empty folder and open `demo.html` for an offline
prediction gallery. The HTML contains its examples and precomputed outputs;
no Python environment or internet connection is needed to view it.

To run native inference, download the matching source revision, follow the
[software setup](setup.md#1-install-the-tools), and extract the model ZIP into
the source repository's `build/trained/` folder:

```bash
mkdir -p build/trained
unzip /path/to/cnn-from-scratch-v0.1.0-model.zip -d build/trained
make all
make verify-trained
make demo
```

These commands use the bundled inputs and reference layers. They require no
MNIST download and do not retrain the model. Use a fresh destination rather
than extracting over another experiment's files.

## Contents and provenance

The ZIP includes the checkpoint, native weights, 32 input examples and labels,
NumPy layer references, training metrics, offline demo, and available evaluation
figures. `MANIFEST.json` records each file's size and SHA-256 hash, the source
commit, and native verification results. The `.sha256` sidecar covers the entire
ZIP; the manifest itself is covered by that archive hash.

The bundle uses an explicit allowlist. Full dataset archives, local notes,
credentials, source history, and compiled executables are not copied into it.
See [data attribution](../DATA_SOURCES.md) for the source of the digit examples.

## Rebuild the figures and release

After downloading MNIST and training, or using a saved matching checkpoint:

For the recorded environment, use Python 3.12 and install
`requirements-reproduce.txt`. The broader `requirements-release.txt` supports
normal use without pinning to that experiment's versions.

```bash
python -m pip install -r requirements-release.txt
make evaluate
make demo
# Commit source changes before packaging so provenance identifies exact code.
make release
```

Evaluation rechecks the saved model against its recorded metrics and test archive
hashes. It creates a confusion matrix, learning curves, and a deterministic
selection of correct/incorrect predictions under `build/trained/evaluation/`.

Packaging verifies the checkpoint, binary weights, sample files, NumPy references,
and both native engines before generating the demo and archive. It refuses a
dirty source tree or an existing archive destination.

For another experiment/version or a repeated build:

```bash
python -m trained.evaluate --out build/experiment-01
python -m trained.release --out build/experiment-01 --dist build/release-02 --version v0.1.1
```

The manual **Build model release bundle** Actions workflow builds downloadable
artifacts from a clean runner. It does not change repository visibility or
publish a GitHub release automatically.
