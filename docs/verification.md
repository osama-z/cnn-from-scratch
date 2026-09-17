# Understand the verification results

[Documentation index](README.md) · [Setup](setup.md) ·
[Recorded baseline](../reports/trained-baseline.md)

A classifier needs several different checks. Good test accuracy does not prove
correct gradients, safe file parsing, or equivalent native implementations.

## Run the checks

From the repository root with the dependencies installed:

```bash
make test-software
make test-hardware
make docs-check
# All of the above:
make test
# Additional native memory checks:
make asan
```

None of these requires an MNIST download. To check a trained bundle, first
generate it using [setup](setup.md), then run `make verify-trained`.

## Software

| Check | Where it lives | What passing establishes |
|---|---|---|
| Frozen model agreement | [Lesson 7 verifier](../lessons/day7/verify.py) | C/C++ match the small NumPy fixture within tolerance |
| Invalid model handling | [Parser checks](../lessons/day7/test_robust.py) | Both programs reject 17 malformed-file cases with the expected exit status |
| Tensor/layer validation | [C++ engine checks](../lessons/day7/test_engine.cpp) | Tested invalid shapes are rejected; tested operations behave as expected |
| Numerical gradients | [Training tests](../tests/test_trained.py) | Convolution and network derivatives match finite differences |
| Synthetic learning | [Training tests](../tests/test_trained.py) | Training learns a small known pattern dataset |
| Export and deployment | [Trained verifier](../trained/verify.py) | Every expected layer is present, finite, and close; predictions agree |
| Failure propagation | [Regression tests](../tests/test_trained.py) | A failed synthesis tool or broken corruption checker cannot silently pass |
| Documentation links | [Link checker](../scripts/check_docs.py) | Local inline Markdown links and supported heading anchors resolve |
| Evaluation and packaging | [Release tests](../tests/test_release.py) | Confusion-matrix orientation, model/sample consistency, archive checksums, and file allowlist |

The trained verifier checks `input`, `conv`, `relu`, `pool`, `flatten`,
`dense`, and `softmax`. It rejects missing or duplicate trace entries and
non-finite values.

Its float comparison uses:

```text
abs(actual - reference) <= 5e-5 + 2e-4 × abs(reference)
```

The older golden-model verifier uses `atol=1e-4, rtol=1e-4`.
The recorded maximum errors are smaller than these acceptance thresholds.
Floating-point summation order can differ between implementations.

The recorded trained bundle contains 32 images, giving 224 checked layer tensors
per engine. Agreement is specific to that bundle, while accuracy is evaluated
separately against all 10,000 test labels.

## Memory checks

`make asan` builds with AddressSanitizer and UndefinedBehaviorSanitizer and
runs native paths, including malformed inputs. A pass means those executed
paths produced no detected violations; it is not proof about every possible input.

LeakSanitizer can fail under ptrace/debugger restrictions. Run in an ordinary
terminal when the error explicitly reports that environment limitation.

## Hardware

The four VHDL testbenches cover the MAC, ReLU/requantization, direct-window
convolution, and streamed-window convolution. Each convolution bench checks
384 outputs against integer reference vectors.

The synthesis gate checks five blocks with GHDL. These are tests of the small
integer convolution fixture, separate from the trained float network.
See the [hardware guide](../hardware/vhdl/README.md).

A passing simulation is not an FPGA timing/resource report.

## Benchmarks

`make benchmark` rebuilds and verifies the runners, then measures five trials.
Each trial warms up for ten inferences before timing 2,000 inferences using a
monotonic clock. File loading and output printing are outside the timer.

C uses preallocated activation buffers. C++ timing includes input copies and
Tensor allocations. This difference is part of the recorded measurement, so the
results are not an isolated comparison of language performance.

The generated JSON records machine, compiler, build commands, model/binary
hashes, and timings. Tracked reports preserve a specific earlier run; historical
compiler paths may reflect the folder layout at that time.

## Investigate a failure

1. Confirm the commands use the intended Python environment and artifact folder.
2. Find the first layer that differs, rather than only the final prediction.
3. Check tensor shape and memory order, padding, and bias application.
4. For integer results, check zero points, scales, rounding, and saturation.
5. Re-run the relevant check after the smallest necessary change.

The [CI workflow](../.github/workflows/verify.yml) runs software/hardware checks,
sanitizers, and regeneration of the frozen model on pushes and pull requests.
