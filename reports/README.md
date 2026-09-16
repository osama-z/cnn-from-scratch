# Recorded experiment results

[Project home](../README.md) · [Verification guide](../docs/verification.md)

These files preserve the September 15, 2026 baseline:

| File | Contents |
|---|---|
| [trained-baseline.md](trained-baseline.md) | Human-readable configuration, accuracy, agreement, and timing results |
| [trained-metrics.json](trained-metrics.json) | Raw training history, split sizes, hashes, and accuracy |
| [host-benchmark.json](host-benchmark.json) | CPU timing trials and machine/compiler details |

Fresh runs write to `build/trained/` or the directory chosen with `--out`.
They do not automatically replace these reports. Historical build commands in
the JSON may use the previous folder layout; keep them as the record of what
was actually run.

Follow [setup](../docs/setup.md) to reproduce the experiment. Accuracy, numerical
agreement, and CPU timing are separate measurements. These reports contain no
measured FPGA timing, resource use, or board throughput.
