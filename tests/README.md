# Regression tests

[Verification guide](../docs/verification.md) · [Project home](../README.md)

[test_trained.py](test_trained.py) checks numerical gradients, pooling edge
cases, learning on generated patterns, checkpoint/export consistency, native
inference, invalid shapes/values, and failure propagation.

From the repository root:

```bash
make test-software
make asan
```

These commands build the native programs before running the tests. No MNIST
download is required. To run just this module after building the programs:

```bash
make -C trained
python -m unittest discover -s tests -v
```

The original parser and C++ engine checks live with the
[Lesson 7 fixture](../lessons/day7/README.md). Hardware testbenches live beside
the [VHDL blocks](../hardware/vhdl/README.md).
