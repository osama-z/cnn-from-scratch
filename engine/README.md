# Shared inference engine

[Project home](../README.md) · [Trained pipeline](../trained/README.md) ·
[Model format](../docs/reference/model-format.md)

This folder contains code shared by the small Lesson 7 fixture and the trained
CNN. It does not depend on a lesson.

| File | Responsibility |
|---|---|
| [cnn_c.h](cnn_c.h) | Scalar float convolution, ReLU, pooling, Dense, and softmax |
| [cnn.hpp](cnn.hpp) | C++ Tensor and float layer implementations |
| [model_io.h](model_io.h) | Header-only binary reader usable from C and C++ |
| [model_format.py](model_format.py) | Python writer for the same binary format |

## Read the call path

For a complete usage example, start with
[infer_c.c](../trained/infer_c.c) or [infer_cpp.cpp](../trained/infer_cpp.cpp).
[run_common.h](../trained/run_common.h) validates model/input compatibility and
handles the runner interface.

```text
Python parameters → model_format.py → weights.bin
                                          ↓
                                     model_io.h
                                      ↙     ↘
                                cnn_c.h     cnn.hpp
```

C and C++ share the parser, but their layer arithmetic is implemented separately.

## Layout and ownership

Inputs use channel-first row-major order; convolution weights use
`(out_channel, in_channel, kernel_y, kernel_x)`; Dense weights use
`(in_features, out_features)`.

The native reader owns one loaded file buffer. Tensor weight pointers borrow
from that buffer: keep the Model alive until every consumer has finished.
The C++ runner wraps the Model in an owning RAII object.

C kernels operate on caller-provided buffers; callers must supply valid shapes
and sufficient storage. The trained runner checks those contracts before calling
the kernels. C++ Tensor/layer interfaces also validate supported shapes.

The parser validates structure and bounds. The trained runners additionally
reject incompatible model shapes and non-finite values. The format's byte-sum
checksum is an accidental-corruption check, not authentication.

## Validate changes

From the repository root:

```bash
make test-software
make asan
```

If trained artifacts are available, also run `make verify-trained`.
See the [verification guide](../docs/verification.md) for what each check covers.
