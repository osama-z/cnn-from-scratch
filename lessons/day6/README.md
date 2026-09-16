# Lesson 6: organize the network in C++

[Previous: integers](../day5/README.md) · [All lessons](../README.md) ·
[Next: a shared model](../day7/README.md)

Goal: express tensors and layers with clear ownership, then explore type-based
float and integer implementations. You need basic C++ classes and templates.

## Files and commands

| File | What it teaches |
|---|---|
| [cnn_forward.cpp](cnn_forward.cpp) | Tensor objects, layer interfaces, and owned layers |
| [templated_cnn.cpp](templated_cnn.cpp) | Float/int8 variants through templates and compile-time branches |
| [Makefile](Makefile) | Normal builds, sanitizer runs, and optional assembly output |

From the repository root:

```bash
make -C lessons/day6 run
make -C lessons/day6 asan
# Optional: inspect generated compiler assembly.
make -C lessons/day6 asm
```

The programs print forward-pass results and comparisons. Sanitizers check the
executed paths for memory errors and undefined behavior.

## Understand ownership

A Tensor owns its values through its container. A network owns its layers through
smart pointers. When an owning object leaves scope, its resources are released.
This is RAII: resource lifetime follows object lifetime.

Shape and arithmetic correctness are separate concerns. Automatic memory
management does not prove that a convolution uses the right index.

## Understand the template boundary

Choosing `float` or `int8_t` changes how values are represented. Integer
convolution still requires a wider accumulator and scale-aware conversion.
Replacing the element type alone does not implement correct quantization.

The original template experiment can use floating-point scale operations.
It is not proof of an entirely integer deployment of the trained model.

Checkpoint: does owning an int8 Tensor imply its convolution sums safely fit
in int8? Answer: no; choose accumulator width from the value range and sum length.

The reusable float C++ layers live in [engine/cnn.hpp](../../engine/cnn.hpp).
Continue with [Lesson 7](../day7/README.md) to compare them against shared weights.
