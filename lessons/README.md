# Learning path

[Project home](../README.md) · [Setup](../docs/setup.md) · [Main CNN guide](../docs/guide.md)

The day numbers describe the order of the original lessons. Work at your own
pace; each folder opens with an explanation and commands you can copy.

| Lesson | Learn | Main language | What to know first |
|---|---|---|---|
| [1 — Forward pass](day1/README.md) | Convolution, activation, pooling, Dense, softmax | Python arrays |
| [2 — Learning](day2/README.md) | Loss, derivatives, backpropagation, optimizers | Lesson 1; basic derivatives |
| [3 — MNIST](day3/README.md) | Train a dense digit classifier | Lesson 2 |
| [4 — C implementation](day4/README.md) | Flat tensor indexing and explicit buffers | C pointers and loops |
| [5 — Integer arithmetic](day5/README.md) | Scales, zero points, accumulation, saturation | Lesson 4 |
| [6 — C++ design](day6/README.md) | Tensor ownership, layers, and templates | Basic C++ |
| [7 — Shared model](day7/README.md) | Export once and compare intermediate results | Lessons 4 and 6 |

Lessons 1–3 use NumPy and Matplotlib. Lessons 4–7 use Make and GCC/G++;
Lesson 7 also needs NumPy. Install dependencies using [setup](../docs/setup.md).

## How to study a lesson

1. Read its goal and worked example.
2. Run one program and inspect the printed values or plot.
3. Find the function that produced a value and calculate a small case yourself.
4. Answer the checkpoint before moving on.

The original scripts contain detailed inline explanations. They are experiments,
so their models and weights differ. Cross-language equivalence is explicitly
checked in Lesson 7; earlier examples are not one shared trained network.

## After the lessons

- [Train the convolutional model](../trained/README.md).
- [Read the reusable engine](../engine/README.md).
- [Explore VHDL convolution](../hardware/vhdl/README.md).
- [Learn what the tests establish](../docs/verification.md).
