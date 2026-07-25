# 📓 Daily Learning Log

> Write 3 sentences every day about what you learned.
> This builds technical writing skill for your research paper.

---

## Day 1 — May 28, 2026
**Topic:** Convolution & CNN Forward Pass (Python)

I built a convolutional neural network from absolute scratch using only NumPy. I learned that an image is just a matrix of numbers, and a convolution is simply sliding a small kernel over it — multiplying and summing at each position. The full pipeline Conv→ReLU→Pool→Dense→Softmax transforms a raw image into class probabilities, and I can now trace every single number through the process.

---

## Day 2 — May 28, 2026
**Topic:** Loss Functions, Backpropagation & Training Loop (Python)

I learned that Cross-Entropy loss is better than MSE for classification because its gradient is huge when the network is confidently wrong, forcing faster correction. Backpropagation uses the Chain Rule to assign "blame" to each weight by multiplying gradients backward through the network. I implemented three optimizers (SGD, Momentum, Adam) and watched my network go from 97% to 100% accuracy on toy data — the moment when random weights became "understanding."

---

## Day 3 — May 28, 2026
**Topic:** MNIST Handwritten Digit Classification (Python)

I scaled from 3 toy images to 60,000 real handwritten digits — the jump from laboratory to reality. Data normalization (dividing by 255) was critical; without it the gradients explode. My 3-layer dense network achieved **95% test accuracy** on 10,000 unseen images, proving the network truly learned to read numbers rather than memorizing the training set.

---

## Day 4 — May 30, 2026
**Topic:** CNN Forward Pass in C (Pointers & Memory)

I rewrote the entire Day 1 Python CNN in pure C using 1D pointers. The key insight: RAM is a single line of bytes, so a 3D tensor `image[c][y][x]` becomes `ptr[c*H*W + y*W + x]`. Manual memory management with `malloc()`/`free()` is mandatory in C — forget to free and your drone runs out of RAM. The C version processes images at **16,000+ FPS** using only **1.4KB** of memory.

---

## Day 5 — June 10, 2026
**Topic:** Fixed-Point Arithmetic (int8 Quantization in C)

I learned that `float` multiplication costs ~20 clock cycles on ARM Cortex-M4 while `int8` costs just 1 cycle — a 20x speedup. The quantization formula `int8 = round(float / scale) + zero_point` allows any float range to be mapped into [-128, 127] with minimal error. The critical trap: ReLU in int8 must use `max(zero_point, x)` not `max(0, x)`, because the integer 0 does NOT represent the float value 0.0.

---

## Day 6 — July 23, 2026
**Topic:** C++ for AI Deployment (RAII, Polymorphism, Templates)

I rewrote the C CNN as C++ classes and learned that RAII is not a convenience but a
guarantee — the destructor runs on *every* exit path, including early returns and thrown
exceptions, which is precisely where C's `free()` gets skipped and a drone slowly leaks its
RAM away. Templates then let me write each layer once and instantiate it twice, as
`CNN<float>` and `CNN<int8_t>`, but the lesson that surprised me is that templating the
storage type alone would have *recreated* Day 5's overflow bug: the accumulator has to be a
separate type, chosen by an `AccumTraits` specialization, so `int8_t` accumulates in
`int32_t`. Day 5 taught that rule as a comment I had to remember, and Day 6 made the
compiler enforce it — and disassembling the two instantiations proved `if constexpr` costs
nothing at runtime, since the float build emits `mulss` while the int8 build emits `movsbl`
and `imull`, sharing not one arithmetic instruction.

---

## Day 7 — July 23, 2026
**Topic:** The Golden Model (Cross-Implementation Verification)

I discovered that my C and C++ CNNs never actually computed the same thing: both seeded
`srand(42)`, but they fed the seed into different weight formulas, so "same seed" produced
two different networks that merely shared a shape — a silent, no-crash bug of the worst
kind. The fix was to stop generating weights and start loading them from a single
`weights.bin`, which forced me to design a real binary model format with a magic number, a
version, a checksum, and 4-byte alignment for zero-copy loading — which is exactly what
ONNX and TFLite are underneath. The verifier compares every intermediate layer, not just
the final prediction, so the first layer that diverges pinpoints the bug; NumPy, C, and C++
now agree to within 1e-10, and the same script emits int8 reference vectors (peak int32
accumulator = 24,384, Day 5's overflow rule measured on real data) for the VHDL testbench I
build in Phase 1.

---
