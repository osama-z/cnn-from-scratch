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

## Day 6 — 
**Topic:** 

(Write your log here after completing Day 6)

---
