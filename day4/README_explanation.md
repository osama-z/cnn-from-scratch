# ⚙️ Day 4: The Hardware Language (C Pointers & Memory)

> **Core Concept:** Python runs on servers. C runs on the microcontroller inside a
> drone. Day 4 translates Day 1 into code that can actually be deployed — and forces
> you to learn what a tensor *really is* in memory.
>
> Every number in this file was produced by running the code in this folder.

---

## 0. The files

| File | Lines | What it builds | Run |
|---|---|---|---|
| `conv2d.c` | 149 | Convolution alone — the first step | `make && ./conv2d` |
| `cnn_forward.c` | 537 | The full network: conv → relu → pool → dense → softmax | `make && ./cnn_forward` |

`conv2d.c` is deliberately separate. It's the "part 1" of Day 4, mirroring the
`part1_/part2_/part3_` progression of Days 1–3 — learn convolution in C before
assembling a whole network in C.

---

## 1. RAM is a single line of bytes

This is the idea the entire day exists to teach.

In Python you wrote:

```python
output[c][y][x] = ...          # clean, readable, and a lie
```

There is no 2D array in memory. There is one flat tape:

```text
Address:  [0x000] [0x004] [0x008] [0x00C] [0x010] [0x014] ...
Data:     [px00]  [px01]  [px02]  [px10]  [px11]  [px12]  ...
            ↑ row 0, col 0          ↑ row 1, col 0
```

So you compute the address yourself:

```text
index = row * width + col                    ← 2D image
index = ch * (H * W) + row * W + col         ← 3D tensor (channel, height, width)
```

> **Why this matters far beyond C.** That formula is exactly what you write again in
> Day 6's `operator()`, in Day 7's loader, and in the VHDL address generator in
> Phase 1. Learning it here means the FPGA work later is arithmetic you already know.
>
> NumPy did this all along — `img[c][y][x]` was computing this same index behind
> `strides`. C just stops hiding it.

---

## 2. What each function does

```text
[Input Image]  float*, a 1D array pretending to be a 3D grid
      │
      ▼
conv2d()   ──► slides a 3×3 kernel over every pixel
      │        nested loops: out_ch → y → x → in_ch → ky → kx
      │        that 6-deep loop is the heart of all computer vision
      ▼
relu()     ──► if (x < 0) x = 0;
      │        IN-PLACE — no second buffer. On an MCU that matters.
      ▼
maxpool2d()──► reads 4 values (2×2), writes the largest
      │        output is half the size in both H and W
      ▼
dense()    ──► out[j] = Σ in[i] * w[i * out_size + j] + b[j]
      │        a dot product; every input touches every output
      ▼
softmax()  ──► exp(x - max) / Σ exp(x - max)
               the "- max" prevents inf. expf(1000) on an MCU is a crash.
```

**Note the loop order in `conv2d`:** `out_ch → y → x → in_ch → ky → kx`. The output
pixel is fixed by the outer loops and the accumulator `sum` lives in a register
across the inner ones. Reordering these loops changes cache behaviour dramatically —
that's Day 11's subject (tiling), and it's why the order is worth noticing now.

---

## 3. Memory management — the real difference

Python has garbage collection; you never think about memory. In C **you** are
responsible, and forgetting is silent.

```c
float* buffer = malloc(64 * sizeof(float));   // 256 bytes
/* ... use it ... */
free(buffer);                                  // MANDATORY
```

**Golden rule:** every `malloc()` needs exactly one `free()`. This file has 5
allocations and 5 frees — verified with AddressSanitizer:

```console
$ gcc -fsanitize=address,undefined cnn_forward.c -lm && ./a.out
   (no leak report)
```

> The failure mode isn't a crash — it's a drone that runs fine for 40 minutes and
> then dies. A leak of 256 bytes per frame at 30 FPS is 27 MB per hour.
>
> **Day 6 removes this problem entirely** with RAII: the destructor runs on every
> exit path, including the early `return` where you'd forget `free()`. That contrast
> is the whole reason Day 4 is written in procedural C and Day 6 in C++.

---

## 4. Measured results

```text
  Total parameters:  119
  Memory (float32):  476 bytes
  Memory (int8):     119 bytes          ← Day 5's payoff, previewed

  3 images processed in: 0.074 ms
  Per image:             0.025 ms
  Estimated FPS:         40,541

  Memory usage:
    Input buffer:    256 bytes
    Conv output:     512 bytes
    Pool output:     128 bytes
    Dense output:     12 bytes
    TOTAL buffers:   908 bytes
```

**908 bytes of working memory + 476 bytes of weights ≈ 1.4 KB total.**

Put that against a Cortex-M4 with 256 KB of SRAM: this network uses **0.5%** of it.
That is what "fits on a microcontroller" actually means, in numbers.

> ⚠️ **FPS is machine-dependent.** 40,541 here, and the docs elsewhere quote 16,000+
> — both are real, measured on different hardware and compiler settings. Quote a
> *floor* you can defend, not your best run. What's *not* machine-dependent is the
> memory figure, which is why it's the better number to cite.

---

## 5. Python vs C

| Metric | Python | C |
|---|---|---|
| Array indexing | `img[c][y][x]` | `ptr[c*H*W + y*W + x]` |
| Memory management | automatic (GC) | manual (`malloc`/`free`) |
| Runs on an MCU? | no | **yes** |
| Binary / runtime size | ~50 MB venv | ~25 KB binary |
| Working memory | MBs | **908 bytes** |
| Leak possible? | no | **yes — silently** |

A drone's Flash might be 256 KB total. A 50 MB Python environment is not 200× too
big — it's 200× too big *before you load a model*. **C isn't an optimization here,
it's the only option.**

---

## 6. Why this file stays procedural

`day4/cnn_forward.c` is **not** written in an object-oriented style, deliberately.

Day 6 rewrites this exact network with classes, RAII and templates, and its
explanation contains a direct comparison table — *"C: distinct function per layer"*
versus *"C++: `virtual forward()`"*. That contrast is the lesson. Refactoring this
file into classes would delete it.

> **Teaching code is meant to be read; engine code is meant to be reused.**
> `day4/cnn_forward.c` is the first. `day7/cnn.hpp` is the second. They are allowed
> to look different.

---

## 7. Where to use it & why

* **Phase:** inference (deployment).
* **Hardware:** anything without a Python interpreter — drones, cameras, phones,
  Cortex-M microcontrollers.
* **Why C specifically:** predictable memory, no runtime, no GC pause. A garbage
  collector that stalls for 10 ms during a landing sequence is not acceptable.

> The weights here are still `float`. A Cortex-M4 without an FPU spends ~20 cycles
> on every float multiply. **That's Day 5's problem** — and the reason `Memory
> (int8): 119 bytes` is already printed in the output above.

---

## ✅ Self-test

1. Write the 1D index for `tensor[c][y][x]` in a `(C, H, W)` tensor. Why is there no 2D array?
2. Why is `relu()` done in-place while `maxpool2d()` needs a separate output buffer?
3. Count the `malloc`s and `free`s in `cnn_forward.c`. How would you *prove* there's no leak?
4. What does the `- max` in softmax prevent, and why is it worse on an MCU?
5. This network needs 1.4 KB. A Cortex-M4 has 256 KB SRAM. What fraction is that, and why does the answer matter more than the FPS number?
6. Why hasn't this file been rewritten in C++ now that Day 6 exists?

<details><summary>Answers</summary>

1. `index = c*(H*W) + y*W + x`. RAM is a flat sequence of addresses; "2D" is a convention you implement with arithmetic.
2. ReLU maps each element to itself or 0 — output position matches input position, so overwriting is safe. Pooling reads 4 inputs to produce 1 output at a *different* index, so writing in place would corrupt values still needed.
3. 5 and 5. Prove it by running under AddressSanitizer (`-fsanitize=address`) — it reports any block never freed, at exit.
4. Overflow: `exp` of a large logit becomes `inf`, then `inf/inf` = `NaN`. On an MCU there's no exception handler and no debugger attached — you get silent wrong output, or a fault.
5. 0.5%. It matters more because memory is deterministic and machine-independent, whereas FPS depends on CPU, compiler and flags — so it's the number you can defend in a paper.
6. Because Day 6's entire lesson is the contrast between procedural C and object-oriented C++. Converting this file would erase the comparison it exists to support.

</details>

---

**Next:** `day5/` — kill the floats. int8 quantization, and the overflow rule that
follows you all the way to the FPGA.
