# 🔢 Day 5: Fixed-Point Arithmetic (Killing the Float)

> **Core Concept:** Microcontrollers on drones are extremely resource-constrained.
> They hate `float` numbers. This day is about replacing ALL floats with `int8`
> — making the CNN 4x smaller in memory and 20x faster to compute.

---

## 1. Why Floats Are a Problem on Hardware

A `float32` number looks simple to you. But to a CPU, it is a nightmare:

```text
float 0.35 stored in RAM (32 bits):

  Bit 31   Bits 30-23    Bits 22-0
  ┌────┬────────────┬───────────────────────┐
  │ 0  │  01111101  │  01100110011001100110 │
  └────┴────────────┴───────────────────────┘
   Sign   Exponent         Mantissa

To MULTIPLY two floats, the CPU does 5 steps:
  1. Extract both exponents
  2. Add the exponents together
  3. Multiply the two 23-bit mantissas
  4. Normalize the result (handle overflow)
  5. Reassemble into float format
  → Total: ~20 clock cycles on ARM Cortex-M4

To MULTIPLY two int8 numbers (e.g. 45 * 33 = 1485):
  1. Just multiply.
  → Total: 1 clock cycle

Speedup: 20x per operation.
A convolution layer does MILLIONS of multiplications.
int8 CNN = 20x faster on the same hardware.
```

---

## 2. The Quantization Idea (The Core Trick)

The number `0.35` cannot live in `int8`. But we can **pretend** it is an integer by agreeing on a scaling factor:

```text
QUANTIZE (float → int8):
  float_value  = 0.35
  scale        = 0.00627  (computed from the range of all weights)
  zero_point   = 0
  int8_value   = round(0.35 / 0.00627) + 0 = round(55.8) = 56

DEQUANTIZE (int8 → float):
  float_value  = (56 - 0) * 0.00627 = 0.3512

ERROR: |0.35 - 0.3512| = 0.0012  (tiny!)
```

This tiny error (~0.1%) is called **Quantization Error**. It costs us ~1.5% accuracy. In return, we get a model that is:
- **4x smaller** in RAM (4 bytes → 1 byte per weight)
- **20x faster** in compute (float multiply → int multiply)
- **Deployable on a $2 microcontroller** instead of a server

---

## 3. The Industry Formula (Memorize This)

```text
QUANTIZE:
    int8_val = round(float_val / scale) + zero_point

DEQUANTIZE:
    float_val = (int8_val - zero_point) * scale

Finding scale and zero_point from your data:
    scale      = (max_val - min_val) / 255.0
    zero_point = round(-min_val / scale) - 128
```

This is exactly what **TensorFlow Lite**, **CMSIS-NN**, and **ONNX Runtime** use internally.

---

## 4. Int8 Convolution — The Critical Detail

When you multiply two `int8` numbers, the result can overflow `int8`:
```text
int8 range:  -128 to 127
127 * 127  = 16,129  ← Does NOT fit in int8!

For a 3x3 kernel: sum of 9 products:
16,129 * 9 = 145,161  ← Needs int32 to hold this!

So the rule for int8 convolution:
  1. Read two int8 values
  2. Multiply → result fits in int32 (it's widened automatically)
  3. ACCUMULATE into int32 accumulator (never overflow)
  4. After all 9 kernel multiplications are done,
     scale the int32 accumulator back down to int8
     (this is called "Re-quantization")
```

In C code:
```c
int32_t acc = bias;                        // int32 accumulator
acc += (int32_t)input_val * kernel_val;    // int8 * int8 → int32 (safe!)
// ... repeat for all 9 kernel positions ...
output = clamp(round(acc * scale), -128, 127);  // scale back to int8
```

---

## 5. ReLU in Int8 — A Subtle Trap!

This is where most beginners make a mistake:

```text
In float:  relu(x) = max(0.0, x)
In int8:   relu(x) = max(zero_point, x)   ← NOT max(0, x) !!

WHY? The integer "0" does NOT represent the float value 0.0.
     The zero_point integer is what represents float 0.0.

Example with scale=0.006, zero_point=128:
  int8 value 128 → dequantized = (128 - 128) * 0.006 = 0.0 ✓
  int8 value 0   → dequantized = (0   - 128) * 0.006 = -0.768 ✗
```

If you use `max(0, x)` instead of `max(zero_point, x)` in your int8 ReLU, your AI will produce silent, wrong answers. No crash — just wrong predictions.

---

## 6. The Float → Int8 Trade-Off Table

| Metric | float32 | int8 | Gain |
| :--- | :--- | :--- | :--- |
| Bytes per weight | 4 bytes | 1 byte | **4x smaller** |
| MNIST model size | 427 KB | 107 KB | **4x smaller** |
| Multiply speed | ~20 cycles | ~1 cycle | **20x faster** |
| Accuracy (MNIST) | 95.0% | ~93.5% | -1.5% loss |
| Fits on Cortex-M4? | Maybe | **YES** | ✓ |

---

## 7. Where to Use It & Why

*   **The Phase:** Deployment (after training is fully done)
*   **The Hardware:** Any embedded chip — ARM Cortex-M4/M7, Raspberry Pi, ESP32, STM32.
*   **The Workflow:**
    1. Train your model in Python with `float32` (full precision for best accuracy)
    2. Quantize the trained weights to `int8` (small + fast for deployment)
    3. Deploy ONLY the `int8` model to the drone
    4. The drone runs inference in `int8` — tiny RAM, fast, low battery drain

> **The Golden Rule:** Train in float. Deploy in int8.
