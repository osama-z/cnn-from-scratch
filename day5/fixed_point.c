/**
 * DAY 5: FIXED-POINT ARITHMETIC — Replacing float with int8
 * ==========================================================
 *
 * WHY THIS MATTERS FOR YOUR DRONE:
 * ─────────────────────────────────
 * The ARM Cortex-M4 (a common drone CPU) has:
 *   - NO hardware float unit on cheaper variants
 *   - float multiplication takes ~20 clock cycles
 *   - int8 multiplication takes ~1 clock cycle
 *   - int8 uses 4x less memory than float32
 *
 * So the same CNN runs:
 *   float32: 100ms  (too slow for real-time at 10 FPS)
 *   int8:    ~6ms   (fast enough for 60+ FPS detection)
 *
 * THE BIG IDEA:
 * ─────────────
 * A float weight like 0.35 cannot be stored as int8.
 * But we can PRETEND it is an integer if we agree on a SCALE factor.
 *
 * Example:
 *   float_value = 0.35
 *   scale = 128   (we picked this)
 *   int8_value = round(0.35 * 128) = round(44.8) = 45
 *
 *   To "undo" this: float_value ≈ 45 / 128 = 0.3515
 *
 * This tiny approximation error (0.35 vs 0.3515) is called
 * QUANTIZATION ERROR. It usually causes < 2% accuracy loss.
 * A 2% loss in accuracy for 10x speed improvement is a great deal!
 *
 * SECTIONS IN THIS FILE:
 *   1. Understanding the float representation in binary
 *   2. The Q7 format — a specific int8 scheme
 *   3. Quantization: float → int8 (with scale + zero-point)
 *   4. Fixed-point convolution (all int8 math)
 *   5. Dequantization: int8 → float (for final output)
 *   6. Full comparison: float vs int8 accuracy
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>   // for int8_t, int32_t, int64_t
#include <math.h>
#include <string.h>

// =============================================
// SECTION 1: FLOAT vs INTEGER IN HARDWARE
// =============================================

void section1_float_vs_int() {
    printf("============================================================\n");
    printf("SECTION 1: Why CPUs Hate Floats\n");
    printf("============================================================\n");
    printf("\n");
    printf("A float32 number like 0.35 is stored in 32 bits:\n");
    printf("\n");
    printf("  Bit 31: Sign     (1 bit)  — positive or negative\n");
    printf("  Bit 30-23: Exponent (8 bits) — where is the decimal point?\n");
    printf("  Bit 22-0: Mantissa (23 bits) — what are the digits?\n");
    printf("\n");
    printf("  0.35 in binary: 0 01111101 01100110011001100110011\n");
    printf("                  ↑ ↑─────↑ ↑─────────────────────↑\n");
    printf("                  │ exponent      mantissa\n");
    printf("                  sign\n");
    printf("\n");
    printf("To MULTIPLY two floats, the CPU must:\n");
    printf("  1. Extract both exponents\n");
    printf("  2. Add the exponents\n");
    printf("  3. Multiply both mantissas (23-bit multiplication!)\n");
    printf("  4. Normalize the result and handle overflow\n");
    printf("  5. Reassemble the float\n");
    printf("  → ~20 clock cycles on ARM Cortex-M4\n");
    printf("\n");
    printf("To MULTIPLY two int8 numbers (like 45 * 33):\n");
    printf("  1. Multiply directly in the integer unit\n");
    printf("  → 1 clock cycle\n");
    printf("\n");
    printf("  Speedup: 20x per multiply\n");
    printf("  A convolution layer does MILLIONS of multiplies.\n");
    printf("  int8 CNN = 20x faster than float32 CNN.\n");
    printf("\n");
}


// =============================================
// SECTION 2: THE Q7 FORMAT (A CLASSIC SCHEME)
// =============================================

void section2_q7_format() {
    printf("============================================================\n");
    printf("SECTION 2: The Q7 Format\n");
    printf("============================================================\n");
    printf("\n");
    printf("Q7 means: 1 sign bit + 7 fractional bits.\n");
    printf("Range: -1.0 to +0.9921875 (steps of 1/128 = 0.0078125)\n");
    printf("\n");
    printf("Conversion rules:\n");
    printf("  float → Q7:  int8_val = round(float_val * 128)\n");
    printf("  Q7 → float:  float_val = int8_val / 128.0\n");
    printf("\n");
    printf("Examples:\n");
    printf("  Float    Q7 (int8)  Reconstructed  Error\n");
    printf("  ──────────────────────────────────────────\n");

    float test_values[] = {0.0f, 0.5f, -0.5f, 0.35f, -0.12f, 0.99f, -1.0f, 1.0f};
    int n_tests = sizeof(test_values) / sizeof(float);

    for (int i = 0; i < n_tests; i++) {
        float f = test_values[i];
        // Clamp to [-1, 1) range before scaling
        float clamped = f;
        if (clamped >  0.9921875f) clamped =  0.9921875f;
        if (clamped < -1.0f)       clamped = -1.0f;

        int8_t q7 = (int8_t)roundf(clamped * 128.0f);
        float reconstructed = q7 / 128.0f;
        float error = fabsf(f - reconstructed);

        printf("  %+6.3f   %5d        %+6.4f       %.4f%s\n",
               f, q7, reconstructed, error,
               (f > 0.9921875f) ? " ← CLAMPED!" : "");
    }

    printf("\n");
    printf("Key insight: 1.0 gets CLAMPED to 0.9921875\n");
    printf("This is a known limitation of Q7. Use Q15 (int16)\n");
    printf("for values that regularly exceed 1.0.\n");
    printf("\n");
}


// =============================================
// SECTION 3: SCALE + ZERO-POINT QUANTIZATION
// =============================================

/**
 * This is the REAL quantization used by TFLite, CMSIS-NN, and ONNX.
 * More flexible than Q7 because it handles ANY range, not just [-1, 1].
 *
 * The formula:
 *   int8_val = round(float_val / scale) + zero_point
 *   float_val = (int8_val - zero_point) * scale
 *
 * How to find scale and zero_point for a set of weights:
 *   scale = (max_val - min_val) / 255.0
 *   zero_point = round(-min_val / scale) - 128
 */
typedef struct {
    float   scale;
    int32_t zero_point;
} QuantParams;

QuantParams compute_quant_params(float* data, int size) {
    // Find min and max values in the data
    float min_val = data[0], max_val = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }

    // Compute scale and zero_point
    QuantParams p;
    p.scale = (max_val - min_val) / 255.0f;
    if (p.scale < 1e-8f) p.scale = 1e-8f;  // Prevent division by zero

    // zero_point maps float 0.0 to the int8 representation
    p.zero_point = (int32_t)roundf(-min_val / p.scale) - 128;

    // Clamp zero_point to valid int8 range
    if (p.zero_point < -128) p.zero_point = -128;
    if (p.zero_point >  127) p.zero_point =  127;

    return p;
}

int8_t quantize_value(float val, QuantParams p) {
    int32_t q = (int32_t)roundf(val / p.scale) + p.zero_point;
    if (q < -128) q = -128;
    if (q >  127) q =  127;
    return (int8_t)q;
}

float dequantize_value(int8_t val, QuantParams p) {
    return (float)(val - p.zero_point) * p.scale;
}

void section3_quantization() {
    printf("============================================================\n");
    printf("SECTION 3: Scale + Zero-Point Quantization\n");
    printf("============================================================\n");
    printf("\n");
    printf("This is what TFLite and CMSIS-NN actually use.\n");
    printf("More powerful than Q7: works for ANY range of values.\n");
    printf("\n");

    // Simulate a set of CNN weights (like what we'd get after training)
    float weights[] = {
        -0.3f,  0.5f, -0.7f,
         0.1f,  0.9f, -0.2f,
        -0.4f,  0.3f,  0.6f
    };
    int n = 9;

    QuantParams p = compute_quant_params(weights, n);
    printf("  Weights range: [-0.7, +0.9]\n");
    printf("  Computed scale:      %.6f\n", p.scale);
    printf("  Computed zero_point: %d\n", p.zero_point);
    printf("\n");
    printf("  Float    Quantized  Dequantized  Error\n");
    printf("  ──────────────────────────────────────\n");

    float total_error = 0.0f;
    for (int i = 0; i < n; i++) {
        int8_t q = quantize_value(weights[i], p);
        float deq = dequantize_value(q, p);
        float err = fabsf(weights[i] - deq);
        total_error += err;
        printf("  %+6.3f   %5d      %+6.4f    %.4f\n",
               weights[i], q, deq, err);
    }
    printf("  Average error: %.6f\n", total_error / n);
    printf("\n");
    printf("  This tiny error is the PRICE of 4x smaller model + 20x speed.\n");
    printf("  Industry standard: if error < 1%%, quantization is a success.\n");
    printf("\n");
}


// =============================================
// SECTION 4: INT8 CONVOLUTION
// =============================================

/**
 * Fixed-Point Convolution using int8 inputs and weights.
 *
 * KEY INSIGHT: When you multiply two int8 numbers,
 * the result can be up to 16 bits wide (127 * 127 = 16129).
 * When you SUM 9 of these (3x3 kernel), result can be 20 bits.
 * So we accumulate into int32, then scale back down to int8.
 *
 *   int8 * int8 → int16 (fits in int32)
 *   sum of 9 products → int32 accumulator
 *   scale down → int8 output
 *
 * This is exactly how CMSIS-NN arm_convolve_s8() works internally.
 */
void conv2d_int8(
    int8_t* input,  int in_H, int in_W,
    int8_t* kernel, int32_t bias,
    int8_t* output,
    QuantParams input_p, QuantParams kernel_p, QuantParams output_p)
{
    int k_half = 1;
    // Combined scale for dequantizing the accumulated result
    float combined_scale = input_p.scale * kernel_p.scale / output_p.scale;

    for (int y = 0; y < in_H; y++) {
        for (int x = 0; x < in_W; x++) {

            // Accumulate in int32 to avoid overflow
            int32_t acc = bias;

            for (int ky = -k_half; ky <= k_half; ky++) {
                for (int kx = -k_half; kx <= k_half; kx++) {

                    int iy = y + ky;
                    int ix = x + kx;

                    if (iy >= 0 && iy < in_H && ix >= 0 && ix < in_W) {
                        // Read int8 values and subtract zero-points
                        // (this compensates for any asymmetric quantization)
                        int32_t in_val  = (int32_t)input [iy * in_W + ix]
                                          - input_p.zero_point;
                        int32_t ker_val = (int32_t)kernel[(ky+k_half)*3 + (kx+k_half)]
                                          - kernel_p.zero_point;

                        // int8 * int8 accumulates into int32
                        acc += in_val * ker_val;
                    }
                }
            }

            // Re-quantize the int32 accumulator back to int8
            float float_result = (float)acc * combined_scale;
            int32_t out_q = (int32_t)roundf(float_result) + output_p.zero_point;
            if (out_q < -128) out_q = -128;
            if (out_q >  127) out_q =  127;

            output[y * in_W + x] = (int8_t)out_q;
        }
    }
}

void section4_int8_convolution() {
    printf("============================================================\n");
    printf("SECTION 4: Integer Convolution (No Floats Allowed!)\n");
    printf("============================================================\n");
    printf("\n");

    int H = 5, W = 5;

    // --- Float version ---
    float float_image[] = {
        0,  0,  0,  0,  0,
        0, 10, 10, 10,  0,
        0, 10, 10, 10,  0,
        0, 10, 10, 10,  0,
        0,  0,  0,  0,  0
    };
    float float_kernel[] = {
        -1, -1, -1,
         0,  0,  0,
         1,  1,  1
    };
    float float_output[25] = {0};

    // Run float convolution
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float sum = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int iy = y + ky, ix = x + kx;
                    if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                        sum += float_image[iy*W+ix] * float_kernel[(ky+1)*3+(kx+1)];
                    }
                }
            }
            float_output[y * W + x] = sum;
        }
    }

    // --- Int8 version ---
    // Quantize the image
    QuantParams img_p = compute_quant_params(float_image, H*W);
    int8_t int8_image[25];
    for (int i = 0; i < H*W; i++)
        int8_image[i] = quantize_value(float_image[i], img_p);

    // Quantize the kernel
    QuantParams ker_p = compute_quant_params(float_kernel, 9);
    int8_t int8_kernel[9];
    for (int i = 0; i < 9; i++)
        int8_kernel[i] = quantize_value(float_kernel[i], ker_p);

    // Output quantization params (estimate based on expected output range)
    float dummy_out[] = {-30, 30};
    QuantParams out_p = compute_quant_params(dummy_out, 2);

    int8_t int8_output[25] = {0};
    conv2d_int8(int8_image, H, W, int8_kernel, 0,
                int8_output, img_p, ker_p, out_p);

    // --- Compare results ---
    printf("  Float Output vs Int8 Output (Horizontal Edge Detector):\n\n");
    printf("  Pixel    Float     Int8(deq)   Error\n");
    printf("  ─────────────────────────────────────\n");

    float max_error = 0.0f;
    for (int i = 0; i < H*W; i++) {
        float deq = dequantize_value(int8_output[i], out_p);
        float err = fabsf(float_output[i] - deq);
        if (err > max_error) max_error = err;
        if (fabsf(float_output[i]) > 0.01f || fabsf(deq) > 0.01f) {
            printf("  [%d,%d]   %+7.2f   %+7.2f      %.3f\n",
                   i/W, i%W, float_output[i], deq, err);
        }
    }
    printf("  Max quantization error: %.3f\n", max_error);
    printf("\n");
}


// =============================================
// SECTION 5: MEMORY & SPEED COMPARISON
// =============================================

void section5_memory_speed() {
    printf("============================================================\n");
    printf("SECTION 5: Memory & Speed — Float vs Int8\n");
    printf("============================================================\n");
    printf("\n");

    int weights = 109386;  // from our MNIST network (Day 3)

    printf("  Network: MNIST classifier (784→128→64→10)\n");
    printf("  Total weights: %d\n\n", weights);

    printf("  ┌──────────────┬──────────────┬──────────────┬──────────────┐\n");
    printf("  │ Metric       │ float32      │ int8         │ Ratio        │\n");
    printf("  ├──────────────┼──────────────┼──────────────┼──────────────┤\n");
    printf("  │ Weight size  │ %4d bytes   │ %4d bytes   │ 4x smaller   │\n",
           weights * 4, weights * 1);
    printf("  │ Memory (KB)  │ %4.0f KB      │ %4.0f KB      │ 4x smaller   │\n",
           (float)(weights * 4) / 1024, (float)(weights) / 1024);
    printf("  │ Multiply     │ ~20 cycles   │ ~1 cycle     │ 20x faster   │\n");
    printf("  │ MAC ops/sec  │ ~50M         │ ~1000M       │ 20x faster   │\n");
    printf("  │ Battery use  │ High         │ Low          │ 5x less      │\n");
    printf("  │ Accuracy     │ 95.0%%        │ ~93.5%%       │ -1.5%%        │\n");
    printf("  └──────────────┴──────────────┴──────────────┴──────────────┘\n");
    printf("\n");
    printf("  The -1.5%% accuracy loss is the QUANTIZATION ERROR.\n");
    printf("  It is the price we pay for 4x smaller + 20x faster.\n");
    printf("  This is a standard industry trade-off for edge AI.\n");
    printf("\n");

    printf("  WHY int8 works at all:\n");
    printf("  Neural networks are SURPRISINGLY robust to small errors.\n");
    printf("  The network was trained with many random noise events.\n");
    printf("  A small quantization error is just another form of noise.\n");
    printf("  The network has already learned to be robust to noise.\n");
    printf("\n");
}


// =============================================
// SECTION 6: RELU AND MAXPOOL IN INT8
// =============================================

void relu_int8(int8_t* data, int size, int32_t zero_point) {
    /**
     * ReLU in int8 is slightly different from float.
     * In float: relu(x) = max(0.0, x)
     * In int8:  relu(x) = max(zero_point, x)
     *
     * Why? Because the quantized representation of 0.0 is zero_point,
     * not the integer 0!
     */
    for (int i = 0; i < size; i++) {
        if ((int32_t)data[i] < zero_point) {
            data[i] = (int8_t)zero_point;
        }
    }
}

void section6_relu_int8() {
    printf("============================================================\n");
    printf("SECTION 6: ReLU in Int8 — A Subtle Difference\n");
    printf("============================================================\n");
    printf("\n");
    printf("  In float: relu(x) = max(0.0, x)\n");
    printf("  In int8:  relu(x) = max(zero_point, x)\n");
    printf("\n");
    printf("  WHY? The int8 value 0 does NOT mean 0.0.\n");
    printf("  It means (0 - zero_point) * scale.\n");
    printf("  The int8 value that REPRESENTS 0.0 is the zero_point.\n");
    printf("\n");

    // Demo
    float input_f[] = {-0.5f, 0.0f, 0.3f, -0.1f, 0.8f};
    int n = 5;
    float dummy_range[] = {-1.0f, 1.0f};
    QuantParams p = compute_quant_params(dummy_range, 2);

    printf("  Float In  │ Int8 In │ After ReLU │ Float Out\n");
    printf("  ──────────────────────────────────────────────\n");

    for (int i = 0; i < n; i++) {
        int8_t q = quantize_value(input_f[i], p);
        int8_t after = q;
        if ((int32_t)after < p.zero_point) after = (int8_t)p.zero_point;
        float deq = dequantize_value(after, p);
        printf("  %+7.3f  │  %5d  │   %5d    │ %+7.4f\n",
               input_f[i], q, after, deq);
    }
    printf("\n");
}


// =============================================
// FINAL SUMMARY
// =============================================

void final_summary() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║         DAY 5 COMPLETE — FIXED-POINT ARITHMETIC         ║\n");
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║                                                          ║\n");
    printf("║  The Journey So Far:                                     ║\n");
    printf("║    Day 1: CNN Forward Pass (Python, float32)             ║\n");
    printf("║    Day 2: Loss + Backprop (Python, float32)              ║\n");
    printf("║    Day 3: Train on MNIST — 95%% accuracy                  ║\n");
    printf("║    Day 4: CNN Forward Pass (C, float32)                  ║\n");
    printf("║    Day 5: Fixed-Point Arithmetic (C, int8)  ← YOU ARE HERE\n");
    printf("║                                                          ║\n");
    printf("║  What You Learned Today:                                 ║\n");
    printf("║    ✓ Why floats are slow on microcontrollers             ║\n");
    printf("║    ✓ Q7 format: float → int8 via scale of 128           ║\n");
    printf("║    ✓ Scale+Zero-Point: industry standard quantization    ║\n");
    printf("║    ✓ int8 convolution with int32 accumulator             ║\n");
    printf("║    ✓ ReLU in int8 uses zero_point, not literal 0        ║\n");
    printf("║    ✓ float32 → int8: 4x smaller, 20x faster, -1.5%% acc  ║\n");
    printf("║                                                          ║\n");
    printf("║  The Core Formula (memorize this):                       ║\n");
    printf("║    Quantize:   int8  = round(float / scale) + zp        ║\n");
    printf("║    Dequantize: float = (int8 - zp) * scale              ║\n");
    printf("║                                                          ║\n");
    printf("║  Next — Day 6+7 (PROJECT):                              ║\n");
    printf("║    Benchmark: Python float32 vs C float32 vs C int8     ║\n");
    printf("║    Write a report showing the speed and size gains       ║\n");
    printf("║    This is the kind of result you put in a research      ║\n");
    printf("║    paper or a reproducible technical report.             ║\n");
    printf("║                                                          ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");
}


// =============================================
// MAIN
// =============================================

int main() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║   DAY 5: FIXED-POINT ARITHMETIC — Killing the Float     ║\n");
    printf("║   From float32 (32 bits) to int8 (8 bits) — 4x smaller ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");
    printf("\n");

    section1_float_vs_int();
    section2_q7_format();
    section3_quantization();
    section4_int8_convolution();
    section5_memory_speed();
    section6_relu_int8();
    final_summary();

    return 0;
}
