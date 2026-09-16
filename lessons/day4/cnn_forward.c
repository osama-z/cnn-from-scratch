/**
 * DAY 4 — COMPLETE CNN FORWARD PASS IN C
 * =======================================
 * 
 * This file contains EVERY operation from Day 1's Python CNN,
 * rewritten in pure C with 1D pointers.
 * 
 * Architecture (same as Day 1):
 *     Input (1, 8, 8)
 *       → Conv2D (3x3 kernel, 2 filters)
 *       → ReLU
 *       → MaxPool (2x2)
 *       → Flatten
 *       → Dense (→ 3 classes)
 *       → Softmax
 *       → Prediction
 * 
 * MEMORY LAYOUT IN C:
 * ==================
 * In Python:  image[channel][row][col]   — nice and clean
 * In C:       image[channel * H * W + row * W + col]  — ugly but FAST
 * 
 * Why? Because RAM is a single tape of bytes:
 * 
 *   Address: 0x0000  0x0004  0x0008  0x000C  0x0010 ...
 *   Data:    [px00]  [px01]  [px02]  [px10]  [px11] ...
 *            ^^^^^^^^^^^^^^^^^^^^^^^^
 *            Row 0                    Row 1
 * 
 * The CPU reads memory in LINES (cache lines). If your data is
 * stored row-by-row (row-major), the CPU can prefetch the next
 * pixels before you even ask for them. This is why C is fast.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// =============================================
// SECTION 1: MEMORY HELPERS
// =============================================

/**
 * Allocate a float array on the heap and zero it out.
 * 
 * malloc() asks the operating system for a chunk of RAM.
 * calloc() does the same but also fills it with zeros.
 * 
 * On a drone's microcontroller, you might use a static array
 * instead (no OS to ask for memory), but the concept is the same.
 */
float* alloc_zeros(int size) {
    float* ptr = (float*)calloc(size, sizeof(float));
    if (ptr == NULL) {
        printf("ERROR: Failed to allocate %d floats (%d bytes)\n", 
               size, (int)(size * sizeof(float)));
        exit(1);
    }
    return ptr;
}

// =============================================
// SECTION 2: CONVOLUTION (3x3, with zero padding)
// =============================================

/**
 * Multi-filter 2D Convolution.
 * 
 * input:   shape (in_channels, H, W) — stored as 1D array
 * kernels: shape (out_channels, in_channels, 3, 3) — stored as 1D
 * biases:  shape (out_channels,)
 * output:  shape (out_channels, H, W) — stored as 1D
 * 
 * Each output channel is computed by:
 *   1. Convolving ALL input channels with their respective kernels
 *   2. Summing the results
 *   3. Adding a bias
 * 
 * This is identical to what you wrote in Day 1 Python:
 *   for each output filter:
 *       for each input channel:
 *           output[f] += convolve(input[c], kernel[f][c])
 *       output[f] += bias[f]
 */
void conv2d(float* input, int in_ch, int H, int W,
            float* kernels, float* biases, int out_ch,
            float* output) {
    
    int k_size = 3;
    int k_half = 1;
    
    for (int oc = 0; oc < out_ch; oc++) {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                
                float sum = biases[oc];
                
                // Sum over ALL input channels
                for (int ic = 0; ic < in_ch; ic++) {
                    // Slide the 3x3 kernel
                    for (int ky = -k_half; ky <= k_half; ky++) {
                        for (int kx = -k_half; kx <= k_half; kx++) {
                            
                            int iy = y + ky;
                            int ix = x + kx;
                            
                            // Zero padding: skip if outside image
                            if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                                // 1D index for input: channel * (H*W) + row * W + col
                                int input_idx = ic * (H * W) + iy * W + ix;
                                
                                // 1D index for kernel: 
                                // out_ch * (in_ch * 3 * 3) + in_ch * (3 * 3) + ky_offset * 3 + kx_offset
                                int kernel_idx = oc * (in_ch * k_size * k_size) 
                                               + ic * (k_size * k_size) 
                                               + (ky + k_half) * k_size 
                                               + (kx + k_half);
                                
                                sum += input[input_idx] * kernels[kernel_idx];
                            }
                        }
                    }
                }
                
                // Store result: output channel * (H*W) + row * W + col
                int out_idx = oc * (H * W) + y * W + x;
                output[out_idx] = sum;
            }
        }
    }
}


// =============================================
// SECTION 3: ReLU ACTIVATION
// =============================================

/**
 * ReLU: f(x) = max(0, x)
 * 
 * The simplest function in all of deep learning.
 * In Python: output = np.maximum(0, input)
 * In C: one loop, one comparison.
 * 
 * This is IN-PLACE: it modifies the array directly (no copy).
 * On a microcontroller, saving memory = saving battery.
 */
void relu(float* data, int size) {
    for (int i = 0; i < size; i++) {
        if (data[i] < 0.0f) {
            data[i] = 0.0f;
        }
    }
}


// =============================================
// SECTION 4: MAX POOLING (2x2, stride 2)
// =============================================

/**
 * Max Pooling: Take the maximum value in each 2x2 window.
 * 
 * Input:  (channels, H, W)
 * Output: (channels, H/2, W/2)
 * 
 * This halves the spatial dimensions.
 * 
 * Why max and not average?
 *   Max says: "The strongest feature in this region is..."
 *   Average says: "The average feature is..."
 *   For edge detection, max is better because edges are sharp signals.
 *   An average would blur them away.
 */
void maxpool2d(float* input, int channels, int H, int W, float* output) {
    int out_H = H / 2;
    int out_W = W / 2;
    
    for (int c = 0; c < channels; c++) {
        for (int y = 0; y < out_H; y++) {
            for (int x = 0; x < out_W; x++) {
                
                // Find the 2x2 window in the input
                int in_y = y * 2;
                int in_x = x * 2;
                
                // Get the 4 values in the window
                float v00 = input[c * (H * W) + (in_y)     * W + (in_x)    ];
                float v01 = input[c * (H * W) + (in_y)     * W + (in_x + 1)];
                float v10 = input[c * (H * W) + (in_y + 1) * W + (in_x)    ];
                float v11 = input[c * (H * W) + (in_y + 1) * W + (in_x + 1)];
                
                // Take the maximum
                float max_val = v00;
                if (v01 > max_val) max_val = v01;
                if (v10 > max_val) max_val = v10;
                if (v11 > max_val) max_val = v11;
                
                // Store in output
                output[c * (out_H * out_W) + y * out_W + x] = max_val;
            }
        }
    }
}


// =============================================
// SECTION 5: DENSE (FULLY CONNECTED) LAYER
// =============================================

/**
 * Dense layer: output = input @ weights + bias
 * 
 * This is matrix-vector multiplication.
 * Every input is connected to every output.
 * 
 * input:   shape (in_size,)
 * weights: shape (in_size, out_size) — stored row-major
 * bias:    shape (out_size,)
 * output:  shape (out_size,)
 * 
 * For each output neuron j:
 *     output[j] = sum(input[i] * weights[i * out_size + j]) + bias[j]
 */
void dense(float* input, int in_size,
           float* weights, float* bias, int out_size,
           float* output) {
    
    for (int j = 0; j < out_size; j++) {
        float sum = bias[j];
        for (int i = 0; i < in_size; i++) {
            sum += input[i] * weights[i * out_size + j];
        }
        output[j] = sum;
    }
}


// =============================================
// SECTION 6: SOFTMAX
// =============================================

/**
 * Softmax: converts raw scores to probabilities.
 * 
 * Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j))
 * 
 * Numerical trick: subtract max(x) before exp() to prevent overflow.
 * Without this trick, exp(100) = infinity and your drone crashes.
 */
void softmax_inplace(float* data, int size) {
    // Step 1: Find maximum (for numerical stability)
    float max_val = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] > max_val) max_val = data[i];
    }
    
    // Step 2: Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        data[i] = expf(data[i] - max_val);
        sum += data[i];
    }
    
    // Step 3: Normalize (divide by sum)
    for (int i = 0; i < size; i++) {
        data[i] /= sum;
    }
}


// =============================================
// SECTION 7: ARGMAX (find the predicted class)
// =============================================

int argmax(float* data, int size) {
    int max_idx = 0;
    float max_val = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    return max_idx;
}


// =============================================
// SECTION 8: PRINT HELPERS
// =============================================

void print_tensor(float* data, int channels, int H, int W, const char* name) {
    printf("\n%s (shape: %d x %d x %d):\n", name, channels, H, W);
    for (int c = 0; c < channels; c++) {
        if (channels > 1) printf("  Channel %d:\n", c);
        for (int y = 0; y < H; y++) {
            printf("    ");
            for (int x = 0; x < W; x++) {
                printf("%6.2f ", data[c * (H * W) + y * W + x]);
            }
            printf("\n");
        }
    }
}

void print_vector(float* data, int size, const char* name) {
    printf("\n%s: [", name);
    for (int i = 0; i < size; i++) {
        printf("%.4f", data[i]);
        if (i < size - 1) printf(", ");
    }
    printf("]\n");
}


// =============================================
// SECTION 9: THE FULL CNN FORWARD PASS
// =============================================

int main() {
    clock_t start, end;
    
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║     DAY 4: COMPLETE CNN FORWARD PASS IN PURE C          ║\n");
    printf("║     Conv → ReLU → Pool → Dense → Softmax               ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");
    
    // ─── Network dimensions ───
    int in_ch = 1, H = 8, W = 8;       // Input: 1 channel, 8x8
    int conv_filters = 2;               // Conv: 2 output filters
    int pool_H = H / 2, pool_W = W / 2; // After pool: 4x4
    int flat_size = conv_filters * pool_H * pool_W;  // 2 * 4 * 4 = 32
    int num_classes = 3;
    
    printf("\n── Architecture ──\n");
    printf("  Input:     (%d, %d, %d) = %d values\n", in_ch, H, W, in_ch*H*W);
    printf("  Conv3x3:   %d filters → (%d, %d, %d) = %d values\n", 
           conv_filters, conv_filters, H, W, conv_filters*H*W);
    printf("  ReLU:      in-place (no extra memory)\n");
    printf("  MaxPool:   2x2 → (%d, %d, %d) = %d values\n", 
           conv_filters, pool_H, pool_W, flat_size);
    printf("  Flatten:   → (%d,)\n", flat_size);
    printf("  Dense:     %d → %d\n", flat_size, num_classes);
    printf("  Softmax:   → probabilities\n");
    
    int total_params = conv_filters * in_ch * 3 * 3   // conv weights
                     + conv_filters                     // conv biases
                     + flat_size * num_classes           // dense weights
                     + num_classes;                      // dense biases
    printf("  Total parameters: %d\n", total_params);
    printf("  Memory (float32): %d bytes\n", total_params * 4);
    printf("  Memory (int8):    %d bytes\n", total_params);
    
    // ─── Allocate all memory ───
    float* input      = alloc_zeros(in_ch * H * W);
    float* conv_out   = alloc_zeros(conv_filters * H * W);
    float* pool_out   = alloc_zeros(flat_size);
    float* dense_out  = alloc_zeros(num_classes);
    
    // ─── Initialize weights (same as Day 1 Python) ───
    // Conv kernels: 2 filters, 1 input channel, 3x3
    float conv_kernels[] = {
        // Filter 0: Horizontal edge detector
        -1, -1, -1,
         0,  0,  0,
         1,  1,  1,
        // Filter 1: Vertical edge detector
        -1,  0,  1,
        -1,  0,  1,
        -1,  0,  1
    };
    float conv_biases[] = {0.0f, 0.0f};
    
    // Dense weights: 32 inputs → 3 outputs (random-ish for demo)
    float* dense_weights = alloc_zeros(flat_size * num_classes);
    float dense_bias[] = {0.1f, -0.1f, 0.0f};
    
    // Initialize dense weights with a simple pattern
    srand(42);
    for (int i = 0; i < flat_size * num_classes; i++) {
        dense_weights[i] = ((float)(rand() % 100) / 100.0f - 0.5f) * 0.1f;
    }
    
    // ─── Create 3 test images (same as Day 1) ───
    const char* class_names[] = {"Horizontal Edge", "Vertical Edge", "Uniform"};
    
    // Image 0: Horizontal edge (top dark, bottom bright)
    float image0[] = {
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        10,10,10,10,10,10,10,10,
        10,10,10,10,10,10,10,10,
        10,10,10,10,10,10,10,10,
        10,10,10,10,10,10,10,10
    };
    
    // Image 1: Vertical edge (left dark, right bright)
    float image1[] = {
        0, 0, 0, 0, 10,10,10,10,
        0, 0, 0, 0, 10,10,10,10,
        0, 0, 0, 0, 10,10,10,10,
        0, 0, 0, 0, 10,10,10,10,
        0, 0, 0, 0, 10,10,10,10,
        0, 0, 0, 0, 10,10,10,10,
        0, 0, 0, 0, 10,10,10,10,
        0, 0, 0, 0, 10,10,10,10
    };
    
    // Image 2: Uniform (all 5s)
    float image2[64];
    for (int i = 0; i < 64; i++) image2[i] = 5.0f;
    
    float* images[] = {image0, image1, image2};
    
    // ─── Run inference on each image ───
    printf("\n══════════════════════════════════════════════════════════\n");
    printf("  RUNNING INFERENCE ON 3 TEST IMAGES\n");
    printf("══════════════════════════════════════════════════════════\n");
    
    start = clock();
    
    for (int img_idx = 0; img_idx < 3; img_idx++) {
        printf("\n── Image %d: %s ──\n", img_idx, class_names[img_idx]);
        
        // Copy image to input buffer
        memcpy(input, images[img_idx], in_ch * H * W * sizeof(float));
        
        // STEP 1: Convolution
        conv2d(input, in_ch, H, W, conv_kernels, conv_biases, conv_filters, conv_out);
        
        if (img_idx == 0) {
            print_tensor(conv_out, conv_filters, H, W, "  After Conv2D");
        }
        
        // STEP 2: ReLU (in-place)
        relu(conv_out, conv_filters * H * W);
        
        // STEP 3: Max Pooling
        maxpool2d(conv_out, conv_filters, H, W, pool_out);
        
        if (img_idx == 0) {
            print_tensor(pool_out, conv_filters, pool_H, pool_W, "  After ReLU + MaxPool");
        }
        
        // STEP 4: Dense (pool_out is already flattened — it's a 1D array!)
        dense(pool_out, flat_size, dense_weights, dense_bias, num_classes, dense_out);
        
        // STEP 5: Softmax
        softmax_inplace(dense_out, num_classes);
        
        // Print prediction
        int predicted = argmax(dense_out, num_classes);
        printf("  Prediction: [");
        for (int c = 0; c < num_classes; c++) {
            printf("%.1f%%", dense_out[c] * 100.0f);
            if (c < num_classes - 1) printf(", ");
        }
        printf("] → Class %d (%s)\n", predicted, class_names[predicted]);
    }
    
    end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    // ─── Performance Analysis ───
    printf("\n══════════════════════════════════════════════════════════\n");
    printf("  PERFORMANCE ANALYSIS\n");
    printf("══════════════════════════════════════════════════════════\n");
    printf("  3 images processed in: %.3f ms\n", time_ms);
    printf("  Per image:             %.3f ms\n", time_ms / 3.0);
    printf("  Estimated FPS:         %.0f FPS\n", 3000.0 / time_ms);
    
    // ─── Memory Analysis ───
    int total_memory = (in_ch*H*W + conv_filters*H*W + flat_size + num_classes) * sizeof(float);
    printf("\n  Memory usage:\n");
    printf("    Input buffer:   %4d bytes\n", (int)(in_ch*H*W*sizeof(float)));
    printf("    Conv output:    %4d bytes\n", (int)(conv_filters*H*W*sizeof(float)));
    printf("    Pool output:    %4d bytes\n", (int)(flat_size*sizeof(float)));
    printf("    Dense output:   %4d bytes\n", (int)(num_classes*sizeof(float)));
    printf("    TOTAL buffers:  %4d bytes\n", total_memory);
    printf("    Weights:        %4d bytes\n", (int)(total_params*sizeof(float)));
    printf("    GRAND TOTAL:    %4d bytes (%.1f KB)\n", 
           total_memory + total_params*4, 
           (total_memory + total_params*4) / 1024.0f);
    
    // ─── Compare with Python ───
    printf("\n══════════════════════════════════════════════════════════\n");
    printf("  C vs PYTHON COMPARISON\n");
    printf("══════════════════════════════════════════════════════════\n");
    printf("  ┌──────────────────┬──────────────┬──────────────┐\n");
    printf("  │ Metric           │ Python       │ C            │\n");
    printf("  ├──────────────────┼──────────────┼──────────────┤\n");
    printf("  │ Language level   │ High-level   │ Low-level    │\n");
    printf("  │ Memory control   │ Automatic    │ Manual       │\n");
    printf("  │ Array indexing   │ img[c][y][x] │ ptr[c*H*W+  │\n");
    printf("  │                  │              │     y*W+x]   │\n");
    printf("  │ Memory mgmt     │ Garbage coll │ malloc/free  │\n");
    printf("  │ Runs on MCU?    │ No           │ YES          │\n");
    printf("  │ Binary size      │ ~50MB (venv) │ ~20KB        │\n");
    printf("  └──────────────────┴──────────────┴──────────────┘\n");
    
    // ─── Cleanup ───
    free(input);
    free(conv_out);
    free(pool_out);
    free(dense_out);
    free(dense_weights);
    
    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║              DAY 4 COMPLETE                              ║\n");
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║                                                          ║\n");
    printf("║  What you built:                                         ║\n");
    printf("║    ✓ conv2d()       — Multi-channel convolution          ║\n");
    printf("║    ✓ relu()         — In-place activation                ║\n");
    printf("║    ✓ maxpool2d()    — 2x2 downsampling                   ║\n");
    printf("║    ✓ dense()        — Matrix-vector multiplication       ║\n");
    printf("║    ✓ softmax()      — Probability conversion             ║\n");
    printf("║    ✓ argmax()       — Class prediction                   ║\n");
    printf("║                                                          ║\n");
    printf("║  Key C concepts learned:                                 ║\n");
    printf("║    ✓ 1D pointer indexing: ptr[c*H*W + y*W + x]          ║\n");
    printf("║    ✓ malloc/free: manual memory management               ║\n");
    printf("║    ✓ In-place operations: save memory on MCU             ║\n");
    printf("║    ✓ clock() timing: measure performance                 ║\n");
    printf("║                                                          ║\n");
    printf("║  This C code can run on a bare-metal ARM chip.           ║\n");
    printf("║  Python CANNOT. That is why embedded AI uses C.          ║\n");
    printf("║                                                          ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");
    
    return 0;
}
