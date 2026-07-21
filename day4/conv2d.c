/**
 * DAY 4: The Hardware Reality (Convolution in C)
 * 
 * Goal: Rewrite the Python convolution from Day 1 in pure C.
 * 
 * Why? 
 * Because Python is slow and runs on massive servers.
 * C is fast, talks directly to memory, and runs on microcontrollers (drones).
 * 
 * Key Concept: 1D Pointers
 * In C, a 2D image (like 5x5) is not actually a 2D grid in RAM.
 * RAM is just one long line of memory.
 * We must use 1D pointers to simulate a 2D grid.
 * Formula: index = (row * width) + col
 */

#include <stdio.h>
#include <stdlib.h> // For malloc and free

// =============================================
// SECTION 1: THE CONVOLUTION FUNCTION
// =============================================

/**
 * Performs 2D convolution on a 1D array representing a 2D image.
 * 
 * @param input   Pointer to the 1D array of image pixels
 * @param width   Width of the image
 * @param height  Height of the image
 * @param kernel  Pointer to the 3x3 kernel (always 9 elements)
 * @param output  Pointer to the 1D array where results are stored
 */
void conv2d(float* input, int width, int height, float* kernel, float* output) {
    // The kernel is 3x3, so the center is at offset 1
    int k_size = 3;
    int k_half = k_size / 2; // 3/2 = 1 (integer division)

    // Slide over every pixel in the image
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            
            float sum = 0.0f;
            
            // Loop through the 3x3 kernel
            for (int ky = -k_half; ky <= k_half; ky++) {
                for (int kx = -k_half; kx <= k_half; kx++) {
                    
                    // Find the absolute coordinates in the image
                    int img_y = y + ky;
                    int img_x = x + kx;
                    
                    // Zero Padding check: If the kernel goes off the edge of the image, treat it as 0
                    if (img_y >= 0 && img_y < height && img_x >= 0 && img_x < width) {
                        
                        // MAGIC FORMULA: Convert 2D coordinates to 1D index
                        int img_index = (img_y * width) + img_x;
                        
                        // Convert 2D kernel coordinates to 1D kernel index
                        // ky is [-1, 0, 1] -> ky+1 is [0, 1, 2]
                        int k_index = ((ky + k_half) * k_size) + (kx + k_half);
                        
                        // Multiply and accumulate
                        sum += input[img_index] * kernel[k_index];
                    }
                }
            }
            
            // Store the result in the output array (1D index)
            int out_index = (y * width) + x;
            output[out_index] = sum;
        }
    }
}

// =============================================
// SECTION 2: PRINTING UTILITY
// =============================================

void print_image(float* img, int width, int height, const char* title) {
    printf("\n%s:\n", title);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = (y * width) + x;
            // Print neatly formatted floats
            printf("%6.1f ", img[index]);
        }
        printf("\n");
    }
}

// =============================================
// SECTION 3: THE MAIN PROGRAM (TESTING)
// =============================================

int main() {
    printf("============================================================\n");
    printf("DAY 4: C CONVOLUTION — POINTERS AND MEMORY\n");
    printf("============================================================\n");
    
    int width = 5;
    int height = 5;
    
    // 1. Allocate memory for input and output on the HEAP using malloc
    // Why? Because images get large. The stack will overflow if you put 60,000 images on it.
    float* image = (float*)malloc(width * height * sizeof(float));
    float* output = (float*)malloc(width * height * sizeof(float));
    
    // Create a 5x5 image with a vertical line of 1s in the middle, 0s elsewhere
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = (y * width) + x;
            if (x == 2) {
                image[index] = 10.0f; // Vertical line
            } else {
                image[index] = 0.0f;
            }
        }
    }
    
    // 2. Define our Edge Detector Kernel (3x3)
    // -1  0  1
    // -1  0  1
    // -1  0  1
    float kernel[9] = {
        -1.0f,  0.0f,  1.0f,
        -1.0f,  0.0f,  1.0f,
        -1.0f,  0.0f,  1.0f
    };
    
    print_image(image, width, height, "Input Image (Vertical Line at x=2)");
    
    // 3. RUN THE CONVOLUTION
    printf("\nRunning Convolution... (Simulating hardware math)\n");
    conv2d(image, width, height, kernel, output);
    
    // 4. Show the result
    print_image(output, width, height, "Output Feature Map (Vertical Edge Detection)");
    
    // 5. MEMORY MANAGEMENT (CRITICAL IN C)
    // If you don't free memory on a drone, it crashes (Memory Leak).
    free(image);
    free(output);
    
    printf("\n============================================================\n");
    printf("DAY 4 COMPLETE: Python logic successfully rewritten in pure C.\n");
    printf("============================================================\n");
    
    return 0;
}
