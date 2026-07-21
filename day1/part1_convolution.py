"""
DAY 1 — PART 1: Convolution From Scratch
=========================================
Goal: Understand convolution at the NUMBER level.
      Not just "it detects edges" — understand WHY it detects edges.

What is convolution?
    It's a mathematical operation: slide a small matrix (kernel) over a big
    matrix (image), at each position multiply element-wise and sum.
    
    That's it. That's the entire foundation of CNN.
    
Technical note:
    What we implement here is actually CORRELATION, not true convolution.
    True convolution flips the kernel 180° first.
    But in deep learning, everyone says "convolution" and means correlation.
    PyTorch nn.Conv2d, TensorFlow tf.nn.conv2d — all do correlation.
    It doesn't matter because the network LEARNS the kernel values anyway.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# PART 1A: What is an image mathematically?
# An image = 2D matrix of integers (grayscale)
# Each value = pixel intensity: 0 (black) to 255 (white)
# This is a 7x7 image with a 3x3 bright square in the center

image = np.array([
    [10,  10,  10,  10,  10,  10,  10],
    [10,  10,  10,  10,  10,  10,  10],
    [10,  10, 255, 255, 255,  10,  10],
    [10,  10, 255, 255, 255,  10,  10],
    [10,  10, 255, 255, 255,  10,  10],
    [10,  10,  10,  10,  10,  10,  10],
    [10,  10,  10,  10,  10,  10,  10],
], dtype=np.float32)

print("=" * 60)
print("IMAGE AS NUMBERS:")
print("=" * 60)
print(image)
print(f"\nShape: {image.shape}  → {image.shape[0]} rows × {image.shape[1]} columns")
print(f"Total pixels: {image.size}")
print(f"Min value: {image.min():.0f} = dark")
print(f"Max value: {image.max():.0f} = bright")
print(f"Mean value: {image.mean():.1f}")

# PART 1B: Implement convolution from scratch
# This is the CORE operation. Every CNN layer does exactly this.
# YOLO does this 200+ times per frame.

def convolution_2d(image, kernel, padding_mode='constant'):
    """
    Manual 2D convolution (technically correlation).
    
    Parameters:
        image: 2D numpy array (H, W)
        kernel: 2D numpy array (kH, kW) — must be odd dimensions
        padding_mode: 'constant' (zero-pad), 'reflect', or 'edge' (replicate)
    
    Returns:
        output: 2D numpy array, same size as input (due to padding)
    
    The algorithm:
        For EVERY pixel (y, x) in the image:
            1. Extract the neighborhood around (y, x), same size as kernel
            2. Multiply each neighborhood pixel by the corresponding kernel value
            3. Sum all the products
            4. That sum = output pixel value at (y, x)
    """
    ih, iw = image.shape        # image height and width
    kh, kw = kernel.shape       # kernel height and width
    
    # Padding: keeps output same size as input
    # Without padding: output would be (ih-kh+1, iw-kw+1) = smaller!
    # This matters because in CNNs, we stack many layers.
    # If each layer shrinks the image, we lose border information fast.
    # This matters because in CNNs, we stack many layers.
    # If each layer shrinks the image, we lose border information fast.
    # pad_h and pad_w define how many zero-pixels we add to all four sides of the image
    pad_h = kh // 2
    pad_w = kw // 2
    
    if padding_mode == 'constant':
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)),
                       mode='constant', constant_values=0)
    elif padding_mode == 'reflect':
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)),
                       mode='reflect')
    elif padding_mode == 'edge':
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)),
                       mode='edge')
    
    # Output: same size as input
    output = np.zeros_like(image)
    
    # THE CORE LOOP: slide kernel over every pixel in the original image dimensions
    for y in range(ih):          # Vertical pass over each row
        for x in range(iw):      # Horizontal pass over each column in the row
            # Extract the 2D region under the kernel (matching the kernel's height and width)
            # y and x map to the top-left edge of the sliding window on the padded image.
            # padded[y:y+kh, x:x+kw] grabs a kh by kw sub-matrix from the padded array.
            region = padded[y:y+kh, x:x+kw]
            
            # Multiply the extracted image region element-by-element with the kernel weights.
            # Then np.sum() adds up all the resulting products to get a single aggregated number.
            # This single aggregated number (dot product) becomes the value of the output pixel at (y,x).
            # This single line IS convolution: it measures how much this region "matches" the kernel.
            output[y, x] = np.sum(region * kernel)
    
    return output

# PART 1C: Different kernels = different features
# KEY INSIGHT: The kernel values determine WHAT the convolution detects.
# Each kernel is a "question" asked to every pixel.

kernels = {
    "Identity": np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ], dtype=np.float32),
    # Identity: "What is your value?" → Output = Input (no change)
    # Why include this? To prove you understand: the kernel center = 1
    # means "keep the pixel as is". Everything else = 0 means "ignore neighbors".

    "Blur (Box)": np.array([
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]
    ], dtype=np.float32),
    # Blur: "What is the AVERAGE of you and your 8 neighbors?"
    # Sum = 1.0 → preserves overall brightness
    # Every pixel becomes the mean of its 3x3 neighborhood
    # This smooths out noise but also blurs edges

    "Sobel_X (vertical edges)": np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32),
    # Sobel X: "Is the right side brighter or darker than the left side?"
    # Sum = 0 → responds ONLY to change. Uniform regions → 0
    # Positive output = bright on right (left-to-right edge)
    # Negative output = bright on left (right-to-left edge)
    # The center column is 0 → it compares LEFT vs RIGHT

    "Sobel_Y (horizontal edges)": np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32),
    # Sobel Y: "Is the bottom brighter or darker than the top?"
    # Same logic as Sobel X but rotated 90°

    "Sharpen": np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32),
    # Sharpen: "How different am I from my neighbors? Amplify that."
    # Sum = 1.0 → preserves brightness
    # Center = 5 (amplify self), neighbors = -1 (subtract neighbors)
    # = identity + edge enhancement

    "Laplacian": np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ], dtype=np.float32),
    # Laplacian: "Am I a peak or valley compared to ALL neighbors?"
    # Sum = 0 → only responds to change
    # Detects edges in ALL directions (isotropic)
}

# PART 1D: Run convolution + analyze results
print("\n" + "=" * 60)
print("CONVOLUTION RESULTS (with analysis):")
print("=" * 60)

results = {}
for name, kernel in kernels.items():
    output = convolution_2d(image, kernel)
    results[name] = output
    
    print(f"\n{'─' * 50}")
    print(f"  {name}")
    print(f"{'─' * 50}")
    print(f"  Kernel:\n{kernel}")
    print(f"  Kernel sum: {kernel.sum():.2f}", end="")
    if abs(kernel.sum()) < 0.01:
        print("  → responds only to CHANGES (edges)")
    elif abs(kernel.sum() - 1.0) < 0.01:
        print("  → preserves overall brightness")
    else:
        print(f"  → scales brightness by {kernel.sum():.1f}x")
    
    print(f"  Output range: [{output.min():.1f}, {output.max():.1f}]")
    print(f"  Output at center (3,3): {output[3,3]:.1f}")
    print(f"  Output at edge  (2,2): {output[2,2]:.1f}")
    print(f"  Output at corner (0,0): {output[0,0]:.1f}")

# PART 1E: Hand-trace one convolution step
print("\n" + "=" * 60)
print("HAND TRACE: Sobel X at position (2,2)")
print("=" * 60)
print("""
Position (2,2) = top-left corner of the bright square.

Image region around (2,2):       Sobel X kernel:
┌─────┬─────┬─────┐             ┌────┬───┬───┐
│  10 │  10 │ 255 │             │ -1 │ 0 │ 1 │
├─────┼─────┼─────┤             ├────┼───┼───┤
│  10 │ 255 │ 255 │      ×      │ -2 │ 0 │ 2 │
├─────┼─────┼─────┤             ├────┼───┼───┤
│  10 │ 255 │ 255 │             │ -1 │ 0 │ 1 │
└─────┴─────┴─────┘             └────┴───┴───┘

Multiply element by element:
  (10×-1) + (10×0) + (255×1) = -10 + 0 + 255 = 245
  (10×-2) + (255×0) + (255×2) = -20 + 0 + 510 = 490
  (10×-1) + (255×0) + (255×1) = -10 + 0 + 255 = 245

Total = 245 + 490 + 245 = 980  ← LARGE positive value
→ Strong vertical edge detected! (dark on left, bright on right)
""")

# Verify our hand calculation
sobel_x = kernels["Sobel_X (vertical edges)"]
hand_result = results["Sobel_X (vertical edges)"][2, 2]
print(f"Our code computed: {hand_result:.1f}")
print(f"Hand calculation:  980.0")
print(f"Match: {'YES ✓' if abs(hand_result - 980.0) < 1.0 else 'NO ✗'}")

print(f"\nNow check center (3,3) — INSIDE the bright square:")
center_result = results["Sobel_X (vertical edges)"][3, 3]
print(f"Output at (3,3): {center_result:.1f}")
print("→ Zero! No edge here — both sides are equally bright.")

# PART 1F: Verify implementation against OpenCV
import cv2

print("\n" + "=" * 60)
print("VERIFICATION — Manual vs OpenCV:")
print("=" * 60)

all_match = True
for name, kernel in kernels.items():
    manual = convolution_2d(image, kernel)
    opencv = cv2.filter2D(image, -1, kernel)
    match = np.allclose(manual, opencv, atol=1.0)
    status = '✓ MATCH' if match else '✗ MISMATCH'
    print(f"  {name}: {status}")
    if not match:
        all_match = False
        max_diff = np.max(np.abs(manual - opencv))
        print(f"    Max difference: {max_diff:.4f}")

if all_match:
    print("\n  All kernels match OpenCV! Your implementation is correct.")

# =============================================
# PART 1G: Visualize everything properly
# =============================================

n_kernels = len(kernels)
fig, axes = plt.subplots(2, n_kernels, figsize=(4 * n_kernels, 8))

fig.suptitle('Day 1, Part 1 — Convolution From Scratch\n'
             'Top: Absolute values (magnitude) | Bottom: Signed values (direction)',
             fontsize=14, fontweight='bold')

for idx, (name, output) in enumerate(results.items()):
    # Top row: absolute magnitude (grayscale) — what you had before
    display_abs = np.abs(output)
    if display_abs.max() > 0:
        display_abs = (display_abs / display_abs.max() * 255).astype(np.uint8)
    axes[0, idx].imshow(display_abs, cmap='gray')
    axes[0, idx].set_title(f'{name}\n|magnitude|', fontsize=9)
    axes[0, idx].axis('off')
    
    # Bottom row: signed values (diverging colormap)
    # Red = positive (bright→direction), Blue = negative (opposite direction)
    vmax = max(abs(output.min()), abs(output.max()))
    if vmax == 0:
        vmax = 1
    axes[1, idx].imshow(output, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1, idx].set_title(f'signed values\nrange: [{output.min():.0f}, {output.max():.0f}]',
                           fontsize=9)
    axes[1, idx].axis('off')

plt.tight_layout()
plt.savefig('part1_results.png', dpi=150, bbox_inches='tight')
print("\nSaved: part1_results.png")

# =============================================
# PART 1H: The critical connection to CNN
# =============================================
print("\n" + "=" * 60)
print("THE KEY INSIGHTS:")
print("=" * 60)
print("""
1. WHAT YOU BUILT:
   - A function that slides a kernel over an image
   - Different kernels detect different features
   - The kernel values determine WHAT is detected

2. WHAT A CNN DOES DIFFERENTLY:
   - Kernels are NOT hand-designed (like Sobel, Laplacian)
   - Kernels START as random numbers
   - Backpropagation ADJUSTS kernel values to minimize error
   - After training, the kernels have LEARNED what to detect

3. HOW CNN LAYERS BUILD HIERARCHY:
   Layer 1 kernels detect: edges, colors, gradients
   Layer 2 kernels detect: corners, textures, simple shapes
   Layer 3 kernels detect: parts (wheels, eyes, windows)
   Layer 4 kernels detect: objects (cars, people, buildings)
   
   Each layer takes the PREVIOUS layer's output as input.
   So layer 2 is doing convolution on edge maps → finds corners.
   Layer 3 convolves corner maps → finds shapes. And so on.

4. THE NUMBER YOU'LL REWRITE IN C:
   Your convolution_2d function = ~15 lines of Python
   In C with 1D pointers, it becomes ~30 lines
   On ARM with NEON SIMD, those 30 lines run 4-8x faster
   That's your Week 1-2 journey.

5. KERNEL SUM RULE (memorize this):
   Sum = 0 → edge detector (only responds to change)
   Sum = 1 → feature extractor (preserves brightness)
   Sum > 1 → amplifier (brightens image)
   Sum < 0 → inverter (inverts image)
""")

print("=" * 60)
print("PART 1 COMPLETE — Next: Part 2 (Multi-Channel + Real Images)")
print("=" * 60)