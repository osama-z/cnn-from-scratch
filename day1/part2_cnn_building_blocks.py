"""
DAY 1 — PART 2: From Convolution to CNN Building Blocks
========================================================
Goal: Bridge the gap between "I can convolve" and "I understand CNN layers"

Part 1 gave you: single-channel convolution with hand-designed kernels
Part 2 gives you:
    1. Stride — how CNNs reduce spatial dimensions
    2. ReLU — why non-linearity is critical
    3. Pooling — another way to reduce dimensions
    4. Multi-channel convolution — how real CNN layers work
    5. Stacking layers — the "deep" in deep learning
    6. Test on a REAL image — move beyond toy examples

After this, you'll understand every operation in a CNN forward pass.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import os

# Reuse convolution from part 1
def convolution_2d(image, kernel, padding_mode='constant'):
    """Same as Part 1 — single channel convolution."""
    ih, iw = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    if padding_mode == 'constant':
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)),
                       mode='constant', constant_values=0)
    else:
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)),
                       mode=padding_mode)
    
    output = np.zeros_like(image)
    for y in range(ih):
        for x in range(iw):
            region = padded[y:y+kh, x:x+kw]
            output[y, x] = np.sum(region * kernel)
    return output

# =============================================
# SECTION 1: STRIDE
# =============================================
# Stride = how many pixels the kernel jumps each step
# Stride 1 = check every pixel (what Part 1 did)
# Stride 2 = skip every other pixel → output is HALF the size

print("=" * 60)
print("SECTION 1: STRIDE — Reducing spatial dimensions")
print("=" * 60)

def convolution_2d_stride(image, kernel, stride=1):
    """
    Convolution with stride.
    
    Stride 1: output size = input size (with same-padding)
    Stride 2: output size ≈ input size / 2
    
    Formula: output_size = floor((input_size + 2*pad - kernel_size) / stride) + 1
    
    Why stride matters:
    - A 640×640 YOLO input would stay 640×640 through every layer without stride
    - That's 409,600 values per channel per layer — too expensive
    - Stride 2 reduces to 320×320 → 160×160 → 80×80 → ...
    - This is how CNNs create the "feature pyramid" in YOLO
    """
    ih, iw = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)),
                   mode='constant', constant_values=0)
    
    # Calculate output dimensions
    oh = (ih + 2*pad_h - kh) // stride + 1
    ow = (iw + 2*pad_w - kw) // stride + 1
    
    output = np.zeros((oh, ow), dtype=np.float32)
    
    for y in range(oh):          # For each row in the expected output
        for x in range(ow):      # For each column in the expected output
            # Calculate the starting index in the padded input image.
            # y*stride and x*stride — this is where the "skip" happens.
            # E.g., if stride=2, we check indices 0, 2, 4... skipping odd positions.
            iy = y * stride
            ix = x * stride
            
            # Extract the sub-region of size kh x kw starting from (iy, ix)
            region = padded[iy:iy+kh, ix:ix+kw]
            
            # Element-wise multiplication followed by sum, yielding one target pixel
            output[y, x] = np.sum(region * kernel)
    
    return output

# Demo stride with our image
image = np.array([
    [10,  10,  10,  10,  10,  10,  10,  10],
    [10,  10,  10,  10,  10,  10,  10,  10],
    [10,  10, 255, 255, 255, 255,  10,  10],
    [10,  10, 255, 255, 255, 255,  10,  10],
    [10,  10, 255, 255, 255, 255,  10,  10],
    [10,  10, 255, 255, 255, 255,  10,  10],
    [10,  10,  10,  10,  10,  10,  10,  10],
    [10,  10,  10,  10,  10,  10,  10,  10],
], dtype=np.float32)

sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

output_s1 = convolution_2d_stride(image, sobel_x, stride=1)
output_s2 = convolution_2d_stride(image, sobel_x, stride=2)

print(f"\nInput shape:          {image.shape}")
print(f"Output (stride=1):    {output_s1.shape}  ← same size")
print(f"Output (stride=2):    {output_s2.shape}  ← HALF size!")
print(f"\nStride 2 output:\n{output_s2}")
print("""
Notice: stride 2 still detects the edges, just at lower resolution.
The information is preserved, the spatial detail is reduced.
This is a key trade-off in CNN design: resolution vs computation.
""")

# =============================================
# SECTION 2: ReLU — The Non-Linearity
# =============================================
# This is the MOST IMPORTANT conceptual piece after convolution

print("=" * 60)
print("SECTION 2: ReLU — Why non-linearity is CRITICAL")
print("=" * 60)

def relu(x):
    """
    ReLU = Rectified Linear Unit
    f(x) = max(0, x)
    
    That's it. That's the function. Dead simple.
    But it's the reason deep learning works.
    """
    # np.maximum operates on the whole array, substituting any negative number with 0.
    # Positive numbers are left as they were. This adds non-linearity!
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.1):
    """
    Leaky ReLU: f(x) = x if x > 0, else alpha * x
    
    YOLO uses this (alpha=0.1).
    Why? Regular ReLU kills negative values completely (→ 0).
    Leaky ReLU keeps a small gradient for negatives.
    This prevents "dead neurons" — neurons that stop learning.
    """
    return np.where(x > 0, x, alpha * x)

# Show why ReLU matters
print("""
WHY DO WE NEED ReLU?

Without ReLU, stacking convolution layers is POINTLESS:
  Conv1(x) = K1 * x + b1      (linear operation)
  Conv2(Conv1(x)) = K2 * (K1 * x + b1) + b2
                  = (K2*K1) * x + (K2*b1 + b2)
                  = K_combined * x + b_combined

  → Two linear layers = ONE linear layer mathematically!
  → 100 linear layers = ONE linear layer!
  → You can't learn complex patterns with linear functions!

WITH ReLU between layers:
  Conv2(ReLU(Conv1(x))) ≠ any single linear operation
  → The network can now learn curves, not just lines
  → Deeper = more complex patterns
""")

# Demonstrate on Sobel output
sobel_output = convolution_2d(image, sobel_x)
relu_output = relu(sobel_output)
leaky_output = leaky_relu(sobel_output)

print(f"Sobel X output range:    [{sobel_output.min():.0f}, {sobel_output.max():.0f}]")
print(f"After ReLU:              [{relu_output.min():.0f}, {relu_output.max():.0f}]")
print(f"After Leaky ReLU (0.1):  [{leaky_output.min():.0f}, {leaky_output.max():.0f}]")
print(f"\nReLU zeroed out {np.sum(relu_output == 0)} of {relu_output.size} values")
print("→ All negative responses (right-to-left edges) are gone.")
print("→ Only left-to-right edges remain. That's FEATURE SELECTION.")

# =============================================
# SECTION 3: POOLING — Another dimension reducer
# =============================================

print("\n" + "=" * 60)
print("SECTION 3: POOLING — Spatial reduction + invariance")
print("=" * 60)

def max_pool_2d(feature_map, pool_size=2, stride=2):
    """
    Max Pooling: take the MAXIMUM value in each pool_size × pool_size region.
    
    Why max? Because we want the STRONGEST activation.
    If an edge was detected anywhere in the 2×2 region, we keep it.
    This makes the network slightly invariant to small shifts.
    
    Parameters:
        feature_map: 2D array (output of conv + relu)
        pool_size: size of pooling window
        stride: typically = pool_size (non-overlapping)
    """
    h, w = feature_map.shape
    oh = h // stride
    ow = w // stride
    output = np.zeros((oh, ow), dtype=np.float32)
    
    for y in range(oh):          # Vertical pass over output map
        for x in range(ow):      # Horizontal pass over output map
            # Calculate input coordinates factoring in stride (normally pool_size).
            # If pool_size=2 and stride=2, windows are [0:2], [2:4], [4:6] so they never overlap.
            region = feature_map[y*stride:y*stride+pool_size,
                                 x*stride:x*stride+pool_size]
            # Take the MAXIMUM value out of these pool_size x pool_size pixels
            # retaining only the strongest feature response inside that small region!
            output[y, x] = np.max(region)
    
    return output

def avg_pool_2d(feature_map, pool_size=2, stride=2):
    """Average pooling — takes the mean instead of max."""
    h, w = feature_map.shape
    oh = h // stride
    ow = w // stride
    output = np.zeros((oh, ow), dtype=np.float32)
    
    for y in range(oh):
        for x in range(ow):
            region = feature_map[y*stride:y*stride+pool_size,
                                x*stride:x*stride+pool_size]
            output[y, x] = np.mean(region)
    
    return output

# Demo pooling
relu_out = relu(convolution_2d(image, sobel_x))
maxpool_out = max_pool_2d(relu_out, pool_size=2, stride=2)
avgpool_out = avg_pool_2d(relu_out, pool_size=2, stride=2)

print(f"\nConv+ReLU output shape:   {relu_out.shape}")
print(f"After MaxPool 2×2:        {maxpool_out.shape}  ← 4× fewer values")
print(f"After AvgPool 2×2:        {avgpool_out.shape}")
print(f"\nMaxPool result:\n{maxpool_out}")
print(f"\nAvgPool result:\n{avgpool_out}")
print("""
MaxPool vs AvgPool:
  MaxPool: keeps the STRONGEST signal → better for detection
  AvgPool: keeps the AVERAGE signal → better for classification at the end
  
  YOLO uses stride-2 convolutions instead of pooling (more modern approach).
  But understanding pooling helps you understand older architectures.
""")

# =============================================ㅊ
# SECTION 4: MULTI-CHANNEL CONVOLUTION
# =============================================
# This is where most tutorials FAIL to explain clearly.
# Your Part 1 did single-channel. Real images have 3 channels (RGB).
# Real CNN layers have MANY output channels (64, 128, 256...).

print("=" * 60)
print("SECTION 4: MULTI-CHANNEL CONVOLUTION (the real deal)")
print("=" * 60)

def conv2d_multichannel(input_volume, kernels, biases=None):
    """
    Multi-channel convolution — what a REAL CNN layer does.
    
    Parameters:
        input_volume: shape (C_in, H, W)  — C_in input channels
        kernels: shape (C_out, C_in, kH, kW) — C_out filters, each spanning all C_in channels
        biases: shape (C_out,) — one bias per output channel (optional)
    
    Returns:
        output: shape (C_out, H, W) — C_out output channels
    
    How it works:
        For each output channel (each filter):
            For each spatial position (y, x):
                Sum over ALL input channels:
                    extract region, multiply by kernel, sum
                Add bias
                Store result
    
    Visual:
        Input: [R channel]    Filter 1: [R kernel]    =    [Output channel 1]
               [G channel]  ×           [G kernel]  → sum → 
               [B channel]              [B kernel]
               
        Each filter produces ONE output channel.
        64 filters → 64 output channels.
    """
    c_out, c_in, kh, kw = kernels.shape
    _, ih, iw = input_volume.shape
    
    pad_h, pad_w = kh // 2, kw // 2
    
    # Pad each input channel
    padded = np.pad(input_volume, 
                    ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                    mode='constant', constant_values=0)
    
    output = np.zeros((c_out, ih, iw), dtype=np.float32)
    
    for f in range(c_out):           # Iterating over each filter to compute one output channel each
        for y in range(ih):          # Sliding the window vertically over the image height
            for x in range(iw):      # Sliding the window horizontally over the image width
                
                # Initialize the accumulator for this specific spatial position.
                # A single output pixel is the sum across ALL input channels.
                total = 0.0
                
                for c in range(c_in):   # For each input channel (e.g. Red channel, Green channel, etc)
                    # Extract the sub-region on current channel `c` 
                    region = padded[c, y:y+kh, x:x+kw]
                    
                    # Multiply the region by the f-th filter's weights for the c-th channel, then sum.
                    # Since kernels has shape (c_out, c_in, kh, kw), kernels[f, c] gives the 2D kernel.
                    total += np.sum(region * kernels[f, c])
                
                # After summing over all channels, add the bias specific to this filter `f`.
                if biases is not None:
                    total += biases[f]
                
                # Finally, store accumulated overall sum in the output tensor.
                output[f, y, x] = total
    
    return output

# Create a fake 3-channel image (like RGB)
print("\nCreating a 3-channel (RGB-like) input image...")
rgb_image = np.stack([
    image * 0.8,    # R channel (slightly dimmer)
    image * 1.0,    # G channel (full brightness)  
    image * 0.6,    # B channel (dimmer)
], axis=0)  # shape: (3, 8, 8)

print(f"Input shape: {rgb_image.shape}  → (channels, height, width)")

# Create 4 filters (output channels), each spanning all 3 input channels
# Shape: (4, 3, 3, 3) = (num_filters, input_channels, kernel_h, kernel_w)
np.random.seed(42)
filters = np.random.randn(4, 3, 3, 3).astype(np.float32) * 0.5
biases = np.zeros(4, dtype=np.float32)

print(f"Filters shape: {filters.shape}  → (num_filters, input_channels, kernel_h, kernel_w)")
print(f"Biases shape: {biases.shape}  → one bias per output channel")

output_multi = conv2d_multichannel(rgb_image, filters, biases)
output_multi = relu(output_multi)  # Apply ReLU

print(f"Output shape: {output_multi.shape}  → (output_channels, height, width)")
print(f"\nSo: 3 input channels → 4 output channels")
print(f"Each output channel = one learned feature detector")
print("""
DIMENSION STORY OF A CNN:
  Input image:      (3, 640, 640)     = 3 channels (RGB)
  After Conv1:      (64, 640, 640)    = 64 feature maps
  After Conv2:      (128, 320, 320)   = stride 2 halves spatial dims
  After Conv3:      (256, 160, 160)   = more features, less spatial
  After Conv4:      (512, 80, 80)     = getting abstract
  ...
  Final:            (1000,)           = class probabilities (for classification)
  
  OR for YOLO:
  Final:            (num_boxes, 5+num_classes) per scale
                    = bounding box predictions
""")

# =============================================
# SECTION 5: STACKING LAYERS — The "Deep" in Deep Learning
# =============================================

print("=" * 60)
print("SECTION 5: STACKING LAYERS — Building a mini 2-layer network")
print("=" * 60)

# Layer 1: 3 input channels → 4 output channels
np.random.seed(42)
layer1_filters = np.random.randn(4, 3, 3, 3).astype(np.float32) * 0.3
layer1_biases = np.zeros(4, dtype=np.float32)

# Layer 2: 4 input channels → 2 output channels
layer2_filters = np.random.randn(2, 4, 3, 3).astype(np.float32) * 0.3
layer2_biases = np.zeros(2, dtype=np.float32)

print("Mini 2-layer CNN forward pass:")
print(f"  Input:           {rgb_image.shape}")

# Forward pass
out1 = conv2d_multichannel(rgb_image, layer1_filters, layer1_biases)
print(f"  After Conv1:     {out1.shape}    (3→4 channels)")
out1 = relu(out1)
print(f"  After ReLU1:     {out1.shape}    (negatives → 0)")

# Max pool each channel independently
pooled = np.zeros((out1.shape[0], out1.shape[1]//2, out1.shape[2]//2), dtype=np.float32)
for c in range(out1.shape[0]):
    pooled[c] = max_pool_2d(out1[c], pool_size=2, stride=2)
print(f"  After MaxPool:   {pooled.shape}  (spatial ÷ 2)")

out2 = conv2d_multichannel(pooled, layer2_filters, layer2_biases)
print(f"  After Conv2:     {out2.shape}    (4→2 channels)")
out2 = relu(out2)
print(f"  After ReLU2:     {out2.shape}    (negatives → 0)")

# Flatten for classification
flat = out2.flatten()
print(f"  After Flatten:   {flat.shape}   (ready for dense layer)")

print("""
This is EXACTLY what happens inside YOLO, just scaled up:
  - YOLO v3: 75 convolutional layers
  - YOLO v5: ~280 layers total
  - Each layer: Conv → BatchNorm → LeakyReLU
  
Your forward pass above = the same logic, just 2 layers instead of 75.
""")

# =============================================
# SECTION 6: TEST ON A REAL IMAGE
# =============================================

print("=" * 60)
print("SECTION 6: REAL IMAGE TEST")
print("=" * 60)

# Create a more interesting test image (simulate a real scene)
# Since we might not have a real image file, let's create a synthetic one
# that's more complex than the 7x7 toy

print("Creating a synthetic 64×64 test image with multiple features...")

test_image = np.zeros((64, 64), dtype=np.float32)

# Background gradient (like sky)
for y in range(64):
    test_image[y, :] = 40 + y * 1.5  # gradient from dark to light

# Add a "building" (rectangle)
test_image[20:55, 10:25] = 200

# Add a "window" inside the building
test_image[25:32, 13:20] = 50

# Add another "building"
test_image[30:60, 35:55] = 180

# Add windows
test_image[35:40, 38:43] = 40
test_image[35:40, 47:52] = 40
test_image[45:50, 38:43] = 40
test_image[45:50, 47:52] = 40

# Add a "road" at bottom
test_image[58:64, :] = 80

# Add a bright "light" (like headlight)
for y in range(58, 64):
    for x in range(28, 34):
        dist = np.sqrt((y - 61)**2 + (x - 31)**2)
        test_image[y, x] = min(255, 80 + 200 * max(0, 1 - dist/4))

# Define all kernels
kernels = {
    "Sobel_X": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
    "Sobel_Y": np.array([[-1,-2,-1], [0,0,0], [1,2,1]], dtype=np.float32),
    "Laplacian": np.array([[0,1,0], [1,-4,1], [0,1,0]], dtype=np.float32),
    "Sharpen": np.array([[0,-1,0], [-1,5,-1], [0,-1,0]], dtype=np.float32),
    "Blur": np.ones((3,3), dtype=np.float32) / 9,
    "Emboss": np.array([[-2,-1,0], [-1,1,1], [0,1,2]], dtype=np.float32),
}

# Apply all kernels
fig, axes = plt.subplots(3, 3, figsize=(14, 14))
fig.suptitle('Day 1, Part 2 — Convolution on Complex Image\n'
             'Each kernel reveals different features', fontsize=14, fontweight='bold')

# Original
axes[0, 0].imshow(test_image, cmap='gray')
axes[0, 0].set_title('Original\n(synthetic scene)', fontsize=10)
axes[0, 0].axis('off')

# ReLU diagram
x_vals = np.linspace(-5, 5, 100)
axes[0, 1].plot(x_vals, np.maximum(0, x_vals), 'b-', linewidth=2, label='ReLU')
axes[0, 1].plot(x_vals, np.where(x_vals > 0, x_vals, 0.1*x_vals), 'r--', linewidth=2, label='Leaky ReLU')
axes[0, 1].set_title('Activation Functions', fontsize=10)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=0, color='k', linewidth=0.5)
axes[0, 1].axvline(x=0, color='k', linewidth=0.5)

# Edge magnitude (combined Sobel)
edge_x = convolution_2d(test_image, kernels["Sobel_X"])
edge_y = convolution_2d(test_image, kernels["Sobel_Y"])
edge_magnitude = np.sqrt(edge_x**2 + edge_y**2)
axes[0, 2].imshow(edge_magnitude, cmap='hot')
axes[0, 2].set_title('Edge Magnitude\n√(Sobel_X² + Sobel_Y²)', fontsize=10)
axes[0, 2].axis('off')

# Individual kernels
positions = [(1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
for idx, (name, kernel) in enumerate(kernels.items()):
    row, col = positions[idx]
    output = convolution_2d(test_image, kernel)
    output_relu = relu(output)
    
    axes[row, col].imshow(output_relu, cmap='gray')
    axes[row, col].set_title(f'{name} + ReLU\nsum={kernel.sum():.1f}', fontsize=10)
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('part2_results.png', dpi=150, bbox_inches='tight')
print("Saved: part2_results.png")

# =============================================
# SECTION 7: Parameter Count — Why This Matters for Embedded
# =============================================

print("\n" + "=" * 60)
print("SECTION 7: PARAMETER COUNT — The Embedded Constraint")
print("=" * 60)
print("""
Why count parameters? Because on ARM/embedded:
  - RAM is limited (256KB - 2MB typically)
  - Each parameter = 4 bytes (float32)
  - Or 1 byte with quantization (Week 2 topic!)

Let's count parameters for our mini network:
""")

# Count parameters
l1_params = layer1_filters.size + layer1_biases.size
l2_params = layer2_filters.size + layer2_biases.size
total_params = l1_params + l2_params

print(f"Layer 1: {layer1_filters.shape} filters + {layer1_biases.shape} biases")
print(f"         = {layer1_filters.size} + {layer1_biases.size} = {l1_params} parameters")
print(f"Layer 2: {layer2_filters.shape} filters + {layer2_biases.shape} biases")
print(f"         = {layer2_filters.size} + {layer2_biases.size} = {l2_params} parameters")
print(f"\nTotal: {total_params} parameters")
print(f"Memory (float32): {total_params * 4} bytes = {total_params * 4 / 1024:.1f} KB")
print(f"Memory (int8 quantized): {total_params} bytes = {total_params / 1024:.2f} KB")
print(f"""
For comparison:
  YOLO-Tiny:     ~6 million parameters = ~6 MB (float32) or ~1.5 MB (int8)
  YOLO v5s:      ~7 million parameters = ~7 MB (float32) or ~1.8 MB (int8)
  YOLO v5x:      ~87 million parameters = too big for embedded!

This is why quantization (Week 2) is CRITICAL for embedded deployment.
float32 → int8 = 4× memory reduction with ~1% accuracy loss.
""")

# =============================================
# SUMMARY
# =============================================

print("=" * 60)
print("DAY 1 PART 2 COMPLETE")
print("=" * 60)
print("""
What you now understand:
  ✓ Convolution — the fundamental operation
  ✓ Stride — reduces spatial dimensions
  ✓ ReLU — enables learning complex patterns (non-linearity)
  ✓ Pooling — another way to reduce dimensions
  ✓ Multi-channel conv — how real CNN layers work
  ✓ Stacking layers — the "deep" in deep learning
  ✓ Parameter counting — the embedded constraint

What comes next (Part 3):
  → Build a FULL forward pass: Input → Conv layers → Output
  → Understand the complete data flow of a CNN
  → See how this connects to YOLO architecture
""")

print("=" * 60)
print("PART 2 COMPLETE — Next: Part 3 (Full CNN Forward Pass)")
print("=" * 60)
