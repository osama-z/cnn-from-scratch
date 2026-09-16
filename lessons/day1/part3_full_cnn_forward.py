"""
DAY 1 — PART 3: Complete CNN Forward Pass From Scratch
======================================================
Goal: Build a COMPLETE tiny CNN that classifies images.
      No PyTorch. No TensorFlow. Just NumPy.
      
This is the capstone of Day 1.
After this, you'll understand EXACTLY what happens when you call model.predict()

Architecture:
  Input (1, 8, 8)
  → Conv1 (4 filters, 3×3) → ReLU → MaxPool
  → Conv2 (8 filters, 3×3) → ReLU → MaxPool
  → Flatten
  → Dense (fully connected) → ReLU
  → Dense (output) → Softmax
  → Class probabilities

This is the SAME architecture pattern as LeNet (1998) to YOLO (2024).
The only differences are: more layers, bigger images, fancier tricks.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# =============================================
# ALL BUILDING BLOCKS (from Parts 1 & 2)
# =============================================

def conv2d_multi(input_vol, weights, biases):
    """Multi-channel convolution. input: (C_in, H, W), weights: (C_out, C_in, kH, kW)"""
    c_out, c_in, kh, kw = weights.shape
    _, ih, iw = input_vol.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(input_vol, ((0,0),(pad_h,pad_h),(pad_w,pad_w)), mode='constant')
    output = np.zeros((c_out, ih, iw), dtype=np.float32)
    
    for f in range(c_out):            # Loop over every output filter map
        for y in range(ih):           # Slide vertically
            for x in range(iw):       # Slide horizontally
                val = 0.0             # Accumulate the dot product over all input channels
                for c in range(c_in): # Loop over each depth/channel map
                    # Extract the target 2D region from padded input at channel `c`
                    region = padded[c, y:y+kh, x:x+kw]
                    # Element-wise product of region and the respective 2D weight slice, summed to val
                    val += np.sum(region * weights[f, c])
                
                # Add the specific bias for filter `f` and record into output tensor
                output[f, y, x] = val + biases[f]
    return output

def relu(x):
    """
    ReLU activation: max(0, x).
    Any negative value becomes 0, silencing the neuron. 
    Positive values remain unchanged. This enables the model to learn complex non-linear limits.
    """
    # Apply np.maximum over the entire multi-dimensional array `x` element-wise
    return np.maximum(0, x)

def max_pool(x, size=2, stride=2):
    """Max pooling on 3D input (C, H, W)"""
    c, h, w = x.shape
    oh, ow = h // stride, w // stride
    out = np.zeros((c, oh, ow), dtype=np.float32)
    for ch in range(c):              # Max pooling acts independently on EACH channel!
        for y in range(oh):          # Output spatial row
            for x_pos in range(ow):  # Output spatial column
                # Map the output coordinates back to input coordinates using the stride
                start_y = y * stride
                start_x = x_pos * stride
                
                # Snip out the pool_size x pool_size region
                region = x[ch, start_y:start_y+size, start_x:start_x+size]
                
                # The MAX picks the strongest activation feature in this region
                out[ch, y, x_pos] = np.max(region)
    return out

def dense(x, weights, biases):
    """
    Fully connected (dense) layer.
    x: (N,) input vector
    weights: (N, M) weight matrix
    biases: (M,) bias vector
    output: (M,) = x @ weights + biases
    
    This is simple matrix multiplication.
    Each output neuron = weighted sum of ALL inputs + bias
    """
    # The '@' operator in NumPy performs matrix multiplication (or dot product for 1D).
    # This multiplies every feature inside `x` by their connection weights in `weights`,
    # sums the branches together per output node, then adds the `biases`.
    return x @ weights + biases

def softmax(x):
    """
    Softmax: converts raw scores to probabilities.
    Each output is between 0 and 1, and they sum to 1.
    
    Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j))
    
    The exp() amplifies differences:
      - If one class has score 5 and others have 1-2,
        softmax gives it ~95% probability
    """
    # Subtract max for numerical stability (prevents np.exp from blowing up overflow limits)
    # the probabilities remain mathematically equivalent anyway.
    e_x = np.exp(x - np.max(x))
    
    # Divide each exponentiated value by the overall sum of ALL exponentiated values
    # ensuring that all final values sum completely perfectly to 1.0
    return e_x / e_x.sum()

# =============================================
# BUILD THE CNN
# =============================================
# Initialize all weights (normally these would be LEARNED)
# We use random weights just to show the forward pass mechanics

print("=" * 60)
print("BUILDING A COMPLETE CNN FROM SCRATCH")
print("=" * 60)

class TinyCNN:
    """
    A complete CNN built with only NumPy.
    
    Architecture:
        Conv1(1→4, 3×3) → ReLU → MaxPool(2×2)
        Conv2(4→8, 3×3) → ReLU → MaxPool(2×2)
        Flatten
        Dense(32→16) → ReLU
        Dense(16→3) → Softmax
    
    This classifies 8×8 grayscale images into 3 classes.
    Tiny, but structurally identical to real CNNs.
    """
    
    def __init__(self, num_classes=3):
        # Kaiming/He initialization: scale by sqrt(2/fan_in)
        # This prevents vanishing/exploding gradients
        
        # Conv1: 1 input channel → 4 output channels, 3×3 kernels
        self.conv1_w = np.random.randn(4, 1, 3, 3).astype(np.float32) * np.sqrt(2.0 / (1*3*3))
        self.conv1_b = np.zeros(4, dtype=np.float32)
        
        # Conv2: 4 input channels → 8 output channels, 3×3 kernels
        self.conv2_w = np.random.randn(8, 4, 3, 3).astype(np.float32) * np.sqrt(2.0 / (4*3*3))
        self.conv2_b = np.zeros(8, dtype=np.float32)
        
        # After Conv1+Pool: (4, 4, 4) = 64 values
        # After Conv2+Pool: (8, 2, 2) = 32 values
        
        # Dense1: 32 inputs → 16 outputs
        self.fc1_w = np.random.randn(32, 16).astype(np.float32) * np.sqrt(2.0 / 32)
        self.fc1_b = np.zeros(16, dtype=np.float32)
        
        # Dense2: 16 inputs → num_classes outputs
        self.fc2_w = np.random.randn(16, num_classes).astype(np.float32) * np.sqrt(2.0 / 16)
        self.fc2_b = np.zeros(num_classes, dtype=np.float32)
        
        self.num_classes = num_classes
    
    def forward(self, x, verbose=True):
        """
        Complete forward pass.
        x: input image, shape (1, 8, 8) — 1 channel, 8×8 pixels
        Returns: class probabilities, shape (num_classes,)
        """
        if verbose:
            print(f"\n{'─' * 50}")
            print(f"  FORWARD PASS (tracing every shape)")
            print(f"{'─' * 50}")
            print(f"  Input:            {x.shape}   = {x.size} values")
        
        # --- LAYER 1: Conv → ReLU → Pool ---
        z1 = conv2d_multi(x, self.conv1_w, self.conv1_b)
        if verbose:
            print(f"  After Conv1:      {z1.shape}  = {z1.size} values  "
                  f"(1→4 channels, 3×3 kernels)")
        
        a1 = relu(z1)
        if verbose:
            zeros_pct = np.sum(a1 == 0) / a1.size * 100
            print(f"  After ReLU1:      {a1.shape}  = {a1.size} values  "
                  f"({zeros_pct:.0f}% zeroed out)")
        
        p1 = max_pool(a1, size=2, stride=2)
        if verbose:
            print(f"  After MaxPool1:   {p1.shape}  = {p1.size} values  "
                  f"(spatial 8→4)")
        
        # --- LAYER 2: Conv → ReLU → Pool ---
        z2 = conv2d_multi(p1, self.conv2_w, self.conv2_b)
        if verbose:
            print(f"  After Conv2:      {z2.shape}  = {z2.size} values  "
                  f"(4→8 channels, 3×3 kernels)")
        
        a2 = relu(z2)
        if verbose:
            zeros_pct = np.sum(a2 == 0) / a2.size * 100
            print(f"  After ReLU2:      {a2.shape}  = {a2.size} values  "
                  f"({zeros_pct:.0f}% zeroed out)")
        
        p2 = max_pool(a2, size=2, stride=2)
        if verbose:
            print(f"  After MaxPool2:   {p2.shape}  = {p2.size} values  "
                  f"(spatial 4→2)")
        
        # --- FLATTEN ---
        flat = p2.flatten()
        if verbose:
            print(f"  After Flatten:    {flat.shape}    = {flat.size} values  "
                  f"(3D → 1D vector)")
        
        # --- DENSE LAYERS ---
        d1 = dense(flat, self.fc1_w, self.fc1_b)
        if verbose:
            print(f"  After Dense1:     {d1.shape}     = {d1.size} values  "
                  f"(32→16)")
        
        d1 = relu(d1)
        if verbose:
            print(f"  After ReLU3:      {d1.shape}     = {d1.size} values")
        
        d2 = dense(d1, self.fc2_w, self.fc2_b)
        if verbose:
            print(f"  After Dense2:     {d2.shape}      = {d2.size} values  "
                  f"(16→{self.num_classes} raw scores)")
        
        probs = softmax(d2)
        if verbose:
            print(f"  After Softmax:    {probs.shape}      = probabilities summing to {probs.sum():.4f}")
            print(f"{'─' * 50}")
        
        return probs
    
    def count_parameters(self):
        """Count total learnable parameters."""
        params = {
            'Conv1 weights': self.conv1_w.size,
            'Conv1 biases': self.conv1_b.size,
            'Conv2 weights': self.conv2_w.size,
            'Conv2 biases': self.conv2_b.size,
            'Dense1 weights': self.fc1_w.size,
            'Dense1 biases': self.fc1_b.size,
            'Dense2 weights': self.fc2_w.size,
            'Dense2 biases': self.fc2_b.size,
        }
        return params

# =============================================
# RUN THE CNN
# =============================================

# Create 3 different "images" representing 3 classes
class_names = ["Horizontal Edge", "Vertical Edge", "Uniform"]

# Image 1: horizontal edges
img1 = np.zeros((1, 8, 8), dtype=np.float32)
img1[0, 0:4, :] = 200
img1[0, 4:8, :] = 50

# Image 2: vertical edges
img2 = np.zeros((1, 8, 8), dtype=np.float32)
img2[0, :, 0:4] = 200
img2[0, :, 4:8] = 50

# Image 3: uniform (no edges)
img3 = np.ones((1, 8, 8), dtype=np.float32) * 128

images = [img1, img2, img3]

# Build and run the CNN
model = TinyCNN(num_classes=3)

print("\n" + "=" * 60)
print("RUNNING FORWARD PASS ON 3 DIFFERENT IMAGES")
print("=" * 60)

all_probs = []
for i, (img, name) in enumerate(zip(images, class_names)):
    print(f"\n▶ Image {i+1}: {name}")
    probs = model.forward(img, verbose=(i == 0))  # Verbose only for first
    if i > 0:
        probs = model.forward(img, verbose=False)
    all_probs.append(probs)
    print(f"  Predictions: {[f'{p:.3f}' for p in probs]}")
    print(f"  Predicted class: {class_names[np.argmax(probs)]} ({probs.max():.1%})")

print("""
NOTE: The predictions are RANDOM because we haven't trained the network!
Training = adjusting weights so that:
  - Image 1 → high probability for class 0
  - Image 2 → high probability for class 1
  - Image 3 → high probability for class 2

Training uses backpropagation (gradient descent).
That's Day 2-3 territory. Today is about understanding the FORWARD pass.
""")

# =============================================
# PARAMETER ANALYSIS
# =============================================

print("=" * 60)
print("PARAMETER ANALYSIS")
print("=" * 60)

params = model.count_parameters()
total = 0
print(f"\n{'Layer':<20} {'Count':>8} {'Bytes (f32)':>12} {'Bytes (int8)':>12}")
print("─" * 56)
for name, count in params.items():
    print(f"{name:<20} {count:>8} {count*4:>12} {count:>12}")
    total += count
print("─" * 56)
print(f"{'TOTAL':<20} {total:>8} {total*4:>12} {total:>12}")
print(f"\n{'TOTAL':<20} {total:>8} {total*4/1024:>11.1f}KB {total/1024:>11.2f}KB")

print(f"""
This tiny model: {total} parameters = {total*4/1024:.1f} KB (float32)

For comparison:
  YOLO-Nano:  ~1.9M params = ~1.9 MB (float32) → ~475 KB (int8)
  YOLO-Tiny:  ~6.0M params = ~6.0 MB (float32) → ~1.5 MB (int8)

  Raspberry Pi 4:  4 GB RAM  → can run YOLO-Small
  STM32H7:        1 MB RAM  → needs YOLO-Nano + aggressive quantization
  
  Your Week 2 goal: understand quantization (float32 → int8)
  Your Week 3-4 goal: deploy this on ARM
""")

# =============================================
# COMPUTE COST ANALYSIS (FLOPs)
# =============================================

print("=" * 60)
print("COMPUTE COST ANALYSIS (FLOPs)")
print("=" * 60)

def count_conv_flops(c_in, c_out, kh, kw, oh, ow):
    """FLOPs for one conv layer. Each output pixel needs kh*kw*c_in multiplies and adds."""
    mults = c_out * oh * ow * c_in * kh * kw
    adds = mults  # roughly equal number of additions
    return mults + adds

conv1_flops = count_conv_flops(1, 4, 3, 3, 8, 8)
conv2_flops = count_conv_flops(4, 8, 3, 3, 4, 4)
dense1_flops = 32 * 16 * 2  # multiply + add for each connection
dense2_flops = 16 * 3 * 2

total_flops = conv1_flops + conv2_flops + dense1_flops + dense2_flops

print(f"\n{'Layer':<20} {'FLOPs':>10}")
print("─" * 32)
print(f"{'Conv1':<20} {conv1_flops:>10,}")
print(f"{'Conv2':<20} {conv2_flops:>10,}")
print(f"{'Dense1':<20} {dense1_flops:>10,}")
print(f"{'Dense2':<20} {dense2_flops:>10,}")
print("─" * 32)
print(f"{'TOTAL':<20} {total_flops:>10,}")

print(f"""
Our tiny model: {total_flops:,} FLOPs per image

For comparison:
  YOLO v5-Nano:  ~4 billion FLOPs per frame
  YOLO v5-Small: ~17 billion FLOPs per frame
  
  ARM Cortex-A72 (Raspberry Pi 4): ~10 GFLOPS
  → YOLO-Nano: ~0.4 seconds per frame (theoretical best)
  → With NEON SIMD optimization: ~0.1 seconds per frame
  
  This is why Week 2 (NEON/CMSIS-NN optimization) matters!
""")

# =============================================
# VISUALIZATION
# =============================================

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle('Day 1, Part 3 — Complete CNN Forward Pass\n'
             'From Raw Pixels to Class Probabilities',
             fontsize=14, fontweight='bold')

# Row 1: Input images
for i, (img, name) in enumerate(zip(images, class_names)):
    axes[0, i].imshow(img[0], cmap='gray', vmin=0, vmax=255)
    axes[0, i].set_title(f'Input: {name}', fontsize=10)
    axes[0, i].axis('off')

# Row 1, col 4: Architecture diagram (text-based)
axes[0, 3].text(0.5, 0.5, 
    'Architecture:\n\n'
    'Input (1, 8, 8)\n'
    '  ↓ Conv1 3×3\n'
    '(4, 8, 8)\n'
    '  ↓ ReLU + MaxPool\n'
    '(4, 4, 4)\n'
    '  ↓ Conv2 3×3\n'
    '(8, 4, 4)\n'
    '  ↓ ReLU + MaxPool\n'
    '(8, 2, 2)\n'
    '  ↓ Flatten\n'
    '(32,)\n'
    '  ↓ Dense+ReLU\n'
    '(16,)\n'
    '  ↓ Dense+Softmax\n'
    f'({model.num_classes},) probs',
    transform=axes[0, 3].transAxes,
    fontsize=9, fontfamily='monospace',
    verticalalignment='center', horizontalalignment='center',
    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
axes[0, 3].set_title('Network Architecture', fontsize=10)
axes[0, 3].axis('off')

# Row 2: Predictions
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
for i, (probs, name) in enumerate(zip(all_probs, class_names)):
    bars = axes[1, i].bar(class_names, probs, color=colors)
    axes[1, i].set_title(f'Predictions for: {name}', fontsize=10)
    axes[1, i].set_ylim(0, 1)
    axes[1, i].set_ylabel('Probability')
    axes[1, i].tick_params(axis='x', rotation=45, labelsize=8)
    # Highlight predicted class
    max_idx = np.argmax(probs)
    bars[max_idx].set_edgecolor('black')
    bars[max_idx].set_linewidth(2)

# Row 2, col 4: Parameter breakdown
param_names = ['Conv1\nweights', 'Conv2\nweights', 'Dense1\nweights', 'Dense2\nweights']
param_counts = [model.conv1_w.size, model.conv2_w.size, model.fc1_w.size, model.fc2_w.size]
axes[1, 3].bar(param_names, param_counts, color=['#FF9F43', '#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[1, 3].set_title('Parameters per Layer', fontsize=10)
axes[1, 3].set_ylabel('Count')
for j, v in enumerate(param_counts):
    axes[1, 3].text(j, v + 5, str(v), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('part3_results.png', dpi=150, bbox_inches='tight')
print("Saved: part3_results.png")

# =============================================
# FINAL SUMMARY: THE COMPLETE PICTURE
# =============================================

print("\n" + "=" * 60)
print("DAY 1 COMPLETE — THE COMPLETE PICTURE")
print("=" * 60)
print("""
╔══════════════════════════════════════════════════════════╗
║                    YOUR DAY 1 JOURNEY                    ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Part 1: Convolution                                     ║
║    ✓ Image = matrix of numbers                           ║
║    ✓ Kernel = small matrix that detects features         ║
║    ✓ Convolution = slide + multiply + sum                ║
║    ✓ Different kernels = different features              ║
║                                                          ║
║  Part 2: CNN Building Blocks                             ║
║    ✓ Stride = reduces spatial dimensions                 ║
║    ✓ ReLU = enables non-linear learning                  ║
║    ✓ Pooling = spatial reduction + invariance             ║
║    ✓ Multi-channel = real CNN layer operation             ║
║    ✓ Stacking = hierarchical feature detection           ║
║                                                          ║
║  Part 3: Complete Forward Pass                           ║
║    ✓ Full CNN: Conv → ReLU → Pool → Dense → Softmax     ║
║    ✓ Shape tracking through every layer                  ║
║    ✓ Parameter counting (memory budget)                  ║
║    ✓ FLOPs counting (compute budget)                     ║
║                                                          ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  WHAT'S NEXT (Day 2-3):                                 ║
║    → Loss functions (how to measure "wrongness")         ║
║    → Backpropagation (how gradients flow backward)       ║
║    → Training loop (how weights get updated)             ║
║    → Then Day 4-7: rewrite ALL of this in C!             ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝

The C version of your forward pass will look like:
  - conv2d_multi()  → nested for-loops with float pointers
  - relu()          → one loop: if (x[i] < 0) x[i] = 0;
  - max_pool()      → nested loops comparing 4 values
  - dense()         → matrix multiply with pointer arithmetic
  - softmax()       → exp() and division with fixed-point tricks

That's Week 1, Day 4-7. You now have the Python reference implementation.
Every line of C you write will correspond to a line of Python you understand.
""")

print("=" * 60)
print("Push Day 1 to GitHub. You've earned it.")
print("=" * 60)
