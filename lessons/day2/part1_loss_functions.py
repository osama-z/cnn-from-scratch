"""
DAY 2 — PART 1: Loss Functions — Measuring How Wrong You Are
=============================================================
Goal: Understand WHY we need loss functions and HOW they guide learning.

The Story So Far (Day 1):
    You built a CNN that takes an image and outputs probabilities like [0.3, 0.5, 0.2].
    But those probabilities are RANDOM — because the weights are random.
    
    To make the network LEARN, we need TWO things:
        1. A way to measure HOW WRONG the predictions are  ← THIS FILE
        2. A way to ADJUST the weights to be less wrong     ← Part 2 & 3

What is a Loss Function?
    Your CNN predicts: [0.3, 0.5, 0.2]   (30% class A, 50% class B, 20% class C)
    The correct answer:  class B           (should be 100% class B)
    
    A loss function takes these two things and outputs ONE NUMBER:
        loss = how_wrong(prediction, truth)
    
    High loss = very wrong = BAD
    Low loss  = almost right = GOOD
    Zero loss = perfect = IMPOSSIBLE in practice
    
    Training = adjusting weights until loss is as small as possible.
    
    The loss function is your COMPASS. Without it, the network is blind.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# SECTION 1: ONE-HOT ENCODING
# Before we measure "wrongness", we need to represent the correct answer
# in a format the math can work with.

print("=" * 60)
print("SECTION 1: ONE-HOT ENCODING — Turning Labels into Numbers")
print("=" * 60)

def one_hot(label, num_classes):
    """
    Convert a class label (integer) to a one-hot vector.
    
    Example: label=2, num_classes=5 → [0, 0, 1, 0, 0]
    
    Why do we need this?
        The network outputs a VECTOR of probabilities: [0.1, 0.2, 0.7]
        The correct answer is a CLASS INDEX: 2
        We can't subtract "2" from "[0.1, 0.2, 0.7]" — different shapes!
        
        One-hot converts the label to the SAME shape:
            label 2 → [0.0, 0.0, 1.0]
        
        Now we can compare:
            predicted: [0.1, 0.2, 0.7]
            target:    [0.0, 0.0, 1.0]
            error:     [0.1, 0.2, 0.3]  ← element-wise difference
    """
    # Start with all zeros — no class is "active"
    vec = np.zeros(num_classes, dtype=np.float32)
    
    # Set the correct class to 1.0 — "this is the answer"
    vec[label] = 1.0
    
    return vec

# Demo one-hot encoding
print("\nOne-hot encoding examples:")
class_names = ["Horizontal Edge", "Vertical Edge", "Uniform"]
for i, name in enumerate(class_names):
    encoded = one_hot(i, num_classes=3)
    print(f"  Class {i} ({name:>16s}) → {encoded}")

print("""
Why "one-hot"?
  Only ONE element is HOT (=1), the rest are COLD (=0).
  It's a binary representation of categorical data.
  
  This is universal in deep learning:
    - 10 classes (MNIST digits) → vectors of length 10
    - 1000 classes (ImageNet) → vectors of length 1000
    - YOLO: each detection has a one-hot class vector
""")


# SECTION 2: MEAN SQUARED ERROR (MSE)
# The simplest loss function. Used in regression.
# We start here because the math is easy to understand.

print("=" * 60)
print("SECTION 2: MEAN SQUARED ERROR (MSE) — The Simplest Loss")
print("=" * 60)

def mse_loss(predicted, target):
    """
    Mean Squared Error: MSE = (1/n) * sum( (predicted - target)^2 )
    
    Steps:
        1. Subtract: how far off is each value?
        2. Square: make all errors positive, punish big errors MORE
        3. Mean: average over all values to get one number
    
    Why square instead of absolute value?
        |error| = same penalty for small and big errors
        error^2 = BIG errors get MUCH bigger penalty
        
        Example:
            error = 0.1 → squared = 0.01  (small)
            error = 0.5 → squared = 0.25  (25x bigger!)
            error = 1.0 → squared = 1.00  (100x bigger!)
        
        Squaring forces the network to fix its BIGGEST mistakes first.
        This is usually what we want.
    """
    n = len(predicted)
    
    # Step 1: How far off is each prediction?
    errors = predicted - target
    
    # Step 2: Square each error (makes all positive, amplifies big errors)
    squared_errors = errors ** 2
    
    # Step 3: Average over all values
    mean_error = np.sum(squared_errors) / n
    
    return mean_error

def mse_gradient(predicted, target):
    """
    Gradient of MSE with respect to predicted values.
    
    d(MSE)/d(predicted) = (2/n) * (predicted - target)
    
    This gradient tells us:
        - DIRECTION: which way to adjust each prediction
            positive gradient → prediction too high → decrease it
            negative gradient → prediction too low → increase it
        - MAGNITUDE: how much to adjust
            big gradient → big error → big adjustment needed
            small gradient → small error → small adjustment
    
    THIS IS THE KEY INSIGHT:
        The gradient is the INSTRUCTION MANUAL for improvement.
        It tells every weight: "move THIS direction by THIS amount".
    """
    n = len(predicted)
    
    # The derivative of (predicted - target)^2 is 2 * (predicted - target)
    # We divide by n because we took the mean in the loss
    return (2.0 / n) * (predicted - target)


# Hand-trace MSE with actual numbers
print("""
HAND TRACE: MSE Loss
─────────────────────────────────────────────────

Prediction: [0.3, 0.5, 0.2]   (CNN output after softmax)
Target:     [0.0, 1.0, 0.0]   (correct answer = class 1)

Step 1 — Errors:
  predicted - target = [0.3-0.0, 0.5-1.0, 0.2-0.0]
                     = [0.3, -0.5, 0.2]

Step 2 — Squared errors:
  [0.3^2, (-0.5)^2, 0.2^2] = [0.09, 0.25, 0.04]

Step 3 — Mean:
  (0.09 + 0.25 + 0.04) / 3 = 0.38 / 3 = 0.1267

MSE Loss = 0.1267
""")

# Verify with code
predicted = np.array([0.3, 0.5, 0.2], dtype=np.float32)
target = one_hot(1, 3)  # class 1

loss = mse_loss(predicted, target)
grad = mse_gradient(predicted, target)

print(f"Code verification:")
print(f"  MSE Loss = {loss:.4f}")
print(f"  Gradient = {grad}")
print(f"  Gradient meaning:")
print(f"    Class 0: grad = {grad[0]:+.4f} → prediction too HIGH, decrease it")
print(f"    Class 1: grad = {grad[1]:+.4f} → prediction too LOW,  increase it")
print(f"    Class 2: grad = {grad[2]:+.4f} → prediction too HIGH, decrease it")


# =============================================
# SECTION 3: CROSS-ENTROPY LOSS
# =============================================
# The loss function actually used for classification.
# More powerful than MSE for probabilities.

print("\n" + "=" * 60)
print("SECTION 3: CROSS-ENTROPY LOSS — The Real Deal")
print("=" * 60)

def cross_entropy_loss(predicted, target):
    """
    Cross-Entropy: CE = -sum( target * log(predicted) )
    
    Since target is one-hot (only one element is 1), this simplifies to:
        CE = -log(predicted[correct_class])
    
    The magic of logarithm:
        If predicted = 0.99 for correct class: -log(0.99) = 0.01  (tiny loss!)
        If predicted = 0.50 for correct class: -log(0.50) = 0.69  (medium loss)
        If predicted = 0.01 for correct class: -log(0.01) = 4.61  (HUGE loss!)
    
    The log function has this beautiful property:
        - Near 1.0: very gentle slope (you're close, small correction needed)
        - Near 0.0: EXTREMELY steep (you're very wrong, MASSIVE correction!)
    
    This is exactly what we want for classification:
        SCREAM when the correct class has low probability
        Whisper when it has high probability
    """
    # Clip predictions to avoid log(0) which is -infinity
    # This is a standard numerical stability trick
    epsilon = 1e-7
    predicted_clipped = np.clip(predicted, epsilon, 1.0 - epsilon)
    
    # CE = -sum( target_i * log(predicted_i) )
    # Since target is one-hot, only the correct class term survives
    loss = -np.sum(target * np.log(predicted_clipped))
    
    return loss

def cross_entropy_gradient(predicted, target):
    """
    Gradient of Cross-Entropy with respect to predictions.
    
    d(CE)/d(predicted) = -target / predicted
    
    For the correct class (target = 1):
        gradient = -1 / predicted
        If predicted = 0.01: gradient = -100  ← HUGE! "FIX THIS NOW!"
        If predicted = 0.99: gradient = -1.01 ← small. "You're fine."
    
    For wrong classes (target = 0):
        gradient = 0  ← CE doesn't care about wrong classes directly
        (but softmax couples them — adjusting one changes others)
    """
    epsilon = 1e-7
    predicted_clipped = np.clip(predicted, epsilon, 1.0 - epsilon)
    
    return -target / predicted_clipped


# Hand-trace cross-entropy
print("""
HAND TRACE: Cross-Entropy Loss
─────────────────────────────────────────────────

Prediction: [0.3, 0.5, 0.2]   (CNN output after softmax)
Target:     [0.0, 1.0, 0.0]   (correct answer = class 1)

CE = -sum( target * log(predicted) )
   = -(0.0 * log(0.3) + 1.0 * log(0.5) + 0.0 * log(0.2))
   = -(0 + 1.0 * (-0.6931) + 0)
   = -(-0.6931)
   = 0.6931

Cross-Entropy Loss = 0.6931

Intuition: the correct class (class 1) got probability 0.5
  -log(0.5) = 0.693 → "You're giving the correct answer only 50%
                         confidence. That's not great."
""")

# Verify
ce_loss = cross_entropy_loss(predicted, target)
ce_grad = cross_entropy_gradient(predicted, target)

print(f"Code verification:")
print(f"  CE Loss = {ce_loss:.4f}")
print(f"  Gradient = [{ce_grad[0]:.4f}, {ce_grad[1]:.4f}, {ce_grad[2]:.4f}]")
print(f"\n  Gradient for correct class (1): {ce_grad[1]:.4f}")
print(f"  → Negative = increase this prediction!")
print(f"  → Magnitude = {abs(ce_grad[1]):.4f} = how urgently")


# =============================================
# SECTION 4: WHY CROSS-ENTROPY BEATS MSE
# =============================================

print("\n" + "=" * 60)
print("SECTION 4: WHY CROSS-ENTROPY > MSE FOR CLASSIFICATION")
print("=" * 60)

# Compare both losses for different prediction confidences
print("\nScenario: Correct answer is class 0. How does loss change with confidence?\n")
print(f"{'Predicted prob':>15s} {'MSE Loss':>10s} {'CE Loss':>10s} {'MSE Grad':>10s} {'CE Grad':>10s}")
print("-" * 60)

test_probs = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
target_0 = one_hot(0, 2)  # [1, 0]

mse_losses = []
ce_losses = []
mse_grads_list = []
ce_grads_list = []

for p in test_probs:
    pred = np.array([p, 1 - p], dtype=np.float32)
    
    ml = mse_loss(pred, target_0)
    cl = cross_entropy_loss(pred, target_0)
    mg = mse_gradient(pred, target_0)[0]  # gradient for correct class
    cg = cross_entropy_gradient(pred, target_0)[0]
    
    mse_losses.append(ml)
    ce_losses.append(cl)
    mse_grads_list.append(mg)
    ce_grads_list.append(cg)
    
    print(f"{p:>15.2f} {ml:>10.4f} {cl:>10.4f} {mg:>+10.4f} {cg:>+10.4f}")

print("""
KEY INSIGHT — Look at the gradients when prediction is VERY WRONG (p=0.01):
  MSE gradient:  -0.66  → "hey, maybe adjust a bit?"
  CE gradient:  -100.0  → "EMERGENCY! FIX THIS IMMEDIATELY!"

And when prediction is almost right (p=0.99):
  MSE gradient:  -0.007 → "tiny nudge"
  CE gradient:  -1.01   → "tiny nudge"

Cross-Entropy is like a STRICT TEACHER:
  - When you're very wrong: SCREAMS at you (huge gradient = fast learning)
  - When you're almost right: whispers (small gradient = fine-tuning)

MSE is like a LAZY TEACHER:
  - When you're very wrong: speaks normally (medium gradient = slow learning)
  - When you're almost right: whispers (same as CE)

Result: CE makes networks learn FASTER from their mistakes.
This is why EVERY classification network uses Cross-Entropy, not MSE.
""")


# =============================================
# SECTION 5: SOFTMAX + CROSS-ENTROPY COMBINED
# =============================================
# This is the form actually used in practice.
# The combined gradient is BEAUTIFUL.

print("=" * 60)
print("SECTION 5: SOFTMAX + CROSS-ENTROPY COMBINED")
print("=" * 60)

def softmax(logits):
    """
    Softmax: converts raw scores (logits) to probabilities.
    softmax(x_i) = exp(x_i) / sum( exp(x_j) )
    
    Subtract max for numerical stability — doesn't change the result
    but prevents exp() from overflowing to infinity.
    """
    e_x = np.exp(logits - np.max(logits))
    return e_x / e_x.sum()

def softmax_cross_entropy_loss(logits, target):
    """
    Combined Softmax + Cross-Entropy.
    
    Why combine?
        1. Numerical stability: separate softmax can produce 0.0 or 1.0,
           then log(0) = -inf → crash!
        2. The combined gradient is INCREDIBLY SIMPLE (see below)
    """
    probs = softmax(logits)
    epsilon = 1e-7
    loss = -np.sum(target * np.log(probs + epsilon))
    return loss, probs

def softmax_cross_entropy_gradient(probs, target):
    """
    THE BEAUTIFUL GRADIENT:
    
        d(Loss)/d(logits) = probs - target
    
    That's it. That's the entire gradient. Let that sink in.
    
    Example:
        probs  = [0.3, 0.5, 0.2]  (what you predicted)
        target = [0.0, 1.0, 0.0]  (what's correct)
        
        gradient = [0.3-0.0, 0.5-1.0, 0.2-0.0]
                 = [0.3, -0.5, 0.2]
    
    Reading the gradient:
        Class 0: +0.3  → "You said 0.3 but should be 0.0. Decrease by 0.3"
        Class 1: -0.5  → "You said 0.5 but should be 1.0. Increase by 0.5"
        Class 2: +0.2  → "You said 0.2 but should be 0.0. Decrease by 0.2"
    
    The gradient IS the error signal. 
    predicted - target = how wrong you are, with direction.
    
    This beautiful simplicity is NOT a coincidence:
        Cross-entropy was DESIGNED to pair with softmax.
        The math cancels out elegantly.
        
    In PyTorch: nn.CrossEntropyLoss does exactly this.
    It takes raw logits (before softmax), not probabilities.
    """
    return probs - target

# Demo
logits = np.array([2.0, 1.0, 0.5], dtype=np.float32)
target_demo = one_hot(1, 3)

loss_demo, probs_demo = softmax_cross_entropy_loss(logits, target_demo)
grad_demo = softmax_cross_entropy_gradient(probs_demo, target_demo)

print(f"""
HAND TRACE: Softmax + Cross-Entropy
─────────────────────────────────────────────────

Raw logits (before softmax): [2.0, 1.0, 0.5]
  These are raw scores from the last dense layer.
  They can be any value: positive, negative, huge, tiny.

Step 1 — Softmax:
  exp([2.0, 1.0, 0.5]) = [{np.exp(2.0):.4f}, {np.exp(1.0):.4f}, {np.exp(0.5):.4f}]
  sum = {np.exp(2.0) + np.exp(1.0) + np.exp(0.5):.4f}
  probs = [{probs_demo[0]:.4f}, {probs_demo[1]:.4f}, {probs_demo[2]:.4f}]
  (sum to {probs_demo.sum():.4f})

Step 2 — Cross-Entropy:
  target = [0, 1, 0]  (class 1 is correct)
  CE = -log(probs[1]) = -log({probs_demo[1]:.4f}) = {loss_demo:.4f}

Step 3 — THE BEAUTIFUL GRADIENT:
  gradient = probs - target
           = [{probs_demo[0]:.4f}-0, {probs_demo[1]:.4f}-1, {probs_demo[2]:.4f}-0]
           = [{grad_demo[0]:+.4f}, {grad_demo[1]:+.4f}, {grad_demo[2]:+.4f}]
  
  Reading: "Decrease class 0 by {grad_demo[0]:.4f}"
           "Increase class 1 by {abs(grad_demo[1]):.4f}"
           "Decrease class 2 by {grad_demo[2]:.4f}"
""")


# =============================================
# SECTION 6: VERIFY WITH NUMERICAL GRADIENT
# =============================================
# This is a CRITICAL skill: always verify your math!

print("=" * 60)
print("SECTION 6: NUMERICAL GRADIENT CHECK — Trust but Verify")
print("=" * 60)

def numerical_gradient(loss_fn, x, target, h=1e-5):
    """
    Compute gradient numerically using finite differences.
    
    For each element x[i]:
        gradient[i] = (loss(x + h) - loss(x - h)) / (2h)
    
    This is SLOW but guaranteed correct (it's the definition of derivative).
    We use it to VERIFY our analytical gradients.
    
    If analytical is close to numerical → our math is correct!
    If they differ → we have a bug!
    """
    grad = np.zeros_like(x)
    
    for i in range(len(x)):
        # Nudge x[i] up by h
        x_plus = x.copy()
        x_plus[i] += h
        loss_plus = loss_fn(x_plus, target)
        
        # Nudge x[i] down by h
        x_minus = x.copy()
        x_minus[i] -= h
        loss_minus = loss_fn(x_minus, target)
        
        # Slope = rise / run
        grad[i] = (loss_plus - loss_minus) / (2 * h)
    
    return grad

# Helper that returns just the loss (not probs) for numerical gradient
def sce_loss_only(logits, target):
    loss, _ = softmax_cross_entropy_loss(logits, target)
    return loss

# Compare analytical vs numerical
analytical = softmax_cross_entropy_gradient(probs_demo, target_demo)
numerical = numerical_gradient(sce_loss_only, logits, target_demo)

print(f"\nVerifying Softmax + Cross-Entropy gradient:")
print(f"  {'':>10s} {'Analytical':>12s} {'Numerical':>12s} {'Match?':>8s}")
print(f"  {'─' * 46}")
all_match = True
for i in range(3):
    match = abs(analytical[i] - numerical[i]) < 1e-4
    status = "✓" if match else "✗"
    if not match:
        all_match = False
    print(f"  Class {i}:  {analytical[i]:>+12.6f} {numerical[i]:>+12.6f}     {status}")

print(f"\n  All gradients match: {'YES ✓' if all_match else 'NO ✗ — BUG!'}")
print("""
Why numerical gradient checking matters:
  When you implement backprop (Part 2), you'll compute gradients analytically.
  Analytical = fast but error-prone (easy to get the math wrong).
  Numerical = slow but almost impossible to get wrong.
  
  ALWAYS verify your analytical gradients with numerical ones.
  This is how researchers debug their implementations.
""")


# =============================================
# SECTION 7: CONNECT TO THE CNN FROM DAY 1
# =============================================

print("=" * 60)
print("SECTION 7: How Wrong is Our Day 1 CNN?")
print("=" * 60)

# Rebuild TinyCNN from Day 1 (forward pass only)
def conv2d_multi(input_vol, weights, biases):
    """Multi-channel convolution from Day 1."""
    c_out, c_in, kh, kw = weights.shape
    _, ih, iw = input_vol.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(input_vol, ((0,0),(pad_h,pad_h),(pad_w,pad_w)), mode='constant')
    output = np.zeros((c_out, ih, iw), dtype=np.float32)
    for f in range(c_out):
        for y in range(ih):
            for x in range(iw):
                val = 0.0
                for c in range(c_in):
                    region = padded[c, y:y+kh, x:x+kw]
                    val += np.sum(region * weights[f, c])
                output[f, y, x] = val + biases[f]
    return output

def relu(x):
    """ReLU from Day 1."""
    return np.maximum(0, x)

def max_pool(x, size=2, stride=2):
    """Max pooling from Day 1."""
    c, h, w = x.shape
    oh, ow = h // stride, w // stride
    out = np.zeros((c, oh, ow), dtype=np.float32)
    for ch in range(c):
        for y in range(oh):
            for xp in range(ow):
                region = x[ch, y*stride:y*stride+size, xp*stride:xp*stride+size]
                out[ch, y, xp] = np.max(region)
    return out

def dense(x, weights, biases):
    """Dense layer from Day 1."""
    return x @ weights + biases

# Build the CNN with same random seed as Day 1
np.random.seed(42)

conv1_w = np.random.randn(4, 1, 3, 3).astype(np.float32) * np.sqrt(2.0 / 9)
conv1_b = np.zeros(4, dtype=np.float32)
conv2_w = np.random.randn(8, 4, 3, 3).astype(np.float32) * np.sqrt(2.0 / 36)
conv2_b = np.zeros(8, dtype=np.float32)
fc1_w = np.random.randn(32, 16).astype(np.float32) * np.sqrt(2.0 / 32)
fc1_b = np.zeros(16, dtype=np.float32)
fc2_w = np.random.randn(16, 3).astype(np.float32) * np.sqrt(2.0 / 16)
fc2_b = np.zeros(3, dtype=np.float32)

def cnn_forward(x):
    """Full forward pass of TinyCNN from Day 1."""
    z1 = conv2d_multi(x, conv1_w, conv1_b)
    a1 = relu(z1)
    p1 = max_pool(a1)
    z2 = conv2d_multi(p1, conv2_w, conv2_b)
    a2 = relu(z2)
    p2 = max_pool(a2)
    flat = p2.flatten()
    d1 = relu(dense(flat, fc1_w, fc1_b))
    logits_out = dense(d1, fc2_w, fc2_b)
    probs_out = softmax(logits_out)
    return logits_out, probs_out

# Create test images (same as Day 1)
img_names = ["Horizontal Edge", "Vertical Edge", "Uniform"]

img1 = np.zeros((1, 8, 8), dtype=np.float32)
img1[0, 0:4, :] = 200
img1[0, 4:8, :] = 50

img2 = np.zeros((1, 8, 8), dtype=np.float32)
img2[0, :, 0:4] = 200
img2[0, :, 4:8] = 50

img3 = np.ones((1, 8, 8), dtype=np.float32) * 128

images = [img1, img2, img3]
labels = [0, 1, 2]

# Compute loss for each image
print(f"\n{'Image':>20s} {'True Class':>12s} {'Predicted':>12s} {'CE Loss':>10s} {'Correct?':>10s}")
print("─" * 68)

total_loss = 0
all_losses = []
all_probs_list = []

for img, label, name in zip(images, labels, img_names):
    tgt = one_hot(label, 3)
    lgts, prbs = cnn_forward(img)
    ce = cross_entropy_loss(prbs, tgt)
    pred_class = np.argmax(prbs)
    correct = "✓" if pred_class == label else "✗"
    
    total_loss += ce
    all_losses.append(ce)
    all_probs_list.append(prbs)
    
    print(f"{name:>20s} {label:>12d} {pred_class:>12d} {ce:>10.4f} {correct:>10s}")

avg_loss = total_loss / 3
print(f"\n  Average CE Loss: {avg_loss:.4f}")
correct_count = sum(1 for i, l in enumerate(labels) if np.argmax(all_probs_list[i]) == l)
print(f"  Accuracy: {correct_count}/3")

print(f"""
The network is making {'RANDOM' if avg_loss > 0.8 else 'poor'} predictions (loss = {avg_loss:.2f}).
For comparison:
  Perfect predictions:   loss ~ 0.01
  Random guessing (1/3): loss ~ {-np.log(1/3):.2f}  (= -log(1/3))
  Completely wrong:      loss ~ 16.12  (= -log(1e-7))

Our loss of {avg_loss:.2f} shows the network hasn't learned anything yet.
The weights are random, so the predictions are random!

NEXT STEP (Part 2): 
  We computed the loss. Now we need to compute the GRADIENT of the loss
  with respect to EVERY weight in the network.
  
  Then (Part 3): use those gradients to UPDATE the weights.
  Repeat 1000 times → the network LEARNS.
""")


# =============================================
# SECTION 8: VISUALIZATION
# =============================================

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Day 2, Part 1 — Loss Functions\n'
             'Understanding How to Measure "Wrongness"',
             fontsize=14, fontweight='bold')

# Plot 1: MSE vs CE loss curves
x_plot = np.linspace(0.01, 0.99, 100)
mse_vals = [(1 - p)**2 for p in x_plot]  # MSE when target=1
ce_vals = [-np.log(p) for p in x_plot]    # CE when target=1

axes[0, 0].plot(x_plot, mse_vals, 'b-', linewidth=2, label='MSE')
axes[0, 0].plot(x_plot, ce_vals, 'r-', linewidth=2, label='Cross-Entropy')
axes[0, 0].set_xlabel('Predicted probability for correct class')
axes[0, 0].set_ylabel('Loss value')
axes[0, 0].set_title('MSE vs Cross-Entropy\n(lower = better)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim(0, 5)

# Plot 2: Gradient comparison
mse_grad_vals = [-2 * (1 - p) for p in x_plot]
ce_grad_vals = [-1.0 / p for p in x_plot]

axes[0, 1].plot(x_plot, mse_grad_vals, 'b-', linewidth=2, label='MSE gradient')
axes[0, 1].plot(x_plot, np.clip(ce_grad_vals, -10, 0), 'r-', linewidth=2, label='CE gradient')
axes[0, 1].set_xlabel('Predicted probability for correct class')
axes[0, 1].set_ylabel('Gradient magnitude')
axes[0, 1].set_title('Gradient Comparison\n(steeper = faster learning)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim(-10, 1)

# Plot 3: Log function
x_log = np.linspace(0.01, 1.0, 100)
axes[0, 2].plot(x_log, -np.log(x_log), 'r-', linewidth=2)
axes[0, 2].set_xlabel('Predicted probability')
axes[0, 2].set_ylabel('-log(p)')
axes[0, 2].set_title('The -log Function\n(why CE penalizes low confidence)')
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].axhline(y=-np.log(0.5), color='gray', linestyle='--', alpha=0.5)
axes[0, 2].annotate('-log(0.5) = 0.69', xy=(0.5, -np.log(0.5)),
                    fontsize=9, ha='left')

# Plot 4-6: Loss for each test image
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
for i, (prbs, name, lss) in enumerate(zip(all_probs_list, img_names, all_losses)):
    bars = axes[1, i].bar(img_names, prbs, color=colors)
    axes[1, i].set_title(f'{name}\nCE Loss = {lss:.4f}', fontsize=10)
    axes[1, i].set_ylim(0, 1)
    axes[1, i].set_ylabel('Probability')
    axes[1, i].tick_params(axis='x', rotation=30, labelsize=8)
    
    # Highlight correct class
    bars[labels[i]].set_edgecolor('black')
    bars[labels[i]].set_linewidth(2)
    
    # Show gradient arrow for correct class
    grd = softmax_cross_entropy_gradient(prbs, one_hot(labels[i], 3))
    axes[1, i].annotate(f'grad={grd[labels[i]]:+.2f}',
                       xy=(labels[i], prbs[labels[i]]),
                       xytext=(labels[i], prbs[labels[i]] + 0.15),
                       fontsize=8, ha='center', color='red',
                       arrowprops=dict(arrowstyle='->', color='red'))

plt.tight_layout()
plt.savefig('part1_results.png', dpi=150, bbox_inches='tight')
print("Saved: part1_results.png")


# =============================================
# FINAL SUMMARY
# =============================================

print("\n" + "=" * 60)
print("DAY 2 PART 1 COMPLETE — LOSS FUNCTIONS")
print("=" * 60)
print("""
╔══════════════════════════════════════════════════════════╗
║                 WHAT YOU LEARNED                         ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  1. One-Hot Encoding:                                    ║
║     class label → binary vector [0, 1, 0]               ║
║                                                          ║
║  2. MSE Loss:                                            ║
║     (1/n) * sum( (predicted - target)^2 )               ║
║     Simple, but weak for classification                  ║
║                                                          ║
║  3. Cross-Entropy Loss:                                  ║
║     -sum( target * log(predicted) )                      ║
║     SCREAMS when wrong, whispers when right              ║
║                                                          ║
║  4. Why CE > MSE:                                        ║
║     CE gradient is HUGE when prediction is wrong         ║
║     → faster learning from mistakes                      ║
║                                                          ║
║  5. Softmax + CE combined:                               ║
║     gradient = predicted - target                        ║
║     THE most elegant formula in deep learning            ║
║                                                          ║
║  6. Numerical gradient checking:                         ║
║     Always verify your math! (f(x+h) - f(x-h)) / 2h    ║
║                                                          ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  KEY EQUATION TO MEMORIZE:                               ║
║                                                          ║
║    gradient = softmax(logits) - one_hot(target)          ║
║                                                          ║
║  This is where ALL learning starts.                      ║
║  Part 2 will push this gradient BACKWARD.                ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

print("=" * 60)
print("PART 1 COMPLETE — Next: Part 2 (Backpropagation)")
print("=" * 60)
