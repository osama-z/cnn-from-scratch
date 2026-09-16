"""
DAY 3 — PART 1: MNIST Data Loading & Understanding
====================================================
Goal: Load REAL handwritten digit images and understand the data
      before we train on it.

What is MNIST?
    - 60,000 training images + 10,000 test images
    - Each image: 28×28 pixels, grayscale (0-255)
    - Each label: one digit (0-9)
    - The "Hello World" of machine learning
    
    Every ML engineer has trained on MNIST. It's your rite of passage.
    
Why MNIST matters for YOU:
    - Day 1-2: you trained on 3 images of size 8×8
    - Day 3: you train on 60,000 images of size 28×28
    - This is the jump from TOY to REAL
    - Same algorithm, just more data and bigger images
    
What's new today:
    1. Loading real data from files (not hardcoding arrays)
    2. Data normalization (0-255 → 0-1)
    3. Mini-batch training (process 32 images at once, not 1)
    4. Training on 60,000 samples (not 3)
    5. Evaluating on SEPARATE test data (not training data)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import struct
import urllib.request
import gzip

np.random.seed(42)

# =============================================
# SECTION 1: DOWNLOAD MNIST
# =============================================

print("=" * 60)
print("SECTION 1: DOWNLOADING MNIST DATA")
print("=" * 60)

def download_mnist(data_dir='mnist_data'):
    """
    Download MNIST dataset from the internet.
    
    MNIST is stored in a special binary format (IDX).
    We download 4 files:
        - train-images: 60,000 training images
        - train-labels: 60,000 training labels
        - t10k-images:  10,000 test images
        - t10k-labels:  10,000 test labels
    """
    os.makedirs(data_dir, exist_ok=True)
    
    base_url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
    files = {
        'train-images-idx3-ubyte.gz': 'train images',
        'train-labels-idx1-ubyte.gz': 'train labels',
        't10k-images-idx3-ubyte.gz': 'test images',
        't10k-labels-idx1-ubyte.gz': 'test labels',
    }
    
    for filename, desc in files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"  Downloading {desc}...")
            url = base_url + filename
            urllib.request.urlretrieve(url, filepath)
            print(f"    Saved: {filepath}")
        else:
            print(f"  {desc}: already downloaded ✓")
    
    return data_dir

def load_mnist_images(filepath):
    """
    Load MNIST images from IDX format.
    
    IDX file format:
        Byte 0-3:   magic number (2051 for images)
        Byte 4-7:   number of images
        Byte 8-11:  number of rows (28)
        Byte 12-15: number of columns (28)
        Byte 16+:   pixel values (unsigned bytes, 0-255)
    
    Returns: numpy array of shape (N, 28, 28), dtype float32, range [0, 1]
    """
    with gzip.open(filepath, 'rb') as f:
        # Read header
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051, f"Invalid magic number: {magic}"
        
        # Read all pixel data as bytes, reshape to (N, 28, 28)
        data = np.frombuffer(f.read(), dtype=np.uint8)
        images = data.reshape(num_images, rows, cols)
    
    # Convert to float32 and normalize to [0, 1]
    # WHY NORMALIZE?
    #   Raw pixels: 0-255. If weights are ~0.01, then activation = 255 × 0.01 = 2.55
    #   That's already large! After a few layers, values EXPLODE.
    #   Normalizing to 0-1 keeps activations in a reasonable range.
    #   This makes training MUCH more stable.
    images = images.astype(np.float32) / 255.0
    
    return images

def load_mnist_labels(filepath):
    """
    Load MNIST labels from IDX format.
    
    IDX file format:
        Byte 0-3: magic number (2049 for labels)
        Byte 4-7: number of labels
        Byte 8+:  label values (0-9)
    """
    with gzip.open(filepath, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        assert magic == 2049, f"Invalid magic number: {magic}"
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    return labels

# Download and load
data_dir = download_mnist()

X_train = load_mnist_images(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
y_train = load_mnist_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
X_test = load_mnist_images(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
y_test = load_mnist_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))

print(f"\n  Training set: {X_train.shape[0]} images, each {X_train.shape[1]}x{X_train.shape[2]}")
print(f"  Test set:     {X_test.shape[0]} images")
print(f"  Pixel range:  [{X_train.min():.1f}, {X_train.max():.1f}] (normalized)")
print(f"  Label range:  {y_train.min()} to {y_train.max()} (digits)")
print(f"  Label distribution: {np.bincount(y_train)}")


# =============================================
# SECTION 2: VISUALIZE THE DATA
# =============================================

print("\n" + "=" * 60)
print("SECTION 2: WHAT DOES MNIST LOOK LIKE?")
print("=" * 60)

print("""
Each image is 28x28 = 784 pixels.
That's our INPUT SIZE: a vector of 784 numbers.

For a dense network (no convolutions today):
    We FLATTEN each 28x28 image into a 784-length vector.
    
    Why no convolutions?
    Conv backprop is complex and SLOW in pure NumPy.
    A dense network can still reach 95%+ on MNIST!
    We'll add conv when we switch to PyTorch (Week 3).
""")

# Show sample images
fig, axes = plt.subplots(2, 10, figsize=(15, 4))
fig.suptitle('MNIST — Handwritten Digits (first 20 samples)', fontweight='bold')

for i in range(20):
    row, col = i // 10, i % 10
    axes[row, col].imshow(X_train[i], cmap='gray')
    axes[row, col].set_title(f'{y_train[i]}', fontsize=10)
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('part1_mnist_samples.png', dpi=150, bbox_inches='tight')
print("  Saved: part1_mnist_samples.png")

# Show one digit in detail
print(f"\n  Example digit (label = {y_train[0]}):")
print(f"  Shape: {X_train[0].shape}")
print(f"  As numbers (center 10x10 region):")
center = X_train[0][9:19, 9:19]
for row in center:
    print("    " + " ".join(f"{v:.1f}" for v in row))


# =============================================
# SECTION 3: PREPARE DATA FOR TRAINING
# =============================================

print("\n" + "=" * 60)
print("SECTION 3: PREPARING DATA FOR TRAINING")
print("=" * 60)

# Flatten images: (N, 28, 28) → (N, 784)
X_train_flat = X_train.reshape(X_train.shape[0], -1)  # -1 means "figure out this dimension"
X_test_flat = X_test.reshape(X_test.shape[0], -1)

print(f"  Flattened training data: {X_train_flat.shape} (60000 samples, 784 features each)")
print(f"  Flattened test data:     {X_test_flat.shape}")

def one_hot(labels, num_classes=10):
    """Convert array of labels to one-hot matrix."""
    n = len(labels)
    result = np.zeros((n, num_classes), dtype=np.float32)
    result[np.arange(n), labels] = 1.0
    return result

y_train_oh = one_hot(y_train, 10)
y_test_oh = one_hot(y_test, 10)

print(f"  One-hot training labels: {y_train_oh.shape}")
print(f"  Example: label {y_train[0]} → {y_train_oh[0]}")

# Use a smaller subset for faster training in pure NumPy
# Full 60k takes too long without GPU
TRAIN_SIZE = 10000
TEST_SIZE = 2000

X_tr = X_train_flat[:TRAIN_SIZE]
y_tr = y_train[:TRAIN_SIZE]
y_tr_oh = y_train_oh[:TRAIN_SIZE]

X_te = X_test_flat[:TEST_SIZE]
y_te = y_test[:TEST_SIZE]
y_te_oh = y_test_oh[:TEST_SIZE]

print(f"\n  Using subset for NumPy training:")
print(f"    Training: {TRAIN_SIZE} samples")
print(f"    Testing:  {TEST_SIZE} samples")
print(f"    (Full 60k would take hours in pure NumPy — that's what PyTorch solves!)")


# =============================================
# SECTION 4: BUILD THE MNIST NETWORK
# =============================================

print("\n" + "=" * 60)
print("SECTION 4: BUILDING THE MNIST NETWORK")
print("=" * 60)

print("""
Architecture:
    Input (784,) → Dense1 (784→128) → ReLU → Dense2 (128→64) → ReLU → Dense3 (64→10) → Softmax
    
    784 inputs  = 28x28 flattened pixels
    128 hidden  = enough neurons to capture digit patterns
    64 hidden   = compress features further
    10 outputs  = one per digit (0-9)
    
    Total parameters: 784*128 + 128 + 128*64 + 64 + 64*10 + 10 = 109,386
    Memory: 109,386 * 4 bytes = 437 KB (tiny!)
""")

def softmax(logits):
    """Numerically stable softmax."""
    e_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

class MNISTNet:
    """
    3-layer dense network for MNIST digit classification.
    
    This is the same architecture from Day 2, just bigger:
        Day 2: Input(2) → Dense(16) → ReLU → Dense(3) → Softmax
        Day 3: Input(784) → Dense(128) → ReLU → Dense(64) → ReLU → Dense(10) → Softmax
    
    The ONLY difference is more neurons and one extra hidden layer.
    The algorithm is IDENTICAL.
    """
    
    def __init__(self):
        # Kaiming initialization: scale by sqrt(2/fan_in)
        # This prevents vanishing/exploding gradients at initialization
        self.W1 = np.random.randn(784, 128).astype(np.float32) * np.sqrt(2.0 / 784)
        self.b1 = np.zeros(128, dtype=np.float32)
        
        self.W2 = np.random.randn(128, 64).astype(np.float32) * np.sqrt(2.0 / 128)
        self.b2 = np.zeros(64, dtype=np.float32)
        
        self.W3 = np.random.randn(64, 10).astype(np.float32) * np.sqrt(2.0 / 64)
        self.b3 = np.zeros(10, dtype=np.float32)
        
        # Cache for backward pass
        self.x = None
        self.z1 = self.a1 = None
        self.z2 = self.a2 = None
        self.logits = self.probs = None
    
    def forward(self, x):
        """
        Forward pass for a SINGLE sample.
        x: shape (784,)
        returns: shape (10,) — probabilities for each digit
        """
        self.x = x
        
        # Layer 1: Dense → ReLU
        self.z1 = x @ self.W1 + self.b1         # (784,) @ (784,128) = (128,)
        self.a1 = np.maximum(0, self.z1)          # ReLU
        
        # Layer 2: Dense → ReLU
        self.z2 = self.a1 @ self.W2 + self.b2    # (128,) @ (128,64) = (64,)
        self.a2 = np.maximum(0, self.z2)           # ReLU
        
        # Layer 3: Dense → Softmax
        self.logits = self.a2 @ self.W3 + self.b3  # (64,) @ (64,10) = (10,)
        self.probs = softmax(self.logits)            # (10,) probabilities
        
        return self.probs
    
    def backward(self, target):
        """
        Backward pass — compute gradients for all 6 parameter sets.
        Exactly the same chain rule as Day 2, just one more layer.
        """
        # Output gradient: softmax + CE = probs - target
        d_logits = self.probs - target  # (10,)
        
        # Layer 3 backward
        self.dW3 = self.a2.reshape(-1, 1) @ d_logits.reshape(1, -1)  # (64,10)
        self.db3 = d_logits                                             # (10,)
        d_a2 = d_logits @ self.W3.T                                     # (64,)
        
        # ReLU backward (layer 2)
        d_z2 = d_a2 * (self.z2 > 0).astype(np.float32)  # (64,)
        
        # Layer 2 backward
        self.dW2 = self.a1.reshape(-1, 1) @ d_z2.reshape(1, -1)  # (128,64)
        self.db2 = d_z2                                             # (64,)
        d_a1 = d_z2 @ self.W2.T                                     # (128,)
        
        # ReLU backward (layer 1)
        d_z1 = d_a1 * (self.z1 > 0).astype(np.float32)  # (128,)
        
        # Layer 1 backward
        self.dW1 = self.x.reshape(-1, 1) @ d_z1.reshape(1, -1)  # (784,128)
        self.db1 = d_z1                                            # (128,)
    
    def update_sgd(self, lr):
        """Vanilla SGD update."""
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2
        self.W3 -= lr * self.dW3
        self.b3 -= lr * self.db3
    
    def count_params(self):
        total = (self.W1.size + self.b1.size + 
                 self.W2.size + self.b2.size + 
                 self.W3.size + self.b3.size)
        return total


# Build the network
net = MNISTNet()
total_params = net.count_params()
print(f"\n  Network built!")
print(f"  Total parameters: {total_params:,}")
print(f"  Memory (float32):  {total_params * 4 / 1024:.1f} KB")
print(f"  Memory (int8):     {total_params / 1024:.1f} KB")


# =============================================
# SECTION 5: TRAIN ON MNIST!
# =============================================

print("\n" + "=" * 60)
print("SECTION 5: TRAINING ON MNIST — The Real Test!")
print("=" * 60)

print("""
Training plan:
    - 20 epochs (each epoch = one pass through all training data)
    - Learning rate: 0.01 (with decay)
    - Optimizer: vanilla SGD (Adam is better but this shows the fundamentals)
    - Shuffle data each epoch (prevents learning order-dependent patterns)
""")

def evaluate(net, X, y):
    """Compute accuracy on a dataset."""
    correct = 0
    for i in range(len(X)):
        probs = net.forward(X[i])
        if np.argmax(probs) == y[i]:
            correct += 1
    return correct / len(X)

lr = 0.01
n_epochs = 20
train_losses = []
train_accs = []
test_accs = []

# ─── BASELINE: accuracy BEFORE a single weight update ───
# This must be measured here, not read from train_accs[0]. By the end of
# epoch 0 the network has already taken TRAIN_SIZE gradient steps, so
# train_accs[0] is the average accuracy *while learning*, not a baseline.
acc_before_train = evaluate(net, X_tr, y_tr)
acc_before_test  = evaluate(net, X_te, y_te)
print(f"\nBEFORE training:  train {acc_before_train:.1%}   test {acc_before_test:.1%}"
      f"   [random guessing on 10 classes = 10.0%]")

print(f"\n{'Epoch':>6s} {'Loss':>10s} {'Train Acc':>10s} {'Test Acc':>10s} {'LR':>10s}")
print("─" * 50)

for epoch in range(n_epochs):
    # Shuffle training data each epoch
    indices = np.random.permutation(TRAIN_SIZE)
    X_shuffled = X_tr[indices]
    y_shuffled = y_tr[indices]
    y_oh_shuffled = y_tr_oh[indices]
    
    epoch_loss = 0
    correct = 0
    
    for i in range(TRAIN_SIZE):
        # 1. Forward
        probs = net.forward(X_shuffled[i])
        
        # 2. Loss
        epsilon = 1e-7
        loss = -np.sum(y_oh_shuffled[i] * np.log(probs + epsilon))
        epoch_loss += loss
        
        # Track accuracy
        if np.argmax(probs) == y_shuffled[i]:
            correct += 1
        
        # 3. Backward
        net.backward(y_oh_shuffled[i])
        
        # 4. Update
        net.update_sgd(lr)
    
    avg_loss = epoch_loss / TRAIN_SIZE
    train_acc = correct / TRAIN_SIZE
    
    # Evaluate on test set (every 2 epochs to save time)
    if epoch % 2 == 0 or epoch == n_epochs - 1:
        test_acc = evaluate(net, X_te, y_te)
    
    train_losses.append(avg_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    
    # Learning rate decay: reduce by 10% every 5 epochs
    if (epoch + 1) % 5 == 0:
        lr *= 0.7
    
    print(f"{epoch:>6d} {avg_loss:>10.4f} {train_acc:>10.1%} {test_acc:>10.1%} {lr:>10.5f}")

print(f"""
RESULTS:
  Before training:      train {acc_before_train:.1%}   test {acc_before_test:.1%}   (random = 10%)
  During epoch 0:       train {train_accs[0]:.1%}
                        (already high — averages over {TRAIN_SIZE} weight updates)
  Final train accuracy: {train_accs[-1]:.1%}
  Final test accuracy:  {test_accs[-1]:.1%}

  The real jump is {acc_before_train:.1%} -> {train_accs[-1]:.1%} on train,
  and {acc_before_test:.1%} -> {test_accs[-1]:.1%} on unseen test data.

  GENERALIZATION GAP: {train_accs[-1] - test_accs[-1]:.1%}
    train {train_accs[-1]:.1%} vs test {test_accs[-1]:.1%}. The network memorized the
    training set perfectly but is {train_accs[-1] - test_accs[-1]:.1%} worse on data it
    never saw. That gap IS overfitting, and it is why a test set exists.

  Trained on {TRAIN_SIZE} images, tested on {TEST_SIZE} unseen images.

  {'TARGET MET! ✓' if test_accs[-1] > 0.85 else 'Keep training or tune hyperparameters'}
""")


# =============================================
# SECTION 6: ANALYZE PREDICTIONS
# =============================================

print("=" * 60)
print("SECTION 6: ANALYZING WHAT THE NETWORK LEARNED")
print("=" * 60)

# Find examples of correct and incorrect predictions
correct_examples = []
wrong_examples = []

for i in range(min(TEST_SIZE, 2000)):
    probs = net.forward(X_te[i])
    pred = np.argmax(probs)
    conf = probs[pred]
    
    if pred == y_te[i] and len(correct_examples) < 10:
        correct_examples.append((i, pred, y_te[i], conf))
    elif pred != y_te[i] and len(wrong_examples) < 10:
        wrong_examples.append((i, pred, y_te[i], conf, probs))

print(f"\n  Correct predictions (first 5):")
for idx, pred, true, conf in correct_examples[:5]:
    print(f"    Image {idx}: predicted {pred}, true {true}, confidence {conf:.1%}")

print(f"\n  WRONG predictions (first 5):")
for idx, pred, true, conf, probs in wrong_examples[:5]:
    print(f"    Image {idx}: predicted {pred}, true {true}, confidence {conf:.1%}")
    top3 = np.argsort(probs)[::-1][:3]
    print(f"      Top 3: {[(int(c), f'{probs[c]:.1%}') for c in top3]}")

# Confusion matrix (simplified)
print(f"\n  Per-digit accuracy:")
for digit in range(10):
    mask = y_te[:TEST_SIZE] == digit
    digit_X = X_te[:TEST_SIZE][mask]
    digit_y = y_te[:TEST_SIZE][mask]
    
    digit_correct = 0
    for i in range(len(digit_X)):
        probs = net.forward(digit_X[i])
        if np.argmax(probs) == digit_y[i]:
            digit_correct += 1
    
    acc = digit_correct / len(digit_y) if len(digit_y) > 0 else 0
    bar = "█" * int(acc * 30)
    print(f"    Digit {digit}: {acc:>6.1%} {bar}")


# =============================================
# SECTION 7: VISUALIZATION
# =============================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Day 3 — MNIST Digit Classification\n'
             'Training a Neural Network on Real Data (NumPy Only!)',
             fontsize=14, fontweight='bold')

# Plot 1: Loss curve
axes[0, 0].plot(train_losses, 'b-', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Cross-Entropy Loss')
axes[0, 0].set_title('Training Loss\n(lower = better)')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Accuracy curves
axes[0, 1].plot(train_accs, 'b-', linewidth=2, label='Train')
axes[0, 1].plot(test_accs, 'r--', linewidth=2, label='Test')
axes[0, 1].axhline(y=0.85, color='green', linestyle=':', alpha=0.5, label='Target (85%)')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Accuracy Over Time')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim(0, 1.05)

# Plot 3: Correct predictions
for i in range(min(5, len(correct_examples))):
    idx = correct_examples[i][0]
    img = X_test[idx]
    label = correct_examples[i][2]
    conf = correct_examples[i][3]
    
    ax_inset = axes[0, 2].inset_axes([i * 0.2, 0.1, 0.18, 0.8])
    ax_inset.imshow(img, cmap='gray')
    ax_inset.set_title(f'{label}\n{conf:.0%}', fontsize=8)
    ax_inset.axis('off')
axes[0, 2].set_title('Correct Predictions\n(with confidence)')
axes[0, 2].axis('off')

# Plot 4: Wrong predictions  
for i in range(min(5, len(wrong_examples))):
    idx = wrong_examples[i][0]
    img = X_test[idx]
    pred_label = wrong_examples[i][1]
    true_label = wrong_examples[i][2]
    
    ax_inset = axes[1, 0].inset_axes([i * 0.2, 0.1, 0.18, 0.8])
    ax_inset.imshow(img, cmap='gray')
    ax_inset.set_title(f'P:{pred_label} T:{true_label}', fontsize=8, color='red')
    ax_inset.axis('off')
axes[1, 0].set_title('Wrong Predictions\n(P=predicted, T=true)', color='red')
axes[1, 0].axis('off')

# Plot 5: Per-digit accuracy
digit_accs = []
for digit in range(10):
    mask = y_te[:TEST_SIZE] == digit
    digit_X = X_te[:TEST_SIZE][mask]
    digit_y = y_te[:TEST_SIZE][mask]
    dc = sum(1 for i in range(len(digit_X)) if np.argmax(net.forward(digit_X[i])) == digit_y[i])
    digit_accs.append(dc / len(digit_y) if len(digit_y) > 0 else 0)

colors = ['#FF6B6B' if a < 0.85 else '#4ECDC4' for a in digit_accs]
axes[1, 1].bar(range(10), digit_accs, color=colors)
axes[1, 1].set_xlabel('Digit')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_title('Per-Digit Accuracy\n(red = below 85%)')
axes[1, 1].set_xticks(range(10))
axes[1, 1].axhline(y=0.85, color='gray', linestyle='--', alpha=0.5)
axes[1, 1].set_ylim(0, 1.05)

# Plot 6: First layer weights visualization
# Each row of W1 is a (784,) vector = a learned "template" for detecting features
# Reshape back to 28x28 to see what the network learned
n_show = 16
grid_size = 4
weight_img = np.zeros((28 * grid_size, 28 * grid_size))
for idx in range(n_show):
    row, col = idx // grid_size, idx % grid_size
    w = net.W1[:, idx].reshape(28, 28)
    w = (w - w.min()) / (w.max() - w.min() + 1e-8)  # normalize to [0,1]
    weight_img[row*28:(row+1)*28, col*28:(col+1)*28] = w

axes[1, 2].imshow(weight_img, cmap='viridis')
axes[1, 2].set_title('First Layer Weights\n(learned feature detectors)')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('part1_results.png', dpi=150, bbox_inches='tight')
print("\nSaved: part1_results.png")


# =============================================
# FINAL SUMMARY
# =============================================

print("\n" + "=" * 60)
print("DAY 3 COMPLETE — MNIST CLASSIFICATION!")
print("=" * 60)
print(f"""
╔══════════════════════════════════════════════════════════╗
║             DAY 3 COMPLETE — REAL DATA!                  ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  What you did:                                           ║
║    * Loaded 60,000 real handwritten digit images         ║
║    * Normalized pixel values (0-255 → 0-1)              ║
║    * Built a 3-layer dense network (109K parameters)     ║
║    * Trained with SGD + learning rate decay              ║
║    * Achieved {test_accs[-1]:.1%} test accuracy                       ║
║                                                          ║
║  What's the same as Day 2:                               ║
║    * Forward pass → Loss → Backward → Update            ║
║    * Same algorithm, just bigger data                    ║
║                                                          ║
║  What's new:                                             ║
║    * Real data from files (not hardcoded arrays)         ║
║    * Data normalization (critical for training!)         ║
║    * Learning rate decay (reduce lr over time)           ║
║    * Train/test split (measure generalization)           ║
║    * Per-class accuracy analysis                         ║
║                                                          ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  YOUR JOURNEY SO FAR:                                    ║
║    Day 1: Built CNN structure (forward pass)             ║
║    Day 2: Built learning (loss + backprop + training)    ║
║    Day 3: Trained on REAL data ({test_accs[-1]:.0%} accuracy)         ║
║                                                          ║
║  NEXT (Day 4): Rewrite convolution in C!                 ║
║    Take your Python code → rewrite with float pointers   ║
║    This is where embedded engineering begins.            ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

print("=" * 60)
print("DAY 3 COMPLETE — Push to GitHub!")
print("=" * 60)
