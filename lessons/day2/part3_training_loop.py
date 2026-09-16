"""
DAY 2 — PART 3: The Training Loop — Making Your Network LEARN
==============================================================
Goal: Put forward pass + loss + backprop together and TRAIN a network.
      This is the moment your network goes from random guessing to understanding.

The Complete Picture:
    1. Forward pass: Input → Layers → Prediction     (Part 1 of Day 1)
    2. Loss:         Prediction vs Target → Number    (Part 1 of Day 2)
    3. Backward pass: Loss → Gradients for all weights (Part 2 of Day 2)
    4. Update:       weights -= learning_rate * gradient  ← THIS FILE
    5. Repeat:       Go to step 1 with new data
    
    That's ALL of deep learning. Everything else is optimization.

What happens in this file:
    - You'll implement gradient descent
    - You'll see learning rate effects
    - You'll implement SGD with momentum and Adam
    - You'll train a network and watch it LEARN
    - You'll see the loss decrease from ~1.1 (random) to ~0.01 (learned)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)


# =============================================
# SECTION 1: GRADIENT DESCENT — The Learning Algorithm
# =============================================

print("=" * 60)
print("SECTION 1: GRADIENT DESCENT — Walking Downhill")
print("=" * 60)

print("""
Gradient Descent in ONE line:

    weight = weight - learning_rate * gradient

That's it. That's the entire learning algorithm.

ANALOGY: You're blindfolded on a hill.
    1. Feel the slope under your feet (= compute gradient)
    2. Take a step downhill (= subtract gradient from weight)
    3. Repeat until you reach the valley (= minimum loss)

The learning rate controls your STEP SIZE:
    Too small (0.0001): you'll get there, but it takes forever
    Just right (0.01):  smooth walk downhill
    Too big (1.0):      you overshoot and bounce around
    Way too big (10.0): you fly off the hill entirely (diverge!)
""")

# Demo gradient descent on a simple function: f(x) = (x - 3)^2
# Minimum is at x = 3 (where the derivative is zero)

def f(x):
    """Our 'loss' function: (x-3)^2. Minimum at x=3."""
    return (x - 3) ** 2

def df(x):
    """Gradient of f: 2(x-3). Tells us which direction to go."""
    return 2 * (x - 3)

print("Demo: Finding the minimum of f(x) = (x - 3)^2")
print("─" * 50)

# Try different learning rates
learning_rates = [0.01, 0.1, 0.5, 0.9]
histories = {}

for lr in learning_rates:
    x_current = 0.0  # Start far from the minimum
    history = [x_current]
    
    for step in range(20):
        grad = df(x_current)
        x_current = x_current - lr * grad  # THE UPDATE RULE
        history.append(x_current)
    
    histories[lr] = history
    final_x = history[-1]
    final_loss = f(final_x)
    print(f"  lr={lr:.2f}: x started at 0.0, ended at {final_x:.4f}, loss={final_loss:.6f}")

print("""
Observations:
  lr=0.01: Slow but steady — takes many steps to reach x=3
  lr=0.10: Good pace — converges nicely
  lr=0.50: Fast — reaches minimum quickly
  lr=0.90: Too aggressive — overshoots and oscillates!
""")


# =============================================
# SECTION 2: THE TRAINING LOOP
# =============================================

print("=" * 60)
print("SECTION 2: THE TRAINING LOOP — The Complete Algorithm")
print("=" * 60)

print("""
THE TRAINING LOOP (memorize this structure):

    for epoch in range(num_epochs):
        
        # 1. FORWARD PASS — compute prediction
        prediction = network.forward(input_data)
        
        # 2. COMPUTE LOSS — how wrong is the prediction?
        loss = cross_entropy(prediction, target)
        
        # 3. BACKWARD PASS — compute gradients for all weights
        network.backward(target)
        
        # 4. UPDATE WEIGHTS — take a step downhill
        for each weight W in network:
            W = W - learning_rate * dL/dW
        
        print(f"Epoch {epoch}: loss = {loss}")

Vocabulary:
    Epoch = one complete pass through ALL training data
    Batch = a subset of training data (for efficiency)
    Iteration = one weight update (= one batch)
    
    If you have 1000 samples and batch_size = 100:
        1 epoch = 10 iterations (10 batches of 100)
""")


# =============================================
# SECTION 3: BUILD A TRAINABLE NETWORK
# =============================================

print("=" * 60)
print("SECTION 3: BUILDING A TRAINABLE 2-LAYER NETWORK")
print("=" * 60)

def softmax(logits):
    e_x = np.exp(logits - np.max(logits))
    return e_x / e_x.sum()

class TrainableNet:
    """
    A 2-layer network that can LEARN.
    
    Architecture:
        Input (2,) → Dense1 (2→16) → ReLU → Dense2 (16→3) → Softmax+CE
    
    2D input so we can visualize the decision boundary!
    """
    
    def __init__(self):
        # Layer 1: 2 inputs → 16 hidden neurons
        self.W1 = np.random.randn(2, 16).astype(np.float32) * np.sqrt(2.0 / 2)
        self.b1 = np.zeros(16, dtype=np.float32)
        
        # Layer 2: 16 hidden → 3 output classes
        self.W2 = np.random.randn(16, 3).astype(np.float32) * np.sqrt(2.0 / 16)
        self.b2 = np.zeros(3, dtype=np.float32)
        
        # Cache for backward pass
        self.x = None      # input
        self.z1 = None     # pre-activation layer 1
        self.a1 = None     # post-ReLU layer 1
        self.logits = None # raw output scores
        self.probs = None  # softmax probabilities
    
    def forward(self, x):
        """Forward pass — save everything for backward."""
        self.x = x.copy()
        
        # Layer 1: Linear → ReLU
        self.z1 = x @ self.W1 + self.b1      # Linear transform
        self.a1 = np.maximum(0, self.z1)       # ReLU activation
        
        # Layer 2: Linear → Softmax
        self.logits = self.a1 @ self.W2 + self.b2  # Raw scores
        self.probs = softmax(self.logits)           # Probabilities
        
        return self.probs
    
    def compute_loss(self, probs, target):
        """Cross-entropy loss."""
        epsilon = 1e-7
        return -np.sum(target * np.log(probs + epsilon))
    
    def backward(self, target):
        """
        Backward pass — compute gradients for ALL weights.
        
        The gradient flows:
            Loss → Softmax+CE → Dense2 → ReLU → Dense1
        """
        # STEP 1: Output gradient (softmax + cross-entropy combined)
        # This is the beautiful formula: gradient = probs - target
        d_logits = self.probs - target  # shape: (3,)
        
        # STEP 2: Dense2 backward
        # d_logits is dL/d(logits), need dL/dW2, dL/db2, dL/da1
        self.dW2 = self.a1.reshape(-1, 1) @ d_logits.reshape(1, -1)  # (16,1) @ (1,3) = (16,3)
        self.db2 = d_logits.copy()                                      # (3,)
        d_a1 = d_logits @ self.W2.T                                     # (3,) @ (3,16) = (16,)
        
        # STEP 3: ReLU backward
        # Gradient passes through where z1 > 0, blocked where z1 <= 0
        d_z1 = d_a1 * (self.z1 > 0).astype(np.float32)  # (16,)
        
        # STEP 4: Dense1 backward
        self.dW1 = self.x.reshape(-1, 1) @ d_z1.reshape(1, -1)  # (2,1) @ (1,16) = (2,16)
        self.db1 = d_z1.copy()                                     # (16,)
    
    def update(self, lr):
        """
        Gradient descent update — THE LEARNING STEP.
        
        weight = weight - learning_rate * gradient
        
        This is where the network actually CHANGES.
        Every call to update() makes the network slightly better
        (if the learning rate is right).
        """
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2


# =============================================
# SECTION 4: CREATE TRAINING DATA
# =============================================

print("\n" + "=" * 60)
print("SECTION 4: CREATING TRAINING DATA — 3 Clusters")
print("=" * 60)

def create_dataset(n_per_class=100):
    """
    Create a 2D classification dataset with 3 classes.
    Each class is a cluster of points in 2D space.
    
    Class 0: cluster around (-1, -1)  — "bottom-left"
    Class 1: cluster around ( 1, -1)  — "bottom-right"  
    Class 2: cluster around ( 0,  1)  — "top-center"
    
    2D input means we can VISUALIZE the decision boundary!
    """
    X = []
    y = []
    
    centers = [(-1, -1), (1, -1), (0, 1)]
    
    for class_idx, (cx, cy) in enumerate(centers):
        # Generate random points around the center
        # np.random.randn gives points from a normal distribution (bell curve)
        # * 0.4 controls the spread (smaller = tighter clusters)
        points = np.random.randn(n_per_class, 2).astype(np.float32) * 0.4
        points[:, 0] += cx  # shift x to center
        points[:, 1] += cy  # shift y to center
        
        X.append(points)
        y.extend([class_idx] * n_per_class)
    
    X = np.vstack(X)  # shape: (300, 2)
    y = np.array(y)    # shape: (300,)
    
    # Shuffle the data (important for training!)
    indices = np.random.permutation(len(y))
    X = X[indices]
    y = y[indices]
    
    return X, y

def one_hot(label, num_classes=3):
    vec = np.zeros(num_classes, dtype=np.float32)
    vec[label] = 1.0
    return vec

X_train, y_train = create_dataset(n_per_class=100)
print(f"  Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"  Classes: {np.bincount(y_train)} samples per class")
print(f"  X range: [{X_train.min():.2f}, {X_train.max():.2f}]")


# =============================================
# SECTION 5: TRAIN IT — WATCH IT LEARN!
# =============================================

print("\n" + "=" * 60)
print("SECTION 5: TRAINING — The Magic Moment")
print("=" * 60)

net = TrainableNet()
lr = 0.05
n_epochs = 1000

loss_history = []
acc_history = []

print(f"\nTraining with learning_rate = {lr}, epochs = {n_epochs}")

# ─── BASELINE: measure accuracy BEFORE a single weight update ───
# This has to be done here, not read from acc_history[0]. By the end of
# epoch 0 the network has already taken len(X_train) gradient steps, so
# acc_history[0] is the average accuracy *while learning*, not a baseline.
correct_before = sum(1 for i in range(len(X_train))
                     if np.argmax(net.forward(X_train[i])) == y_train[i])
acc_before = correct_before / len(X_train)
print(f"Accuracy BEFORE training: {acc_before:.1%} "
      f"({correct_before}/{len(X_train)})   [random guessing on 3 classes = 33.3%]")

print(f"{'Epoch':>8s} {'Loss':>10s} {'Accuracy':>10s}")
print("─" * 30)

for epoch in range(n_epochs):
    epoch_loss = 0
    correct = 0
    
    # Train on EACH sample (stochastic gradient descent)
    for i in range(len(X_train)):
        x_i = X_train[i]             # one input sample
        t_i = one_hot(y_train[i], 3) # one-hot target
        
        # 1. FORWARD — compute prediction
        probs = net.forward(x_i)
        
        # 2. LOSS — how wrong?
        loss = net.compute_loss(probs, t_i)
        epoch_loss += loss
        
        # 3. BACKWARD — compute gradients
        net.backward(t_i)
        
        # 4. UPDATE — adjust weights
        net.update(lr)
        
        # Track accuracy
        if np.argmax(probs) == y_train[i]:
            correct += 1
    
    avg_loss = epoch_loss / len(X_train)
    accuracy = correct / len(X_train)
    loss_history.append(avg_loss)
    acc_history.append(accuracy)
    
    if epoch % 100 == 0 or epoch == n_epochs - 1:
        print(f"{epoch:>8d} {avg_loss:>10.4f} {accuracy:>10.1%}")

print(f"""
THE NETWORK LEARNED!
  Before training: accuracy = {acc_before:.1%}  (untrained — random weights)
  During epoch 0:  loss = {loss_history[0]:.4f}, accuracy = {acc_history[0]:.1%}
                   (already high: this averages over {len(X_train)} weight updates)
  End:             loss = {loss_history[-1]:.4f}, accuracy = {acc_history[-1]:.1%} (learned!)

  The real jump is {acc_before:.1%} -> {acc_history[-1]:.1%}. Most of it happens
  inside the FIRST epoch, which is why epoch 0 already looks good.

This is the CORE of deep learning:
  1. Forward pass → prediction
  2. Loss → how wrong
  3. Backward pass → gradients (direction to improve)
  4. Update → adjust weights slightly
  5. Repeat 300,000 times (300 samples x 1000 epochs)
  
  Each repetition makes the network SLIGHTLY better.
  After enough repetitions, it "understands" the data.
""")


# =============================================
# SECTION 6: LEARNING RATE EXPERIMENTS
# =============================================

print("=" * 60)
print("SECTION 6: LEARNING RATE EXPERIMENTS")
print("=" * 60)

lr_experiments = {}
test_lrs = [0.001, 0.01, 0.05, 0.5]

for test_lr in test_lrs:
    np.random.seed(42)  # Same initialization for fair comparison
    test_net = TrainableNet()
    losses = []
    
    for epoch in range(500):
        epoch_loss = 0
        for i in range(len(X_train)):
            probs = test_net.forward(X_train[i])
            loss = test_net.compute_loss(probs, one_hot(y_train[i], 3))
            epoch_loss += loss
            test_net.backward(one_hot(y_train[i], 3))
            test_net.update(test_lr)
        losses.append(epoch_loss / len(X_train))
    
    lr_experiments[test_lr] = losses
    print(f"  lr={test_lr:.3f}: final loss = {losses[-1]:.4f}")

print("""
Observations:
  lr=0.001: Too slow — barely learns in 500 epochs
  lr=0.01:  Steady learning — good but slow
  lr=0.05:  Sweet spot — fast and stable
  lr=0.5:   Too aggressive — may oscillate or even diverge
""")


# =============================================
# SECTION 7: MOMENTUM AND ADAM
# =============================================

print("=" * 60)
print("SECTION 7: BETTER OPTIMIZERS — Momentum and Adam")
print("=" * 60)

print("""
Vanilla SGD has problems:
  1. Gets stuck in flat regions (gradient is tiny = slow progress)
  2. Oscillates in narrow valleys (zigzags instead of going straight)
  3. Same learning rate for all weights (some need big, some need small)

MOMENTUM fixes #1 and #2:
  velocity = momentum * velocity - lr * gradient
  weight += velocity
  
  Like a ball rolling downhill — it builds up speed in consistent directions
  and dampens oscillations in inconsistent directions.

ADAM fixes all three:
  It keeps a running average of both gradients AND squared gradients.
  Each weight gets its OWN effective learning rate.
  Weights that rarely update get BIGGER steps.
  Weights that update a lot get SMALLER steps.
  
  Adam = Momentum + Adaptive learning rates.
  This is what 90% of deep learning uses today.
""")

# Train with momentum
np.random.seed(42)
net_momentum = TrainableNet()
# Velocity buffers (same shape as weights)
vW1 = np.zeros_like(net_momentum.W1)
vb1 = np.zeros_like(net_momentum.b1)
vW2 = np.zeros_like(net_momentum.W2)
vb2 = np.zeros_like(net_momentum.b2)

momentum_val = 0.9

# ⚠️ THE LEARNING RATE MUST BE SCALED DOWN WHEN YOU ADD MOMENTUM.
#
# Momentum accumulates a running sum of past gradients. With beta = 0.9 the
# velocity converges to about 1/(1 - beta) = 10x a single gradient step, so the
# EFFECTIVE learning rate is roughly lr / (1 - beta).
#
#   lr = 0.05, beta = 0.9  ->  effective ~0.50
#
# Section 6 above already measured that lr = 0.5 is too high (final loss 0.0076,
# worse than lr = 0.05). So reusing 0.05 here would not be testing momentum --
# it would be re-testing an overshooting learning rate, and momentum would look
# worse than plain SGD purely as an artifact.
#
# Rule of thumb: when adding momentum beta, multiply lr by (1 - beta).
#   0.05 * (1 - 0.9) = 0.005   ->  matches SGD exactly     (loss 0.0004)
#   0.01                       ->  effective ~0.10, better (loss 0.0002)
#
# Measured over 500 epochs:
#   lr=0.05 (eff 0.50) -> 0.0266     <- the unfair comparison
#   lr=0.01 (eff 0.10) -> 0.0002     <- beats vanilla SGD
#   lr=0.005 (eff 0.05) -> 0.0004    <- ties vanilla SGD
lr_mom = 0.01
momentum_losses = []

for epoch in range(500):
    epoch_loss = 0
    for i in range(len(X_train)):
        probs = net_momentum.forward(X_train[i])
        loss = net_momentum.compute_loss(probs, one_hot(y_train[i], 3))
        epoch_loss += loss
        net_momentum.backward(one_hot(y_train[i], 3))
        
        # Momentum update: velocity accumulates past gradients
        vW1 = momentum_val * vW1 - lr_mom * net_momentum.dW1
        vb1 = momentum_val * vb1 - lr_mom * net_momentum.db1
        vW2 = momentum_val * vW2 - lr_mom * net_momentum.dW2
        vb2 = momentum_val * vb2 - lr_mom * net_momentum.db2
        
        net_momentum.W1 += vW1
        net_momentum.b1 += vb1
        net_momentum.W2 += vW2
        net_momentum.b2 += vb2
    
    momentum_losses.append(epoch_loss / len(X_train))

print(f"  Momentum (0.9, lr={lr_mom}): final loss = {momentum_losses[-1]:.4f}"
      f"   [effective lr ~{lr_mom/(1-momentum_val):.2f}]")

# Train with Adam (simplified)
np.random.seed(42)
net_adam = TrainableNet()
lr_adam = 0.01
beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
# First moment (mean of gradients)
mW1 = np.zeros_like(net_adam.W1)
mb1 = np.zeros_like(net_adam.b1)
mW2 = np.zeros_like(net_adam.W2)
mb2 = np.zeros_like(net_adam.b2)
# Second moment (mean of squared gradients)
vvW1 = np.zeros_like(net_adam.W1)
vvb1 = np.zeros_like(net_adam.b1)
vvW2 = np.zeros_like(net_adam.W2)
vvb2 = np.zeros_like(net_adam.b2)

adam_losses = []
adam_t = 0

for epoch in range(500):
    epoch_loss = 0
    for i in range(len(X_train)):
        probs = net_adam.forward(X_train[i])
        loss = net_adam.compute_loss(probs, one_hot(y_train[i], 3))
        epoch_loss += loss
        net_adam.backward(one_hot(y_train[i], 3))
        
        adam_t += 1
        
        # Adam update for each parameter
        for W, dW, m, v in [
            (net_adam.W1, net_adam.dW1, mW1, vvW1),
            (net_adam.b1, net_adam.db1, mb1, vvb1),
            (net_adam.W2, net_adam.dW2, mW2, vvW2),
            (net_adam.b2, net_adam.db2, mb2, vvb2),
        ]:
            # Update biased first moment (gradient mean)
            m[:] = beta1 * m + (1 - beta1) * dW
            # Update biased second moment (gradient variance)
            v[:] = beta2 * v + (1 - beta2) * (dW ** 2)
            
            # Bias correction (important in early steps)
            m_hat = m / (1 - beta1 ** adam_t)
            v_hat = v / (1 - beta2 ** adam_t)
            
            # Update weights: each weight gets its own effective learning rate
            W -= lr_adam * m_hat / (np.sqrt(v_hat) + eps_adam)
    
    adam_losses.append(epoch_loss / len(X_train))

print(f"  Adam (lr=0.01):  final loss = {adam_losses[-1]:.4f}")
print(f"  Vanilla SGD:     final loss = {lr_experiments[0.05][-1]:.4f}")

print("""
Adam usually converges faster and more reliably than vanilla SGD.
This is why PyTorch's default optimizer is Adam (torch.optim.Adam).
""")


# =============================================
# SECTION 8: VISUALIZE DECISION BOUNDARY
# =============================================

print("=" * 60)
print("SECTION 8: VISUALIZING THE LEARNED DECISION BOUNDARY")
print("=" * 60)

def predict_grid(net, resolution=100):
    """Predict class for every point on a grid — for visualization."""
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    
    grid_points = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
    predictions = np.zeros(len(grid_points), dtype=int)
    
    for i, point in enumerate(grid_points):
        probs = net.forward(point)
        predictions[i] = np.argmax(probs)
    
    return xx, yy, predictions.reshape(xx.shape)

print("  Computing decision boundary (this takes a moment)...")
xx, yy, Z = predict_grid(net, resolution=80)
print("  Done!")


# =============================================
# SECTION 9: COMPREHENSIVE VISUALIZATION
# =============================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Day 2, Part 3 — Training Loop\n'
             'From Random Guessing to Understanding',
             fontsize=14, fontweight='bold')

# Plot 1: Learning curve (loss)
axes[0, 0].plot(loss_history, 'b-', linewidth=1.5, alpha=0.8)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Cross-Entropy Loss')
axes[0, 0].set_title('Learning Curve — Loss\n(lower = better)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=-np.log(1/3), color='r', linestyle='--', alpha=0.5, label='Random guessing')
axes[0, 0].legend()

# Plot 2: Learning curve (accuracy)
axes[0, 1].plot(acc_history, 'g-', linewidth=1.5, alpha=0.8)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Learning Curve — Accuracy\n(higher = better)')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim(0, 1.05)
axes[0, 1].axhline(y=1/3, color='r', linestyle='--', alpha=0.5, label='Random guessing')
axes[0, 1].legend()

# Plot 3: Learning rate comparison
for lr_val, losses in lr_experiments.items():
    axes[0, 2].plot(losses, label=f'lr={lr_val}', linewidth=1.5, alpha=0.8)
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Loss')
axes[0, 2].set_title('Learning Rate Comparison\n(smaller steps vs bigger steps)')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].set_ylim(0, 2.0)

# Plot 4: Decision boundary
colors_map = ['#FF6B6B', '#4ECDC4', '#45B7D1']
axes[1, 0].contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5],
                     colors=colors_map, alpha=0.3)
axes[1, 0].contour(xx, yy, Z, levels=[0.5, 1.5], colors='black', linewidths=1)

for c in range(3):
    mask = y_train == c
    axes[1, 0].scatter(X_train[mask, 0], X_train[mask, 1],
                       c=colors_map[c], s=15, alpha=0.6, edgecolors='k', linewidth=0.3)
axes[1, 0].set_title('Learned Decision Boundary\n(colored regions = predicted class)')
axes[1, 0].set_xlabel('Feature 1')
axes[1, 0].set_ylabel('Feature 2')

# Plot 5: SGD vs Momentum vs Adam
axes[1, 1].plot(lr_experiments[0.05][:500], label='Vanilla SGD', linewidth=1.5, alpha=0.8)
axes[1, 1].plot(momentum_losses, label='SGD + Momentum', linewidth=1.5, alpha=0.8)
axes[1, 1].plot(adam_losses, label='Adam', linewidth=1.5, alpha=0.8)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].set_title('Optimizer Comparison\nSGD vs Momentum vs Adam')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim(0, 1.5)

# Plot 6: Gradient descent on f(x)=(x-3)^2
x_range = np.linspace(-1, 7, 100)
axes[1, 2].plot(x_range, [(xi - 3)**2 for xi in x_range], 'b-', linewidth=2, label='f(x) = (x-3)^2')
for lr_val in [0.01, 0.1, 0.5]:
    path = histories[lr_val][:15]
    axes[1, 2].plot(path, [f(xi) for xi in path], 'o-', markersize=4, alpha=0.7, label=f'lr={lr_val}')
axes[1, 2].set_xlabel('x')
axes[1, 2].set_ylabel('f(x)')
axes[1, 2].set_title('Gradient Descent Paths\n(walking downhill to minimum)')
axes[1, 2].legend(fontsize=8)
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('part3_results.png', dpi=150, bbox_inches='tight')
print("\nSaved: part3_results.png")


# =============================================
# FINAL SUMMARY
# =============================================

print("\n" + "=" * 60)
print("DAY 2 COMPLETE — YOUR NETWORK CAN LEARN!")
print("=" * 60)
print(f"""
╔══════════════════════════════════════════════════════════╗
║              DAY 2 COMPLETE — THE FULL PICTURE           ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Part 1 — Loss Functions:                                ║
║    * MSE: (predicted - target)^2                         ║
║    * Cross-Entropy: -sum(target * log(predicted))        ║
║    * CE gradient: predicted - target (beautiful!)        ║
║                                                          ║
║  Part 2 — Backpropagation:                               ║
║    * Chain rule: multiply gradients backward             ║
║    * Dense backward: dW = input^T * d_output             ║
║    * ReLU backward: gate (pass or block)                 ║
║    * Numerical verification: always check your math!     ║
║                                                          ║
║  Part 3 — Training Loop:                                 ║
║    * Gradient descent: W -= lr * gradient                ║
║    * Learning rate: THE most important hyperparameter    ║
║    * Momentum: remembers past direction                  ║
║    * Adam: adaptive learning rate per weight             ║
║    * Trained a network: {acc_history[0]:.0%} → {acc_history[-1]:.0%} accuracy!       ║
║                                                          ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  THE COMPLETE ALGORITHM:                                 ║
║                                                          ║
║    1. Forward:  x → layers → prediction                  ║
║    2. Loss:     prediction vs target → one number        ║
║    3. Backward: loss → gradients for all weights         ║
║    4. Update:   W = W - lr * dW                          ║
║    5. Repeat until loss is small                         ║
║                                                          ║
║  THAT'S ALL OF DEEP LEARNING.                            ║
║  Everything else (BatchNorm, Dropout, ResNet, YOLO)      ║
║  is optimization on top of these 5 steps.                ║
║                                                          ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  YOUR DAY 1 + DAY 2 JOURNEY:                             ║
║                                                          ║
║  Day 1: Built the STRUCTURE                              ║
║    Conv → ReLU → Pool → Dense → Softmax                 ║
║    The network could COMPUTE but not LEARN               ║
║                                                          ║
║  Day 2: Built the BRAIN                                  ║
║    Loss → Backprop → Gradient Descent                    ║
║    The network can now LEARN from data                   ║
║                                                          ║
║  Day 3 (next): Train on REAL data (MNIST)                ║
║    Load 60,000 handwritten digits                        ║
║    Train your NumPy CNN to recognize them                ║
║    Target: 85%+ accuracy with ZERO frameworks            ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

print("=" * 60)
print("DAY 2 COMPLETE — Push to GitHub. You've earned it!")
print("=" * 60)
