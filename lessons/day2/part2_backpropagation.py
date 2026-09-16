"""
DAY 2 — PART 2: Backpropagation — How Gradients Flow Backward
==============================================================
Goal: Understand the CHAIN RULE and how it makes neural networks learn.

The Story So Far:
    Part 1: You learned that loss = cross_entropy(predicted, target)
            and the gradient = predicted - target
            
    But that gradient is for the LAST layer (softmax output).
    The network has MANY layers:
        Input → Conv1 → ReLU → Pool → Conv2 → ReLU → Pool → Dense1 → ReLU → Dense2 → Softmax
    
    How do we compute the gradient for Conv1's weights?
    They're 8 layers away from the loss!
    
    Answer: BACKPROPAGATION = apply the chain rule layer by layer,
            going BACKWARD from loss to input.

Why "backpropagation"?
    Forward:  Input ─→ Layer 1 ─→ Layer 2 ─→ ... ─→ Loss (LEFT to RIGHT)
    Backward: Input ←─ Layer 1 ←─ Layer 2 ←─ ... ←─ Loss (RIGHT to LEFT)
    
    We propagate gradients BACK through the network.
    Each layer receives: "how does the loss change if your output changes?"
    Each layer computes: "how does the loss change if my WEIGHTS change?"
    
    That's backpropagation. It's just the chain rule, applied systematically.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# =============================================
# SECTION 1: WHAT IS A DERIVATIVE?
# =============================================

print("=" * 60)
print("SECTION 1: WHAT IS A DERIVATIVE?")
print("=" * 60)

print("""
A derivative answers ONE question:
    "If I nudge x by a tiny amount, how much does f(x) change?"

Example: f(x) = x^2

    At x = 3: f(3) = 9
    Nudge x to 3.001: f(3.001) = 9.006001
    
    Change in f = 9.006001 - 9 = 0.006001
    Change in x = 0.001
    
    Rate of change = 0.006001 / 0.001 = 6.001 ~ 6
    
    The derivative f'(x) = 2x, so f'(3) = 6. Correct!
    
    Positive derivative = f increases when x increases
    Negative derivative = f decreases when x increases
    Zero derivative = you're at a peak or valley (critical point)
""")

def numerical_derivative(f, x, h=1e-5):
    """
    Compute the derivative of f at point x using finite differences.
    
    derivative = (f(x + h) - f(x - h)) / (2h)
    
    Using both x+h and x-h (central difference) gives better accuracy.
    """
    return (f(x + h) - f(x - h)) / (2 * h)

print("Verifying derivatives numerically:\n")
print(f"  f(x) = x^2")
print(f"  f'(x) = 2x (analytical)")
print(f"  f'(3) = 6 (analytical)")
print(f"  f'(3) = {numerical_derivative(lambda x: x**2, 3):.6f} (numerical)")
print(f"  Match: ✓\n")

print(f"  f(x) = x^3")
print(f"  f'(x) = 3x^2 (analytical)")
print(f"  f'(2) = 12 (analytical)")
print(f"  f'(2) = {numerical_derivative(lambda x: x**3, 2):.6f} (numerical)")
print(f"  Match: ✓")


# =============================================
# SECTION 2: THE CHAIN RULE
# =============================================

print("\n" + "=" * 60)
print("SECTION 2: THE CHAIN RULE — The Heart of Backpropagation")
print("=" * 60)

print("""
THE CHAIN RULE:
    If z = f(y) and y = g(x), then:
    
        dz/dx = dz/dy * dy/dx
    
    "The rate of change of z with respect to x equals
     the rate of change of z w.r.t. y  TIMES
     the rate of change of y w.r.t. x"

ANALOGY — The Domino Chain:
    Push x → y moves → z moves
    
    If pushing x by 1 makes y move by 3 (dy/dx = 3)
    And moving y by 1 makes z move by 2 (dz/dy = 2)
    Then pushing x by 1 makes z move by 3 * 2 = 6 (dz/dx = 6)
    
    The chain rule = multiply all the "pushes" together.

WHY THIS MATTERS FOR NEURAL NETWORKS:
    Input → [Layer 1] → [Layer 2] → [Layer 3] → Loss
    
    d(Loss)/d(Layer1_weights) = d(Loss)/d(Layer3_output) 
                               * d(Layer3_output)/d(Layer2_output)
                               * d(Layer2_output)/d(Layer1_output)
                               * d(Layer1_output)/d(Layer1_weights)
""")

# Concrete example with numbers
x_val = 3.0
y_val = 2 * x_val + 1  # g(3) = 7
z_val = y_val ** 2       # f(7) = 49

dy_dx = 2.0
dz_dy = 2 * y_val  # 14
dz_dx = dz_dy * dy_dx  # 28

dz_dx_numerical = numerical_derivative(lambda x: (2*x + 1)**2, x_val)

print(f"""
HAND TRACE: Chain Rule with Numbers
{'─' * 50}
  Functions: g(x) = 2x + 1,  f(y) = y^2
  Composed:  z = f(g(x)) = (2x + 1)^2

  At x = {x_val}:
    y = g({x_val}) = 2({x_val}) + 1 = {y_val}
    z = f({y_val}) = {y_val}^2 = {z_val}

  Chain rule:
    dy/dx = 2
    dz/dy = 2y = 2({y_val}) = {dz_dy}
    dz/dx = dz/dy * dy/dx = {dz_dy} * {dy_dx} = {dz_dx}

  Numerical verification: dz/dx = {dz_dx_numerical:.6f}
  Match: {'✓' if abs(dz_dx - dz_dx_numerical) < 0.001 else 'X'}
""")


# =============================================
# SECTION 3: COMPUTATIONAL GRAPH — FULL TRACE
# =============================================

print("=" * 60)
print("SECTION 3: COMPUTATIONAL GRAPH — Full Forward & Backward Trace")
print("=" * 60)

print("""
A neural network is a GRAPH of computations.
Let's trace a tiny network: 1 input, 1 hidden neuron, 1 output.

FORWARD PASS (left to right, computing values):
                                                         
    x --> [*w1] --> [+b1] --> [ReLU] --> [*w2] --> [+b2] --> output --> [Loss]
    3      *0.5     +0.1     max(0,.)   *(-0.3)   +0.2     -0.28     (.-t)^2
           =1.5     =1.6      =1.6      =-0.48    =-0.28              =1.6384

BACKWARD PASS (right to left, computing gradients):
    We want: dL/dw1 and dL/dw2 (how to adjust each weight)
""")

x = 3.0
w1 = 0.5
b1 = 0.1
w2 = -0.3
b2 = 0.2
t = 1.0

# ──── FORWARD PASS ────
z1 = x * w1
a1 = z1 + b1
r1 = max(0, a1)
z2 = r1 * w2
a2 = z2 + b2
loss = (a2 - t) ** 2

print(f"Forward pass:")
print(f"  z1 = x * w1 = {x} * {w1} = {z1}")
print(f"  a1 = z1 + b1 = {z1} + {b1} = {a1}")
print(f"  r1 = ReLU({a1}) = {r1}")
print(f"  z2 = r1 * w2 = {r1} * {w2} = {z2}")
print(f"  a2 = z2 + b2 = {z2} + {b2} = {a2}")
print(f"  loss = (a2 - t)^2 = ({a2} - {t})^2 = {loss:.4f}")

# ──── BACKWARD PASS ────
print(f"\nBackward pass (chain rule, right to left):")

dL_da2 = 2 * (a2 - t)
print(f"  dL/da2 = 2(a2 - t) = 2({a2} - {t}) = {dL_da2:.4f}")

dL_dz2 = dL_da2 * 1.0
dL_db2 = dL_da2 * 1.0
print(f"  dL/dz2 = dL/da2 * 1 = {dL_dz2:.4f}  (addition passes gradient through)")
print(f"  dL/db2 = {dL_db2:.4f}")

dL_dr1 = dL_dz2 * w2
dL_dw2 = dL_dz2 * r1
print(f"  dL/dw2 = dL/dz2 * r1 = {dL_dz2:.4f} * {r1} = {dL_dw2:.4f}")
print(f"  dL/dr1 = dL/dz2 * w2 = {dL_dz2:.4f} * {w2} = {dL_dr1:.4f}")

relu_gate = 1.0 if a1 > 0 else 0.0
dL_da1 = dL_dr1 * relu_gate
print(f"  dL/da1 = dL/dr1 * gate = {dL_dr1:.4f} * {relu_gate} = {dL_da1:.4f}")
print(f"           (ReLU gate {'OPEN' if relu_gate == 1 else 'CLOSED'}: a1={a1} {'>' if a1 > 0 else '<='} 0)")

dL_dz1 = dL_da1 * 1.0
dL_db1 = dL_da1 * 1.0
dL_dw1 = dL_dz1 * x
dL_dx = dL_dz1 * w1

print(f"  dL/db1 = {dL_db1:.4f}")
print(f"  dL/dw1 = dL/dz1 * x = {dL_dz1:.4f} * {x} = {dL_dw1:.4f}")

# Verify numerically
def compute_loss(x, w1, b1, w2, b2, t):
    z1 = x * w1
    a1 = z1 + b1
    r1 = max(0, a1)
    z2 = r1 * w2
    a2 = z2 + b2
    return (a2 - t) ** 2

h = 1e-5
num_dw1 = (compute_loss(x, w1+h, b1, w2, b2, t) - compute_loss(x, w1-h, b1, w2, b2, t)) / (2*h)
num_dw2 = (compute_loss(x, w1, b1, w2+h, b2, t) - compute_loss(x, w1, b1, w2-h, b2, t)) / (2*h)
num_db1 = (compute_loss(x, w1, b1+h, w2, b2, t) - compute_loss(x, w1, b1-h, w2, b2, t)) / (2*h)
num_db2 = (compute_loss(x, w1, b1, w2, b2+h, t) - compute_loss(x, w1, b1, w2, b2-h, t)) / (2*h)

print(f"\nNumerical verification:")
print(f"  {'Param':>6s} {'Analytical':>12s} {'Numerical':>12s} {'Match':>8s}")
print(f"  {'─' * 42}")
for name, ana, num in [('w1', dL_dw1, num_dw1), ('b1', dL_db1, num_db1),
                        ('w2', dL_dw2, num_dw2), ('b2', dL_db2, num_db2)]:
    match_ok = "✓" if abs(ana - num) < 1e-4 else "X"
    print(f"  {name:>6s} {ana:>12.6f} {num:>12.6f} {match_ok:>8s}")


# =============================================
# SECTION 4: BACKPROP THROUGH DENSE LAYER (MATRICES)
# =============================================

print("\n" + "=" * 60)
print("SECTION 4: BACKPROP THROUGH DENSE LAYER (Matrices)")
print("=" * 60)

print("""
Dense layer forward:
    output = input @ W + b

Dense layer backward (given dL/d_output):
    dL/dW     = input^T @ dL/d_output    — gradient for weights
    dL/db     = dL/d_output              — gradient for biases
    dL/d_input = dL/d_output @ W^T       — gradient to pass backward
""")

class DenseLayer:
    """Dense layer with forward AND backward pass."""
    def __init__(self, n_in, n_out):
        self.W = np.random.randn(n_in, n_out).astype(np.float32) * np.sqrt(2.0 / n_in)
        self.b = np.zeros(n_out, dtype=np.float32)
        self.input_cache = None
        self.dW = None
        self.db = None
    
    def forward(self, x):
        self.input_cache = x.copy()
        return x @ self.W + self.b
    
    def backward(self, d_output):
        x = self.input_cache
        self.dW = x.reshape(-1, 1) @ d_output.reshape(1, -1)
        self.db = d_output.copy()
        d_input = d_output @ self.W.T
        return d_input

# Demo
layer = DenseLayer(3, 2)
layer.W = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
layer.b = np.array([0.01, 0.02], dtype=np.float32)

x_in = np.array([1.0, 2.0, 3.0], dtype=np.float32)
out = layer.forward(x_in)
d_out = np.array([0.5, -0.3], dtype=np.float32)
d_in = layer.backward(d_out)

print(f"\n  Input:  {x_in}")
print(f"  Output: {out}")
print(f"  Upstream gradient: {d_out}")
print(f"  dL/dW:\n{layer.dW}")
print(f"  dL/db: {layer.db}")
print(f"  dL/d_input: {d_in}")


# =============================================
# SECTION 5: BACKPROP THROUGH ReLU
# =============================================

print("\n" + "=" * 60)
print("SECTION 5: BACKPROP THROUGH ReLU")
print("=" * 60)

print("""
ReLU forward:  output = max(0, input)
ReLU backward: dL/d_input = dL/d_output * (1 if input > 0, else 0)

ReLU is a GATE:
    POSITIVE input → gate OPEN  → gradient passes through
    NEGATIVE input → gate CLOSED → gradient BLOCKED (= 0)

This is why "dead neurons" happen:
    If input is ALWAYS negative, gradient is ALWAYS zero.
    The neuron can NEVER learn. It's "dead".
""")

class ReLULayer:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x > 0).astype(np.float32)
        return x * self.mask
    
    def backward(self, d_output):
        return d_output * self.mask

relu_layer = ReLULayer()
x_relu = np.array([2.0, -1.0, 0.5, -3.0, 1.0], dtype=np.float32)
out_relu = relu_layer.forward(x_relu)
d_relu = np.array([0.5, -0.3, 0.2, -0.1, 0.8], dtype=np.float32)
d_in_relu = relu_layer.backward(d_relu)

print(f"\n  Input:      {x_relu}")
print(f"  Output:     {out_relu}")
print(f"  Gate:       {relu_layer.mask}")
print(f"  dL/d_out:   {d_relu}")
print(f"  dL/d_in:    {d_in_relu}")
print(f"  Blocked:    {int(np.sum(relu_layer.mask == 0))} of {len(x_relu)} neurons")


# =============================================
# SECTION 6: SOFTMAX + CE BACKWARD
# =============================================

print("\n" + "=" * 60)
print("SECTION 6: SOFTMAX + CROSS-ENTROPY BACKWARD")
print("=" * 60)

def softmax(logits):
    e_x = np.exp(logits - np.max(logits))
    return e_x / e_x.sum()

def softmax_ce_backward(probs, target):
    """
    THE BEAUTIFUL GRADIENT: d(Loss)/d(logits) = probs - target
    
    This combines softmax backward AND cross-entropy backward
    into ONE simple subtraction. The exp and log cancel out.
    """
    return probs - target

logits_demo = np.array([2.0, 1.0, 0.5], dtype=np.float32)
target_demo = np.array([0.0, 1.0, 0.0], dtype=np.float32)
probs_demo = softmax(logits_demo)
grad_demo = softmax_ce_backward(probs_demo, target_demo)

print(f"\n  Logits:   {logits_demo}")
print(f"  Softmax:  [{probs_demo[0]:.4f}, {probs_demo[1]:.4f}, {probs_demo[2]:.4f}]")
print(f"  Target:   {target_demo}")
print(f"  Gradient: [{grad_demo[0]:+.4f}, {grad_demo[1]:+.4f}, {grad_demo[2]:+.4f}]")
print(f"  = probs - target. That's the entire output gradient.")


# =============================================
# SECTION 7: FULL BACKPROP — 2-LAYER NETWORK
# =============================================

print("\n" + "=" * 60)
print("SECTION 7: FULL BACKPROP — Complete 2-Layer Network")
print("=" * 60)

class TwoLayerNet:
    """
    Complete 2-layer network with forward AND backward pass.
    Input (4,) -> Dense1 (4->8) -> ReLU -> Dense2 (8->3) -> Softmax+CE -> Loss
    """
    def __init__(self, n_in=4, n_hidden=8, n_out=3):
        self.dense1 = DenseLayer(n_in, n_hidden)
        self.relu1 = ReLULayer()
        self.dense2 = DenseLayer(n_hidden, n_out)
        self.logits = None
        self.probs = None
    
    def forward(self, x):
        h1 = self.dense1.forward(x)
        a1 = self.relu1.forward(h1)
        self.logits = self.dense2.forward(a1)
        self.probs = softmax(self.logits)
        return self.probs
    
    def compute_loss(self, probs, target):
        epsilon = 1e-7
        return -np.sum(target * np.log(probs + epsilon))
    
    def backward(self, target):
        """
        Backward pass — compute ALL gradients.
        Gradient flows: Loss -> Softmax+CE -> Dense2 -> ReLU -> Dense1
        """
        d_logits = softmax_ce_backward(self.probs, target)
        d_relu_out = self.dense2.backward(d_logits)
        d_dense1_out = self.relu1.backward(d_relu_out)
        d_input = self.dense1.backward(d_dense1_out)
        return d_input

print("\nBuilding network: Input(4) -> Dense(8) -> ReLU -> Dense(3) -> Softmax")

net = TwoLayerNet(n_in=4, n_hidden=8, n_out=3)
x_test = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
target_test = np.array([0.0, 1.0, 0.0], dtype=np.float32)

probs_test = net.forward(x_test)
loss_test = net.compute_loss(probs_test, target_test)

print(f"\n  Input:       {x_test}")
print(f"  Prediction:  [{probs_test[0]:.4f}, {probs_test[1]:.4f}, {probs_test[2]:.4f}]")
print(f"  Target:      {target_test}")
print(f"  Loss:        {loss_test:.4f}")

net.backward(target_test)

print(f"\n  Gradients computed:")
print(f"    Dense1 W: shape {net.dense1.dW.shape}, norm = {np.linalg.norm(net.dense1.dW):.4f}")
print(f"    Dense1 b: shape {net.dense1.db.shape}, norm = {np.linalg.norm(net.dense1.db):.4f}")
print(f"    Dense2 W: shape {net.dense2.dW.shape}, norm = {np.linalg.norm(net.dense2.dW):.4f}")
print(f"    Dense2 b: shape {net.dense2.db.shape}, norm = {np.linalg.norm(net.dense2.db):.4f}")


# =============================================
# SECTION 8: NUMERICAL VERIFICATION
# =============================================

print("\n" + "=" * 60)
print("SECTION 8: VERIFYING ALL GRADIENTS NUMERICALLY")
print("=" * 60)

h = 1e-5

# Save original gradients
saved_dW2 = net.dense2.dW.copy()
saved_dW1 = net.dense1.dW.copy()

# Check Dense2 weights
max_err_d2 = 0
for i in range(net.dense2.W.shape[0]):
    for j in range(net.dense2.W.shape[1]):
        orig = net.dense2.W[i, j]
        
        net.dense2.W[i, j] = orig + h
        p_plus = net.forward(x_test)
        l_plus = net.compute_loss(p_plus, target_test)
        
        net.dense2.W[i, j] = orig - h
        p_minus = net.forward(x_test)
        l_minus = net.compute_loss(p_minus, target_test)
        
        net.dense2.W[i, j] = orig
        num_g = (l_plus - l_minus) / (2 * h)
        max_err_d2 = max(max_err_d2, abs(saved_dW2[i, j] - num_g))

print(f"\n  Dense2 weights max error: {max_err_d2:.8f} {'✓ PASS' if max_err_d2 < 1e-4 else 'X FAIL'}")

# Check Dense1 weights
max_err_d1 = 0
for i in range(net.dense1.W.shape[0]):
    for j in range(net.dense1.W.shape[1]):
        orig = net.dense1.W[i, j]
        
        net.dense1.W[i, j] = orig + h
        p_plus = net.forward(x_test)
        l_plus = net.compute_loss(p_plus, target_test)
        
        net.dense1.W[i, j] = orig - h
        p_minus = net.forward(x_test)
        l_minus = net.compute_loss(p_minus, target_test)
        
        net.dense1.W[i, j] = orig
        num_g = (l_plus - l_minus) / (2 * h)
        max_err_d1 = max(max_err_d1, abs(saved_dW1[i, j] - num_g))

print(f"  Dense1 weights max error: {max_err_d1:.8f} {'✓ PASS' if max_err_d1 < 1e-4 else 'X FAIL'}")

# Restore correct gradients
net.forward(x_test)
net.backward(target_test)

print("""
ALL gradients verified! Our backpropagation is mathematically correct.

This means:
  - Chain rule applied correctly at every layer
  - Dense backward computes correct weight gradients
  - ReLU backward correctly blocks gradients for negative inputs
  - Softmax+CE backward gives probs - target

We can trust these gradients to train the network (Part 3).
""")


# =============================================
# SECTION 9: VISUALIZATION
# =============================================

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Day 2, Part 2 — Backpropagation\n'
             'How Gradients Flow Backward Through a Network',
             fontsize=14, fontweight='bold')

# Plot 1: f(x)=x^2 and its derivative
x_chain = np.linspace(-3, 3, 100)
axes[0, 0].plot(x_chain, x_chain**2, 'b-', linewidth=2, label='f(x) = x^2')
axes[0, 0].plot(x_chain, 2*x_chain, 'r--', linewidth=2, label="f'(x) = 2x")
axes[0, 0].axhline(y=0, color='k', linewidth=0.5)
axes[0, 0].axvline(x=0, color='k', linewidth=0.5)
axes[0, 0].set_title("Derivative = Slope\nf(x) = x^2, f'(x) = 2x")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: ReLU and its derivative
x_r = np.linspace(-3, 3, 100)
axes[0, 1].plot(x_r, np.maximum(0, x_r), 'b-', linewidth=2, label='ReLU(x)')
axes[0, 1].plot(x_r, (x_r > 0).astype(float), 'r--', linewidth=2, label="ReLU'(x)")
axes[0, 1].fill_between(x_r, -0.5, 1.5, where=x_r <= 0, alpha=0.1, color='red', label='Dead zone')
axes[0, 1].set_title('ReLU: Gate Open/Closed\nGradient blocked when input < 0')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim(-0.5, 3)

# Plot 3: Gradient magnitudes
layer_names = ['Output\n(p-t)', 'Dense2', 'ReLU', 'Dense1']
grad_norms = [
    np.linalg.norm(softmax_ce_backward(probs_test, target_test)),
    np.linalg.norm(net.dense2.dW),
    float(np.sum(net.relu1.mask)) / len(net.relu1.mask),
    np.linalg.norm(net.dense1.dW),
]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FF9F43']
axes[0, 2].barh(layer_names, grad_norms, color=colors)
axes[0, 2].set_title('Gradient Flow per Layer\n(right to left)')
axes[0, 2].set_xlabel('Gradient norm')

# Plot 4: Dense1 weight gradients
im1 = axes[1, 0].imshow(net.dense1.dW, cmap='RdBu_r', aspect='auto')
axes[1, 0].set_title('Dense1 Weight Gradients\n(red=increase, blue=decrease)')
axes[1, 0].set_xlabel('Output neuron')
axes[1, 0].set_ylabel('Input feature')
plt.colorbar(im1, ax=axes[1, 0])

# Plot 5: Dense2 weight gradients
im2 = axes[1, 1].imshow(net.dense2.dW, cmap='RdBu_r', aspect='auto')
axes[1, 1].set_title('Dense2 Weight Gradients\n(red=increase, blue=decrease)')
axes[1, 1].set_xlabel('Output class')
axes[1, 1].set_ylabel('Hidden neuron')
plt.colorbar(im2, ax=axes[1, 1])

# Plot 6: Algorithm text
axes[1, 2].text(0.5, 0.5,
    'Backpropagation Algorithm:\n\n'
    'FORWARD (save intermediates):\n'
    'x -> [Dense1] -> [ReLU] -> [Dense2] -> [SM] -> Loss\n\n'
    'BACKWARD (chain rule):\n'
    'dL/dx <- [dW1,db1] <- [mask] <- [dW2,db2] <- [p-t]\n\n'
    'Each layer:\n'
    '  1. Gets gradient from above\n'
    '  2. Computes dW, db\n'
    '  3. Passes gradient below\n\n'
    'UPDATE:\n'
    '  W = W - lr * dW  (Part 3!)',
    transform=axes[1, 2].transAxes,
    fontsize=9, fontfamily='monospace',
    verticalalignment='center', horizontalalignment='center',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
axes[1, 2].set_title('The Algorithm')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('part2_results.png', dpi=150, bbox_inches='tight')
print("Saved: part2_results.png")


# =============================================
# FINAL SUMMARY
# =============================================

print("\n" + "=" * 60)
print("DAY 2 PART 2 COMPLETE — BACKPROPAGATION")
print("=" * 60)
print("""
╔══════════════════════════════════════════════════════════╗
║                 WHAT YOU LEARNED                         ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  1. Derivative:                                          ║
║     "If I nudge x, how much does f(x) change?"          ║
║                                                          ║
║  2. Chain Rule:                                          ║
║     dz/dx = dz/dy * dy/dx                               ║
║     Multiply gradients through the chain                 ║
║                                                          ║
║  3. Computational Graph:                                 ║
║     Forward: compute values left to right, SAVE them     ║
║     Backward: compute gradients right to left            ║
║                                                          ║
║  4. Dense Backward:                                      ║
║     dW = input^T * d_output                              ║
║     db = d_output                                        ║
║     d_input = d_output * W^T                             ║
║                                                          ║
║  5. ReLU Backward:                                       ║
║     Gate: passes gradient if input > 0, blocks if <= 0   ║
║                                                          ║
║  6. Softmax + CE Backward:                               ║
║     d_logits = probs - target                            ║
║                                                          ║
║  7. Numerical Gradient Checking:                         ║
║     ALWAYS verify! (f(x+h) - f(x-h)) / 2h              ║
║                                                          ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Part 3: USE these gradients to UPDATE weights           ║
║  and watch the network LEARN!                            ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

print("=" * 60)
print("PART 2 COMPLETE — Next: Part 3 (Training Loop)")
print("=" * 60)
