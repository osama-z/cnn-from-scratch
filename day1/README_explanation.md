# 👁️ Day 1: The Visual Cortex (Forward Pass)

> **Core Concept:** Teaching a computer to "see" by breaking images down into mathematical patterns, step-by-step. Data flows in one direction: Input → Output.

---

## 1. The Architecture (Visualized)

Here is exactly what the Day 1 code builds. Imagine a 28x28 image of a digit passing through these physical filters:

```text
[Input Image] (28x28 pixels) 
      │
      ▼
[Convolution] ───► Slides a 3x3 "flashlight" over the image to detect edges.
      │            (Outputs Feature Maps)
      ▼
[ReLU Gate]   ───► Math: f(x) = max(0, x)
      │            Blocks all negative numbers. Allows the network to learn non-linear shapes (curves, circles).
      ▼
[Max Pooling] ───► Math: Keeps only the largest number in a 2x2 grid.
      │            Shrinks the image size by half. Makes the network ignore exact pixel locations (Translation Invariance).
      ▼
[Flatten]     ───► Converts the 2D grids into a single 1D line of numbers.
      │
      ▼
[Dense Layer] ───► Fully connects every feature to a final set of "guesses".
      │            Math: (Inputs • Weights) + Bias
      ▼
[Softmax]     ───► Math: e^x / Σ(e^x)
                   Forces all raw scores to become percentages (e.g., 90% Class 1, 10% Class 2).
```

---

## 2. Deep Dive: How Convolution Actually Works

A computer doesn't see a "line". It sees numbers.
Convolution is the act of taking a small matrix (a **Kernel**) and multiplying it against the image.

**Example: A Vertical Edge Detector Kernel**
```text
[-1  0  1]      If you slide this over an image where the left side is dark (0) 
[-1  0  1]      and the right side is bright (255), the result is a huge positive number.
[-1  0  1]      The network "lights up" saying: "I FOUND A VERTICAL EDGE!"
```
In modern AI, *we do not hardcode these kernels*. We initialize them randomly, and the network **learns** what numbers to put inside them during Day 2.

---

## 3. How to Use It in Code

The Forward Pass is extremely clean. You wrap the math in a class so you can easily pass an image through it.

```python
# The Day 1 Workflow
cnn = TinyCNN()
image = load_image("cat.png")
prediction = cnn.forward(image)  # The image physically passes through all the math layers
print(prediction) # Output: [0.10, 0.85, 0.05] (85% sure it's class 1)
```

---

## 4. Where to Use It & Why

*   **The Phase:** Inference (Production)
*   **The Hardware:** Edge devices (Drones, iPhones, Raspberry Pi).
*   **The "Why":** The Forward Pass requires no learning and no historical memory. It is purely multiplying matrices and adding numbers. Because it is so lightweight, you can run it 60 times a second on a drone's camera feed to detect objects in real-time. 

When you deploy AI to the real world, you **strip away Day 2 and Day 3**, and you only deploy the code from Day 1.
