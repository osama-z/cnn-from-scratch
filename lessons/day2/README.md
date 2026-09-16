# Lesson 2: make weights learn

[Previous: forward pass](../day1/README.md) · [All lessons](../README.md) ·
[Next: MNIST](../day3/README.md)

Goal: understand the connection between a prediction, a loss, a derivative,
and a parameter update. Begin with Lesson 1 and basic derivatives.

## Read and run in order

| File | Main idea |
|---|---|
| [part1_loss_functions.py](part1_loss_functions.py) | MSE, cross-entropy, stable softmax, numerical gradients |
| [part2_backpropagation.py](part2_backpropagation.py) | Chain rule and explicit Dense/ReLU backward passes |
| [part3_training_loop.py](part3_training_loop.py) | Repeat updates and compare SGD, momentum, and Adam |

From the repository root:

```bash
(cd lessons/day2 && python part1_loss_functions.py)
(cd lessons/day2 && python part2_backpropagation.py)
(cd lessons/day2 && python part3_training_loop.py)
```

Expect numerical examples, gradient comparisons, and learning curves. Plots are
saved as [part 1](part1_results.png), [part 2](part2_results.png), and
[part 3](part3_results.png) in this folder.

## Work through one update

Suppose a weight is 0.5, its loss gradient is 2, and the learning rate is 0.01:

```text
new weight = 0.5 - 0.01 × 2 = 0.48
```

The positive derivative says that increasing this weight locally increases the
loss, so gradient descent moves it downward. The derivative does not promise a
large step will help.

Backpropagation computes these local derivatives by applying the chain rule
from the output toward the input. Adam uses moving averages to adjust the
updates; it still depends on correct gradients.

## Read the results carefully

The toy network can learn a small synthetic dataset very well. That establishes
that the example learns those patterns; it does not establish MNIST accuracy.

A low loss and a correct class are related but different. Cross-entropy can
improve even when the predicted class remains unchanged.

Checkpoint: if the gradient is -3 with the same learning rate, does the weight
increase or decrease? Answer: it increases by 0.03.

Continue with [MNIST](../day3/README.md). Explicit convolution backpropagation and
its regression tests are in [model.py](../../trained/model.py) and
[test_trained.py](../../tests/test_trained.py).
