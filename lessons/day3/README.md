# Lesson 3: train a dense MNIST classifier

[Previous: learning](../day2/README.md) · [All lessons](../README.md) ·
[Next: C](../day4/README.md)

Goal: connect a training loop to real labeled images and evaluate on held-out
data. This lesson uses a **dense network**, with no convolutional layers.

## Model and files

[part1_mnist_classifier.py](part1_mnist_classifier.py) loads MNIST, normalizes
pixels, trains the model, and plots its results.

```text
28×28 grayscale image → flatten to 784 → Dense 128 → Dense 64 → 10 classes
```

Read the activation and loss implementations in the script alongside this
shape outline. The historical run used 10,000 training images and 2,000 test
images and reached about 95% accuracy. It did not use all 60,000 training images.

## Run it

From the repository root, after installing
[lesson dependencies](../../docs/setup.md#1-install-the-tools):

```bash
python -m trained.download
(cd lessons/day3 && python part1_mnist_classifier.py)
```

The explicit downloader verifies and stores the MNIST archives in
`lessons/day3/mnist_data/`. The original lesson also has its own download helper;
pre-downloading lets it reuse the existing files.

Expect printed training progress, test accuracy, a
[sample sheet](part1_mnist_samples.png), and [results](part1_results.png).
The dataset directory is ignored by Git.

## Understand the preprocessing

A pixel byte of 128 becomes `128 / 255 ≈ 0.502`. Flattening turns 28 rows of
28 pixels into a vector of 784 values. All pixels then connect to the first
Dense layer; this differs from a convolution's local, shared weights.

A test prediction is correct only when its winning class equals the label.
Keep test images out of the training updates.

Checkpoint: why is the 95% result not evidence about convolution? Answer: this
model has no convolutional layer.

For the current CNN, validation-based checkpoint selection, and full 10,000-image
test evaluation, continue to [the trained pipeline](../../trained/README.md).
