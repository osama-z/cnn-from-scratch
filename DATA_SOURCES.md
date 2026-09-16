# Data sources and attribution

The digit examples used for training, evaluation figures, and the offline demo
come from MNIST, credited to Yann LeCun, Corinna Cortes, and Christopher J. C.
Burges. MNIST contains 60,000 training and 10,000 test images of handwritten digits.
See the [MNIST catalogue and citation](https://www.tensorflow.org/datasets/catalog/mnist)
and [original dataset page](http://yann.lecun.com/exdb/mnist/).

The explicit downloader retrieves the four IDX archives from the
[Open Source Computer Vision Infrastructure mirror](https://ossci-datasets.s3.amazonaws.com/mnist/)
and checks their SHA-256 hashes. The repository does not include the full archives.
Training splits and source hashes are recorded in the experiment metrics.

The model/demo release bundle includes a small selection of test examples,
normalized pixel arrays, and their reference outputs for reproducibility.
The prediction figure shows the first six correct and first six incorrect
examples in test-set order; these are illustrative examples, not a new test split.

The project's MIT license applies to its original code. This project does not
claim authorship of MNIST or relicense the source dataset under MIT.
