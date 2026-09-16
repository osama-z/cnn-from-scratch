# Changelog

## v0.1.0

- Train a NumPy CNN with explicit convolution/pooling derivatives and Adam.
- Export the same learned parameters to independent C and C++ inference engines.
- Verify gradients, layer outputs, parser rejection, and native memory behavior.
- Provide a frozen integer convolution fixture and tested VHDL building blocks.
- Organize lessons, shared engines, trained deployment, hardware, and documentation.
- Add full-test-set evaluation figures, an offline prediction gallery, and a
  checksummed model bundle with source revision and per-file provenance.

Recorded baseline: 91.87% accuracy on 10,000 MNIST test images. The exported
32-image native comparison agrees at all 224 checked layer tensors per engine.

The trained model uses float32 inference. Full trained-network integer RTL,
microcontroller deployment, and measured FPGA timing/resources are future work.
