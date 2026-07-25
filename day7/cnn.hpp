/**
 * cnn.hpp — THE REUSABLE FLOAT INFERENCE ENGINE
 * ==============================================
 *
 * WHY THIS FILE EXISTS
 * --------------------
 * The layer classes were originally copy-pasted into golden_cpp.cpp. That is
 * fine once and a liability twice: fix a bug in Conv2D here and the other copy
 * silently keeps the bug, which is exactly the class of divergence the golden
 * model was built to catch. So the classes live in one header and everything
 * includes it.
 *
 * WHAT ABOUT day6/?
 * -----------------
 * day6/cnn_forward.cpp and day6/templated_cnn.cpp deliberately spell the
 * classes out in full. They are TEACHING artifacts — you read them top to
 * bottom to learn how classes, virtual dispatch and templates work, and the
 * whole point of Day 6 is watching the same network re-expressed. Making them
 * include a header would delete the lesson.
 *
 * This header is the opposite: it is the ENGINE. Written once, used by Day 7's
 * verification, and by the ARM/NEON build in Phase 2 and the benchmark in
 * Phase 3. Teaching code is meant to be read; engine code is meant to be
 * reused. Do not merge the two roles.
 *
 * DESIGN NOTE: BORROWED WEIGHTS
 * -----------------------------
 * Layers hold `const float*` pointing INTO the Model's blob. They do not own
 * or copy the weights. That keeps loading zero-copy (see model_io.h), but it
 * means the Model MUST outlive the network. This is a deliberate trade: on a
 * microcontroller you cannot afford a second copy of the weights, so the
 * lifetime discipline is the price of fitting in SRAM.
 */

#ifndef CNN_HPP
#define CNN_HPP

#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <cmath>
#include <utility>

namespace cnn {

// =====================================================================
// Tensor — a 3D buffer with the Day 4 index formula
// =====================================================================
//
//   C   (Day 4):  data[c * H * W + y * W + x]
//   C++ (here):   t(c, y, x)
//
// Same arithmetic; written once so an off-by-one has exactly one place to hide.

class Tensor {
public:
    int channels = 0, height = 0, width = 0;
    std::vector<float> data;

    Tensor() = default;

    Tensor(int c, int h, int w)
        : channels(c), height(h), width(w),
          data(static_cast<size_t>(c) * h * w, 0.0f) {}

    explicit Tensor(int n)
        : channels(1), height(1), width(n), data(n, 0.0f) {}

    int size() const { return static_cast<int>(data.size()); }

    float& operator()(int c, int y, int x) {
        return data[c * (height * width) + y * width + x];
    }
    const float& operator()(int c, int y, int x) const {
        return data[c * (height * width) + y * width + x];
    }

    float&       operator[](int i)       { return data[i]; }
    const float& operator[](int i) const { return data[i]; }
};


// =====================================================================
// Layer — the abstract interface
// =====================================================================
//
// `= 0` makes forward() pure virtual: any layer that forgets to implement it
// fails to COMPILE rather than failing at run time.
//
// The virtual destructor is not optional. Deleting a derived object through a
// Layer* with a non-virtual destructor is undefined behaviour — in practice the
// derived part never runs its destructor and you leak.

class Layer {
public:
    std::string name;
    explicit Layer(std::string n) : name(std::move(n)) {}
    virtual Tensor forward(const Tensor& input) = 0;
    virtual ~Layer() = default;
};


// =====================================================================
// Layers
// =====================================================================

class Conv2D : public Layer {
public:
    int in_ch, out_ch, ksize;
    const float* weights;   // borrowed — see the lifetime note at the top
    const float* biases;

    Conv2D(int in_channels, int out_channels, int k,
           const float* w, const float* b)
        : Layer("conv"), in_ch(in_channels), out_ch(out_channels), ksize(k),
          weights(w), biases(b) {}

    Tensor forward(const Tensor& in) override {
        const int H = in.height, W = in.width, kh = ksize / 2;
        Tensor out(out_ch, H, W);

        for (int oc = 0; oc < out_ch; ++oc)
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    float sum = biases[oc];
                    for (int ic = 0; ic < in_ch; ++ic)
                        for (int ky = -kh; ky <= kh; ++ky)
                            for (int kx = -kh; kx <= kh; ++kx) {
                                const int iy = y + ky, ix = x + kx;
                                if (iy < 0 || iy >= H || ix < 0 || ix >= W)
                                    continue;                 // zero padding
                                const int wi = oc * (in_ch * ksize * ksize)
                                             + ic * (ksize * ksize)
                                             + (ky + kh) * ksize + (kx + kh);
                                sum += in(ic, iy, ix) * weights[wi];
                            }
                    out(oc, y, x) = sum;
                }
        return out;
    }
};

class ReLU : public Layer {
public:
    ReLU() : Layer("relu") {}
    Tensor forward(const Tensor& in) override {
        Tensor out = in;
        for (int i = 0; i < out.size(); ++i)
            if (out[i] < 0.0f) out[i] = 0.0f;
        return out;
    }
};

class MaxPool2D : public Layer {
public:
    int pool;
    explicit MaxPool2D(int p) : Layer("pool"), pool(p) {}
    Tensor forward(const Tensor& in) override {
        const int oh = in.height / pool, ow = in.width / pool;
        Tensor out(in.channels, oh, ow);
        for (int c = 0; c < in.channels; ++c)
            for (int y = 0; y < oh; ++y)
                for (int x = 0; x < ow; ++x) {
                    float best = in(c, y * pool, x * pool);
                    for (int py = 0; py < pool; ++py)
                        for (int px = 0; px < pool; ++px)
                            best = std::max(best, in(c, y * pool + py, x * pool + px));
                    out(c, y, x) = best;
                }
        return out;
    }
};

class Flatten : public Layer {
public:
    Flatten() : Layer("flatten") {}
    Tensor forward(const Tensor& in) override {
        Tensor out(in.size());
        out.data = in.data;       // pure reshape, no arithmetic
        return out;
    }
};

class Dense : public Layer {
public:
    int in_size, out_size;
    const float* weights;         // layout (in_size, out_size): w[i * out_size + j]
    const float* biases;

    Dense(int in_n, int out_n, const float* w, const float* b)
        : Layer("dense"), in_size(in_n), out_size(out_n), weights(w), biases(b) {}

    Tensor forward(const Tensor& in) override {
        Tensor out(out_size);
        for (int j = 0; j < out_size; ++j) {
            float sum = biases[j];
            for (int i = 0; i < in_size; ++i)
                sum += in[i] * weights[i * out_size + j];
            out[j] = sum;
        }
        return out;
    }
};

class Softmax : public Layer {
public:
    Softmax() : Layer("softmax") {}
    Tensor forward(const Tensor& in) override {
        Tensor out = in;
        // Subtract the max before exp() — the Day 1 stability trick.
        // Without it, exp(large) overflows to inf and the result is NaN.
        const float mx = *std::max_element(out.data.begin(), out.data.end());
        float sum = 0.0f;
        for (int i = 0; i < out.size(); ++i) { out[i] = std::exp(out[i] - mx); sum += out[i]; }
        for (int i = 0; i < out.size(); ++i) out[i] /= sum;
        return out;
    }
};


// =====================================================================
// Network — owns the layers, borrows the weights
// =====================================================================
//
// unique_ptr owns each Layer and frees it automatically (RAII). The weight
// pointers inside those layers are borrowed from the Model, which is owned
// elsewhere and must outlive this object.

class Network {
public:
    std::vector<std::unique_ptr<Layer>> layers;

    void add(std::unique_ptr<Layer> l) { layers.push_back(std::move(l)); }

    /** Run the whole network. */
    Tensor forward(const Tensor& input) const {
        Tensor x = input;
        for (const auto& l : layers) x = l->forward(x);
        return x;
    }

    /**
     * Run the network, invoking `fn(layer_name, tensor)` after every layer.
     *
     * Per-layer visibility is what makes golden-model debugging tractable: if
     * two implementations disagree only at the end, the bug could be in any
     * layer, but the FIRST layer that diverges pins it down. Same reason you
     * probe intermediate signals in an RTL testbench rather than only the
     * output port.
     */
    template <typename Fn>
    Tensor forward_traced(const Tensor& input, Fn&& fn) const {
        Tensor x = input;
        for (const auto& l : layers) {
            x = l->forward(x);
            fn(l->name, x);
        }
        return x;
    }
};

} // namespace cnn

#endif // CNN_HPP
