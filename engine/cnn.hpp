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
 * WHAT ABOUT lessons/day6/?
 * -----------------
 * lessons/day6/cnn_forward.cpp and lessons/day6/templated_cnn.cpp deliberately spell the
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
#include <limits>
#include <stdexcept>

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
    static size_t checked_size(int c, int h, int w) {
        if (c <= 0 || h <= 0 || w <= 0)
            throw std::invalid_argument("tensor dimensions must be positive");
        size_t n = static_cast<size_t>(c);
        for (int d : {h, w}) {
            if (n > static_cast<size_t>(std::numeric_limits<int>::max()) / d)
                throw std::invalid_argument("tensor is too large");
            n *= d;
        }
        return n;
    }
public:
    int channels = 0, height = 0, width = 0;
    std::vector<float> data;

    Tensor() = default;

    Tensor(int c, int h, int w)
        : channels(c), height(h), width(w),
          data(checked_size(c, h, w), 0.0f) {}

    explicit Tensor(int n)
        : Tensor(1, 1, n) {}

    void validate() const {
        if (data.size() != checked_size(channels, height, width))
            throw std::invalid_argument("tensor storage does not match its shape");
    }

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
          weights(w), biases(b) {
        if (in_ch <= 0 || out_ch <= 0 || k <= 0 || k % 2 == 0 || !w || !b)
            throw std::invalid_argument("convolution needs positive channels, odd kernel, and weights");
        // Bound products used by the int-based weight indexing below.
        size_t count = static_cast<size_t>(out_ch);
        for (int d : {in_ch, k, k}) {
            if (count > static_cast<size_t>(std::numeric_limits<int>::max()) / d)
                throw std::invalid_argument("convolution weights are too large");
            count *= d;
        }
    }

    Tensor forward(const Tensor& in) override {
        in.validate();
        if (in.channels != in_ch)
            throw std::invalid_argument("convolution input channel mismatch");
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
        in.validate();
        Tensor out = in;
        for (int i = 0; i < out.size(); ++i)
            if (out[i] < 0.0f) out[i] = 0.0f;
        return out;
    }
};

class MaxPool2D : public Layer {
public:
    int pool;
    explicit MaxPool2D(int p) : Layer("pool"), pool(p) {
        if (p <= 0) throw std::invalid_argument("pool size must be positive");
    }
    Tensor forward(const Tensor& in) override {
        in.validate();
        if (in.height < pool || in.width < pool)
            throw std::invalid_argument("pool exceeds input dimensions");
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
        in.validate();
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
        : Layer("dense"), in_size(in_n), out_size(out_n), weights(w), biases(b) {
        if (in_n <= 0 || out_n <= 0 || in_n > std::numeric_limits<int>::max() / out_n || !w || !b)
            throw std::invalid_argument("invalid dense dimensions or weights");
    }

    Tensor forward(const Tensor& in) override {
        in.validate();
        if (in.size() != in_size)
            throw std::invalid_argument("dense input size mismatch");
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
        in.validate();
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

    void add(std::unique_ptr<Layer> l) {
        if (!l) throw std::invalid_argument("null layer");
        layers.push_back(std::move(l));
    }

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
