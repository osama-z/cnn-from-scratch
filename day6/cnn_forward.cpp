/**
 * DAY 6: CNN FORWARD PASS IN C++
 * ===============================
 *
 * WHY C++ AND NOT JUST C?
 * ───────────────────────
 * Every major AI inference engine is written in C++:
 *   - TensorFlow Lite  → C++
 *   - ONNX Runtime     → C++
 *   - PyTorch LibTorch → C++
 *   - TensorRT         → C++
 *   - ncnn             → C++
 *
 * If you want to READ their source code, DEBUG them, or EXTEND them,
 * you MUST understand C++ classes, smart pointers, and templates.
 *
 * WHAT'S DIFFERENT FROM DAY 4 (C)?
 * ─────────────────────────────────
 * Day 4 (C):
 *   float* data = (float*)malloc(size * sizeof(float));
 *   conv2d(data, ...);   // pass raw pointers around
 *   free(data);           // forget this = MEMORY LEAK = drone crash
 *
 * Day 6 (C++):
 *   auto layer = std::make_unique<Conv2D>(1, 2, 3);
 *   auto output = layer->forward(input);   // object-oriented
 *   // No free() needed! Smart pointer auto-cleans when it goes out of scope.
 *   // This is called RAII: Resource Acquisition Is Initialization.
 *
 * SECTIONS:
 *   1. Tensor class (replaces raw float* pointers)
 *   2. Layer base class (abstract interface)
 *   3. Conv2D layer
 *   4. ReLU layer
 *   5. MaxPool2D layer
 *   6. Flatten layer
 *   7. Dense layer
 *   8. Softmax layer
 *   9. CNN class (chains layers together)
 *  10. Main: run inference and compare with C version
 */

#include <iostream>
#include <vector>
#include <memory>      // std::unique_ptr, std::make_unique
#include <cmath>       // expf, fabsf
#include <cassert>     // assert()
#include <string>
#include <chrono>      // high-resolution timing
#include <algorithm>   // std::max, std::fill
#include <numeric>     // std::accumulate
#include <cstdlib>     // srand, rand
#include <iomanip>     // std::setw, std::fixed

// =============================================
// SECTION 1: TENSOR CLASS
// =============================================

/**
 * A Tensor is a multi-dimensional array of floats.
 *
 * In C (Day 4), you used:
 *     float* data = (float*)malloc(channels * H * W * sizeof(float));
 *     int index = c * H * W + y * W + x;
 *     data[index] = 42.0f;
 *     free(data);  // DON'T FORGET!
 *
 * In C++ (Day 6), you use:
 *     Tensor t(channels, H, W);
 *     t(c, y, x) = 42.0f;
 *     // No free() needed — destructor handles it automatically (RAII)
 *
 * The key C++ concepts here:
 *   - std::vector<float> instead of malloc/free
 *   - operator() overloading for clean indexing
 *   - Copy constructor & move semantics
 */
class Tensor {
public:
    int channels, height, width;
    std::vector<float> data;  // std::vector manages memory automatically!

    // Constructor: creates a tensor filled with zeros
    Tensor(int c, int h, int w)
        : channels(c), height(h), width(w), data(c * h * w, 0.0f) {}

    // Constructor for 1D tensors (like Dense layer output)
    explicit Tensor(int size)
        : channels(1), height(1), width(size), data(size, 0.0f) {}

    // Default constructor (empty tensor)
    Tensor() : channels(0), height(0), width(0) {}

    // Total number of elements
    int size() const { return static_cast<int>(data.size()); }

    // Access element at (c, y, x) — same formula as Day 4!
    // operator() lets you write: tensor(c, y, x) instead of tensor.at(c, y, x)
    float& operator()(int c, int y, int x) {
        return data[c * (height * width) + y * width + x];
    }

    const float& operator()(int c, int y, int x) const {
        return data[c * (height * width) + y * width + x];
    }

    // Access element by flat index (for 1D tensors)
    float& operator[](int i) { return data[i]; }
    const float& operator[](int i) const { return data[i]; }

    // Print the tensor nicely
    void print(const std::string& name) const {
        std::cout << "\n" << name << " (shape: " << channels << "x"
                  << height << "x" << width << "):\n";
        for (int c = 0; c < channels; c++) {
            if (channels > 1) std::cout << "  Channel " << c << ":\n";
            for (int y = 0; y < height; y++) {
                std::cout << "    ";
                for (int x = 0; x < width; x++) {
                    std::cout << std::setw(7) << std::fixed
                              << std::setprecision(2) << (*this)(c, y, x);
                }
                std::cout << "\n";
            }
        }
    }
};


// =============================================
// SECTION 2: LAYER BASE CLASS (Abstract Interface)
// =============================================

/**
 * This is an ABSTRACT BASE CLASS.
 *
 * In C (Day 4), you had separate functions:
 *     void conv2d(float* input, ..., float* output);
 *     void relu(float* data, int size);
 *     void maxpool2d(float* input, ..., float* output);
 *
 * In C++ (Day 6), every layer is an OBJECT with a common interface:
 *     class Conv2D : public Layer { Tensor forward(const Tensor& input); }
 *     class ReLU   : public Layer { Tensor forward(const Tensor& input); }
 *
 * This lets you chain them in a loop:
 *     for (auto& layer : layers) { x = layer->forward(x); }
 *
 * The "= 0" makes forward() a PURE VIRTUAL function.
 * This means: "Any class that inherits from Layer MUST implement forward()."
 * If you forget, the compiler will give you an error. Safety!
 */
class Layer {
public:
    std::string name;

    Layer(const std::string& name) : name(name) {}

    // Pure virtual function — every layer MUST implement this
    virtual Tensor forward(const Tensor& input) = 0;

    // Virtual destructor — required for proper cleanup of derived classes
    virtual ~Layer() = default;
};


// =============================================
// SECTION 3: CONV2D LAYER
// =============================================

class Conv2D : public Layer {
public:
    int in_channels, out_channels, kernel_size;
    std::vector<float> weights;  // shape: (out_ch, in_ch, k, k)
    std::vector<float> biases;   // shape: (out_ch,)

    Conv2D(int in_ch, int out_ch, int k_size)
        : Layer("Conv2D(" + std::to_string(in_ch) + "→" + std::to_string(out_ch) + ")"),
          in_channels(in_ch), out_channels(out_ch), kernel_size(k_size),
          weights(out_ch * in_ch * k_size * k_size, 0.0f),
          biases(out_ch, 0.0f)
    {
        // Initialize weights randomly (Xavier-like)
        srand(42);
        float scale = 1.0f / std::sqrt(static_cast<float>(in_ch * k_size * k_size));
        for (auto& w : weights) {
            w = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f * scale;
        }
    }

    // Set specific kernel values (for testing — match Day 4 results)
    void set_kernel(int oc, int ic, const std::vector<float>& values) {
        assert(static_cast<int>(values.size()) == kernel_size * kernel_size);
        for (int i = 0; i < kernel_size * kernel_size; i++) {
            int idx = oc * (in_channels * kernel_size * kernel_size)
                    + ic * (kernel_size * kernel_size) + i;
            weights[idx] = values[i];
        }
    }

    Tensor forward(const Tensor& input) override {
        int H = input.height, W = input.width;
        int k_half = kernel_size / 2;
        Tensor output(out_channels, H, W);

        for (int oc = 0; oc < out_channels; oc++) {
            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    float sum = biases[oc];

                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int ky = -k_half; ky <= k_half; ky++) {
                            for (int kx = -k_half; kx <= k_half; kx++) {
                                int iy = y + ky, ix = x + kx;
                                if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                                    int w_idx = oc * (in_channels * kernel_size * kernel_size)
                                              + ic * (kernel_size * kernel_size)
                                              + (ky + k_half) * kernel_size
                                              + (kx + k_half);
                                    sum += input(ic, iy, ix) * weights[w_idx];
                                }
                            }
                        }
                    }
                    output(oc, y, x) = sum;
                }
            }
        }
        return output;
    }
};


// =============================================
// SECTION 4: RELU LAYER
// =============================================

class ReLU : public Layer {
public:
    ReLU() : Layer("ReLU") {}

    Tensor forward(const Tensor& input) override {
        Tensor output = input;  // Copy (in C you'd need memcpy + malloc)
        for (int i = 0; i < output.size(); i++) {
            if (output[i] < 0.0f) output[i] = 0.0f;
        }
        return output;
    }
};


// =============================================
// SECTION 5: MAXPOOL2D LAYER
// =============================================

class MaxPool2D : public Layer {
public:
    int pool_size;

    MaxPool2D(int size = 2) : Layer("MaxPool2D(" + std::to_string(size) + ")"), pool_size(size) {}

    Tensor forward(const Tensor& input) override {
        int out_H = input.height / pool_size;
        int out_W = input.width / pool_size;
        Tensor output(input.channels, out_H, out_W);

        for (int c = 0; c < input.channels; c++) {
            for (int y = 0; y < out_H; y++) {
                for (int x = 0; x < out_W; x++) {
                    float max_val = -1e30f;
                    for (int py = 0; py < pool_size; py++) {
                        for (int px = 0; px < pool_size; px++) {
                            float val = input(c, y * pool_size + py, x * pool_size + px);
                            if (val > max_val) max_val = val;
                        }
                    }
                    output(c, y, x) = max_val;
                }
            }
        }
        return output;
    }
};


// =============================================
// SECTION 6: FLATTEN LAYER
// =============================================

class Flatten : public Layer {
public:
    Flatten() : Layer("Flatten") {}

    Tensor forward(const Tensor& input) override {
        Tensor output(input.size());
        output.data = input.data;  // Just reshape — data stays the same
        return output;
    }
};


// =============================================
// SECTION 7: DENSE LAYER
// =============================================

class Dense : public Layer {
public:
    int in_size, out_size;
    std::vector<float> weights;  // shape: (in_size, out_size)
    std::vector<float> biases;   // shape: (out_size,)

    Dense(int in, int out)
        : Layer("Dense(" + std::to_string(in) + "→" + std::to_string(out) + ")"),
          in_size(in), out_size(out),
          weights(in * out, 0.0f),
          biases(out, 0.0f)
    {
        srand(42);
        float scale = 1.0f / std::sqrt(static_cast<float>(in));
        for (auto& w : weights) {
            w = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f * scale;
        }
    }

    Tensor forward(const Tensor& input) override {
        Tensor output(out_size);
        for (int j = 0; j < out_size; j++) {
            float sum = biases[j];
            for (int i = 0; i < in_size; i++) {
                sum += input[i] * weights[i * out_size + j];
            }
            output[j] = sum;
        }
        return output;
    }
};


// =============================================
// SECTION 8: SOFTMAX LAYER
// =============================================

class Softmax : public Layer {
public:
    Softmax() : Layer("Softmax") {}

    Tensor forward(const Tensor& input) override {
        Tensor output = input;

        // Numerical stability: subtract max
        float max_val = *std::max_element(output.data.begin(), output.data.end());
        float sum = 0.0f;
        for (int i = 0; i < output.size(); i++) {
            output[i] = std::exp(output[i] - max_val);
            sum += output[i];
        }
        for (int i = 0; i < output.size(); i++) {
            output[i] /= sum;
        }
        return output;
    }
};


// =============================================
// SECTION 9: CNN CLASS (Chains layers together)
// =============================================

/**
 * This is the power of C++ object-oriented design.
 *
 * In C (Day 4), you called functions one by one:
 *     conv2d(input, ..., conv_out);
 *     relu(conv_out, ...);
 *     maxpool2d(conv_out, ..., pool_out);
 *     dense(pool_out, ..., dense_out);
 *     softmax_inplace(dense_out, ...);
 *
 * In C++ (Day 6), you add layers to a list and loop:
 *     cnn.add(std::make_unique<Conv2D>(1, 2, 3));
 *     cnn.add(std::make_unique<ReLU>());
 *     cnn.add(std::make_unique<MaxPool2D>(2));
 *     Tensor output = cnn.forward(input);
 *
 * std::unique_ptr ensures each layer is automatically freed
 * when the CNN object goes out of scope. NO MEMORY LEAKS.
 * This is RAII: the constructor acquires, the destructor releases.
 */
class CNN {
public:
    // std::unique_ptr = smart pointer that auto-frees when destroyed
    // This replaces malloc/free entirely. RAII in action!
    std::vector<std::unique_ptr<Layer>> layers;

    void add(std::unique_ptr<Layer> layer) {
        layers.push_back(std::move(layer));
    }

    Tensor forward(const Tensor& input, bool verbose = false) {
        Tensor x = input;
        for (auto& layer : layers) {
            x = layer->forward(x);
            if (verbose) {
                std::cout << "  " << std::setw(25) << std::left << layer->name
                          << " → output shape: " << x.channels << "x"
                          << x.height << "x" << x.width
                          << " (" << x.size() << " values)\n";
            }
        }
        return x;
    }

    int total_params() const {
        int total = 0;
        for (const auto& layer : layers) {
            if (auto* conv = dynamic_cast<Conv2D*>(layer.get())) {
                total += conv->weights.size() + conv->biases.size();
            }
            if (auto* dense = dynamic_cast<Dense*>(layer.get())) {
                total += dense->weights.size() + dense->biases.size();
            }
        }
        return total;
    }
};


// =============================================
// SECTION 10: MAIN — BUILD AND RUN THE CNN
// =============================================

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║   DAY 6: CNN FORWARD PASS IN C++                        ║\n";
    std::cout << "║   Classes • Smart Pointers • RAII • No Memory Leaks     ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";

    // ─── Build the CNN ───
    CNN cnn;
    cnn.add(std::make_unique<Conv2D>(1, 2, 3));    // 1→2 channels, 3x3 kernel
    cnn.add(std::make_unique<ReLU>());
    cnn.add(std::make_unique<MaxPool2D>(2));         // 2x2 pooling
    cnn.add(std::make_unique<Flatten>());
    cnn.add(std::make_unique<Dense>(2 * 4 * 4, 3)); // 32→3 classes
    cnn.add(std::make_unique<Softmax>());

    // Set known kernels (match Day 4 for comparison)
    auto* conv = dynamic_cast<Conv2D*>(cnn.layers[0].get());
    conv->set_kernel(0, 0, {-1,-1,-1, 0,0,0, 1,1,1});  // Horizontal
    conv->set_kernel(1, 0, {-1,0,1, -1,0,1, -1,0,1});  // Vertical

    std::cout << "── Architecture ──\n";
    std::cout << "  Total parameters: " << cnn.total_params() << "\n";
    std::cout << "  Memory (float32): " << cnn.total_params() * 4 << " bytes\n\n";

    // ─── Create test images ───
    const char* class_names[] = {"Horizontal Edge", "Vertical Edge", "Uniform"};

    // Image 0: Horizontal edge
    Tensor img0(1, 8, 8);
    for (int y = 4; y < 8; y++)
        for (int x = 0; x < 8; x++)
            img0(0, y, x) = 10.0f;

    // Image 1: Vertical edge
    Tensor img1(1, 8, 8);
    for (int y = 0; y < 8; y++)
        for (int x = 4; x < 8; x++)
            img1(0, y, x) = 10.0f;

    // Image 2: Uniform
    Tensor img2(1, 8, 8);
    for (int i = 0; i < img2.size(); i++) img2[i] = 5.0f;

    std::vector<Tensor*> images = {&img0, &img1, &img2};

    // ─── Run inference ───
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "  RUNNING INFERENCE (verbose on first image)\n";
    std::cout << "══════════════════════════════════════════════════════════\n\n";

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 3; i++) {
        std::cout << "── Image " << i << ": " << class_names[i] << " ──\n";

        Tensor output = cnn.forward(*images[i], i == 0);  // verbose for first only

        // Find prediction
        int pred = 0;
        float max_prob = output[0];
        for (int j = 1; j < output.size(); j++) {
            if (output[j] > max_prob) { max_prob = output[j]; pred = j; }
        }

        std::cout << "  Prediction: [";
        for (int j = 0; j < output.size(); j++) {
            std::cout << std::fixed << std::setprecision(1) << output[j] * 100.0f << "%";
            if (j < output.size() - 1) std::cout << ", ";
        }
        std::cout << "] → Class " << pred << " (" << class_names[pred] << ")\n\n";
    }

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    // ─── Performance ───
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "  PERFORMANCE\n";
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "  3 images in:    " << std::fixed << std::setprecision(3) << ms << " ms\n";
    std::cout << "  Per image:      " << ms / 3.0 << " ms\n";
    std::cout << "  Estimated FPS:  " << static_cast<int>(3000.0 / ms) << "\n\n";

    // ─── C vs C++ Comparison ───
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "  C (Day 4) vs C++ (Day 6) COMPARISON\n";
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "  ┌─────────────────────┬──────────────────┬──────────────────┐\n";
    std::cout << "  │ Feature             │ C (Day 4)        │ C++ (Day 6)      │\n";
    std::cout << "  ├─────────────────────┼──────────────────┼──────────────────┤\n";
    std::cout << "  │ Memory management   │ malloc/free      │ RAII (auto)      │\n";
    std::cout << "  │ Data structure      │ float* + int     │ Tensor class     │\n";
    std::cout << "  │ Layer interface     │ void func(...)   │ class Layer      │\n";
    std::cout << "  │ Adding layers       │ manual calls     │ cnn.add(layer)   │\n";
    std::cout << "  │ Memory leaks        │ Possible         │ Impossible (RAII)│\n";
    std::cout << "  │ Code readability    │ Low              │ High             │\n";
    std::cout << "  │ Speed               │ Same             │ Same             │\n";
    std::cout << "  │ Binary size         │ ~20KB            │ ~50KB            │\n";
    std::cout << "  │ Used by TFLite?     │ No               │ YES              │\n";
    std::cout << "  │ Used by ONNX RT?    │ No               │ YES              │\n";
    std::cout << "  └─────────────────────┴──────────────────┴──────────────────┘\n";

    // ─── Summary ───
    std::cout << "\n╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║              DAY 6 COMPLETE                              ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════╣\n";
    std::cout << "║                                                          ║\n";
    std::cout << "║  C++ Concepts Learned:                                   ║\n";
    std::cout << "║    ✓ Classes: Tensor, Conv2D, Dense, CNN                ║\n";
    std::cout << "║    ✓ Inheritance: Layer → Conv2D, ReLU, Dense           ║\n";
    std::cout << "║    ✓ Virtual functions: forward() override              ║\n";
    std::cout << "║    ✓ std::vector: replaces malloc/free                  ║\n";
    std::cout << "║    ✓ std::unique_ptr: smart pointer (RAII)              ║\n";
    std::cout << "║    ✓ operator(): custom indexing tensor(c,y,x)         ║\n";
    std::cout << "║    ✓ std::chrono: high-resolution timing               ║\n";
    std::cout << "║    ✓ dynamic_cast: safe type checking                  ║\n";
    std::cout << "║                                                          ║\n";
    std::cout << "║  Key Insight:                                            ║\n";
    std::cout << "║    C++ is NOT slower than C.                             ║\n";
    std::cout << "║    It is C with safety and structure added on top.       ║\n";
    std::cout << "║    RAII makes memory leaks IMPOSSIBLE by design.         ║\n";
    std::cout << "║                                                          ║\n";
    std::cout << "║  This is how TFLite, ONNX Runtime, and TensorRT         ║\n";
    std::cout << "║  structure their inference engines internally.           ║\n";
    std::cout << "║                                                          ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n";

    // No free(), no delete, no cleanup needed.
    // When main() ends, all unique_ptrs and vectors auto-destruct. RAII!

    return 0;
}
