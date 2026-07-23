/**
 * DAY 6, PART 2: ONE SOURCE, TWO NUMBER TYPES (C++ TEMPLATES)
 * ===========================================================
 *
 * WHAT THIS FILE ADDS OVER cnn_forward.cpp
 * ─────────────────────────────────────────
 * cnn_forward.cpp is the object-oriented CNN: classes, inheritance, RAII.
 * But its Tensor holds `float` and only `float`. To run the int8 network
 * from Day 5, you would have to copy every layer and change the types by
 * hand — two copies of the same logic, drifting apart over time.
 *
 * This file writes each layer ONCE and instantiates it twice:
 *
 *     CNN<float>    fnet;   // float32 inference
 *     CNN<int8_t>   qnet;   // int8 quantized inference
 *
 * THE TRAP THIS SOLVES (and why templates are not just syntax sugar)
 * ──────────────────────────────────────────────────────────────────
 * Day 5 (day5/README_explanation.md, section 4) established:
 *
 *     int8 x int8 OVERFLOWS.  127 * 127 = 16,129, which does not fit in
 *     int8. Sum nine of those and you need 145,161. You MUST accumulate
 *     in int32.
 *
 * On Day 5 that was a rule you had to REMEMBER while writing C.
 * Forget it and there is no crash — just silently wrong predictions.
 *
 * A naive `Tensor<int8_t>` does NOT fix this. If you template the storage
 * type and let the accumulator follow it, you get an int8 accumulator and
 * the exact overflow Day 5 warned about.
 *
 * The fix is to make the accumulator a SEPARATE type, chosen by the
 * compiler from the storage type:
 *
 *     Tensor<float>   ->  accumulate in float
 *     Tensor<int8_t>  ->  accumulate in int32_t
 *
 * That is what AccumTraits below does. Day 5 made this a comment you had
 * to obey. Here the type system obeys it for you.
 *
 * This is not a toy pattern — CMSIS-NN and TFLite Micro carry exactly this
 * storage-type/accumulator-type split through their kernels.
 *
 * SECTIONS
 *   1. QuantParams + AccumTraits   <- the core idea
 *   2. Tensor<T>
 *   3. Layer<T> base class
 *   4. Conv2D<T>
 *   5. ReLU<T>          <- Day 5's zero-point trap, in one `if constexpr`
 *   6. MaxPool2D<T>, Flatten<T>
 *   7. Dense<T>
 *   8. Softmax<T>
 *   9. CNN<T>
 *  10. Calibration (a preview of Month 3's PTQ)
 *  11. main(): build both networks, run both, compare
 */

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <cstdint>
#include <string>
#include <algorithm>
#include <limits>
#include <type_traits>   // std::is_same_v
#include <iomanip>

// =====================================================================
// SECTION 1: QUANTIZATION PARAMS + THE ACCUMULATOR TRAIT
// =====================================================================

/**
 * The Day 5 formulas, unchanged (day5/README_explanation.md, section 3):
 *
 *     QUANTIZE:    q = round(x / scale) + zero_point
 *     DEQUANTIZE:  x = (q - zero_point) * scale
 */
struct QuantParams {
    float   scale      = 1.0f;
    int32_t zero_point = 0;
};

/**
 * ── THE HEART OF THIS FILE ──
 *
 * A "trait" is a compile-time lookup table: give it a type, it gives back
 * another type. The general template says "accumulate in float". The
 * SPECIALIZATION for int8_t overrides that and says "accumulate in int32".
 *
 * When you write Conv2D<int8_t>, the compiler consults this table and
 * stamps out a version whose accumulator is int32_t. You cannot forget.
 * You cannot get it wrong. There is no runtime cost — the decision was
 * made while compiling.
 */
template <typename T> struct AccumTraits            { using type = float;   };
template <>           struct AccumTraits<int8_t>    { using type = int32_t; };

// Convenience alias so we can write Accum<T> instead of the long form.
template <typename T> using Accum = typename AccumTraits<T>::type;

// Proof, checked by the compiler every single build. If someone later
// "simplifies" AccumTraits so that int8 accumulates in int8, this file stops
// compiling instead of quietly producing wrong numbers on a drone.
static_assert(std::is_same_v<Accum<float>,  float>,
              "float network must accumulate in float");
static_assert(std::is_same_v<Accum<int8_t>, int32_t>,
              "int8 network MUST accumulate in int32 -- see day5 section 4");
static_assert(sizeof(Accum<int8_t>) == 4,
              "int8 accumulator must be 32-bit wide: 127*127*9 = 145161 needs it");

// A compile-time yes/no: "is this the quantized network?"
// Used with `if constexpr` below to switch behaviour at COMPILE time.
template <typename T> constexpr bool is_quantized_v = std::is_same_v<T, int8_t>;

// Clamp an int32 down into the int8 range. This is "saturation" — the same
// idea as saturating arithmetic in DSP hardware.
inline int8_t saturate_i8(int32_t v) {
    return static_cast<int8_t>(std::max(-128, std::min(127, v)));
}

/**
 * Requantization: an int32 accumulator carries units of (scale_in * scale_w).
 * To store it back as int8 we rescale into the output's units.
 *
 *     M = (scale_in * scale_w) / scale_out
 *
 * Real engines precompute M as a fixed-point integer multiplier plus a shift,
 * because embedded targets may have no FPU. A float M is the readable version
 * of the same arithmetic.
 */
inline int8_t requantize(int32_t acc, float M, int32_t zero_point) {
    return saturate_i8(static_cast<int32_t>(std::lround(acc * M)) + zero_point);
}

// Build quant params covering a float range. Range always includes 0.0,
// otherwise zero_point would fall outside int8 and padding/ReLU would break.
inline QuantParams qparams_from_range(float lo, float hi) {
    lo = std::min(lo, 0.0f);
    hi = std::max(hi, 0.0f);
    if (hi - lo < 1e-9f) hi = lo + 1e-6f;
    float scale = (hi - lo) / 255.0f;
    return { scale, static_cast<int32_t>(std::lround(-lo / scale)) - 128 };
}

// Weights use SYMMETRIC quantization: zero_point is exactly 0.
// That makes the conv inner loop cheaper — no subtraction on the weight side.
inline QuantParams qparams_symmetric(const std::vector<float>& w) {
    float m = 0.0f;
    for (float v : w) m = std::max(m, std::fabs(v));
    if (m < 1e-9f) m = 1e-6f;
    return { m / 127.0f, 0 };
}


// =====================================================================
// SECTION 2: TENSOR<T>
// =====================================================================

/**
 * Identical index arithmetic to Day 4's C:
 *
 *     C   (Day 4):  data[c * H * W + y * W + x]
 *     C++ (Day 6):  tensor(c, y, x)
 *
 * Same formula. The difference is that operator() hides it, so an off-by-one
 * in the index math can only exist in ONE place instead of everywhere.
 *
 * The tensor carries its own QuantParams. For Tensor<float> they are ignored;
 * for Tensor<int8_t> they say what the stored integers actually mean.
 */
template <typename T>
class Tensor {
public:
    int channels = 0, height = 0, width = 0;
    std::vector<T> data;
    QuantParams q;

    Tensor() = default;

    Tensor(int c, int h, int w)
        : channels(c), height(h), width(w),
          data(static_cast<size_t>(c) * h * w, T{}) {}

    explicit Tensor(int n)
        : channels(1), height(1), width(n), data(n, T{}) {}

    int size() const { return static_cast<int>(data.size()); }

    T& operator()(int c, int y, int x) {
        return data[c * (height * width) + y * width + x];
    }
    const T& operator()(int c, int y, int x) const {
        return data[c * (height * width) + y * width + x];
    }

    T&       operator[](int i)       { return data[i]; }
    const T& operator[](int i) const { return data[i]; }

    // Read element i as a real number, whatever T is.
    // This is the ONLY place the two number systems have to meet.
    float real(int i) const {
        if constexpr (is_quantized_v<T>)
            return (static_cast<float>(data[i]) - q.zero_point) * q.scale;
        else
            return static_cast<float>(data[i]);
    }
};

using TensorF = Tensor<float>;    // float32 network
using TensorQ = Tensor<int8_t>;   // int8 network


// =====================================================================
// SECTION 3: LAYER<T> — THE ABSTRACT INTERFACE
// =====================================================================

/**
 * WHY IS THE WHOLE CLASS TEMPLATED, NOT JUST forward()?
 *
 * You might reasonably try this instead:
 *
 *     class Layer {
 *         template <typename T>
 *         virtual Tensor<T> forward(const Tensor<T>&) = 0;   // ILLEGAL
 *     };
 *
 * C++ forbids it: a member function cannot be both `virtual` and a template.
 *
 * The reason is how virtual dispatch works. Each polymorphic class has a
 * vtable — a fixed-size array of function pointers, built at compile time.
 * Templates are instantiated on demand, so a virtual template would need one
 * vtable slot per type anyone ever instantiates it with, including types in
 * source files not yet written. The table size would be unbounded, so the
 * compiler cannot lay it out.
 *
 * So the choice is: templated class with virtual methods (what we do here —
 * Layer<float> and Layer<int8_t> are two unrelated classes, each with its own
 * ordinary vtable), or a non-templated class with type erasure.
 *
 * This is a genuine language constraint, not a style preference, and it is
 * the kind of detail that separates "I used C++" from "I understand C++".
 */
template <typename T>
class Layer {
public:
    std::string name;
    explicit Layer(std::string n) : name(std::move(n)) {}
    virtual Tensor<T> forward(const Tensor<T>& input) = 0;
    virtual ~Layer() = default;
};


// =====================================================================
// SECTION 4: CONV2D<T>
// =====================================================================

template <typename T>
class Conv2D : public Layer<T> {
public:
    int in_channels, out_channels, kernel_size;

    std::vector<T>        weights;   // T           -> float or int8_t
    std::vector<Accum<T>> biases;    // Accum<T>    -> float or int32_t
    QuantParams w_q;                 // weight quantization (symmetric)
    QuantParams out_q;               // output quantization

    Conv2D(int in_ch, int out_ch, int k)
        : Layer<T>("Conv2D(" + std::to_string(in_ch) + "->" + std::to_string(out_ch) + ")"),
          in_channels(in_ch), out_channels(out_ch), kernel_size(k),
          weights(static_cast<size_t>(out_ch) * in_ch * k * k, T{}),
          biases(out_ch, Accum<T>{}) {}

    Tensor<T> forward(const Tensor<T>& input) override {
        const int H = input.height, W = input.width;
        const int kh = kernel_size / 2;

        Tensor<T> output(out_channels, H, W);

        // Hoisted out of the loops: the requantization multiplier is constant
        // for the whole layer. Real engines precompute it once at load time.
        float M = 1.0f;
        if constexpr (is_quantized_v<T>) {
            output.q = out_q;
            M = (input.q.scale * w_q.scale) / out_q.scale;
        }

        for (int oc = 0; oc < out_channels; ++oc) {
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {

                    // ── THE LINE THIS WHOLE FILE EXISTS FOR ──
                    // float network  -> Accum<T> is float
                    // int8  network  -> Accum<T> is int32_t
                    // Chosen by the compiler. Day 5's rule, enforced.
                    Accum<T> acc = biases[oc];

                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int ky = -kh; ky <= kh; ++ky) {
                            for (int kx = -kh; kx <= kh; ++kx) {
                                const int iy = y + ky, ix = x + kx;
                                if (iy < 0 || iy >= H || ix < 0 || ix >= W)
                                    continue;   // zero padding

                                const int w_idx =
                                      oc * (in_channels * kernel_size * kernel_size)
                                    + ic * (kernel_size * kernel_size)
                                    + (ky + kh) * kernel_size + (kx + kh);

                                if constexpr (is_quantized_v<T>) {
                                    // Subtract zero points to recover the real
                                    // integers, then multiply. Both operands are
                                    // int8; the product is promoted to int32
                                    // BEFORE it can overflow.
                                    acc += (static_cast<int32_t>(input(ic, iy, ix)) - input.q.zero_point)
                                         * (static_cast<int32_t>(weights[w_idx]) - w_q.zero_point);
                                } else {
                                    acc += input(ic, iy, ix) * weights[w_idx];
                                }
                            }
                        }
                    }

                    if constexpr (is_quantized_v<T>)
                        output(oc, y, x) = requantize(acc, M, out_q.zero_point);
                    else
                        output(oc, y, x) = acc;
                }
            }
        }
        return output;
    }
};


// =====================================================================
// SECTION 5: RELU<T> — DAY 5'S TRAP, IN ONE `if constexpr`
// =====================================================================

/**
 * day5/README_explanation.md section 5:
 *
 *     float: relu(x) = max(0.0, x)
 *     int8 : relu(x) = max(zero_point, x)      <- NOT max(0, x)
 *
 * The integer 0 does not represent the real value 0.0. The zero_point does.
 * Using max(0, x) in int8 produces no crash and no warning — just quietly
 * wrong answers, which is the worst kind of bug on a drone.
 *
 * `if constexpr` is resolved at COMPILE time. The float instantiation of this
 * class contains literally none of the int8 branch — not a dead branch, not a
 * predicted-away branch: the machine code is never emitted. "One code path"
 * is a convenience for the reader, never a cost for the CPU.
 */
template <typename T>
class ReLU : public Layer<T> {
public:
    ReLU() : Layer<T>("ReLU") {}

    Tensor<T> forward(const Tensor<T>& input) override {
        Tensor<T> out = input;
        if constexpr (is_quantized_v<T>) {
            const T zp = static_cast<T>(out.q.zero_point);
            for (int i = 0; i < out.size(); ++i)
                if (out[i] < zp) out[i] = zp;
        } else {
            for (int i = 0; i < out.size(); ++i)
                if (out[i] < T{0}) out[i] = T{0};
        }
        return out;
    }
};


// =====================================================================
// SECTION 6: MAXPOOL2D<T> AND FLATTEN<T>
// =====================================================================

/**
 * MaxPool needs NO int8 special case at all.
 *
 * Quantization is monotonic — scale is always positive — so if q1 > q2 then
 * real(q1) > real(q2). Picking the largest integer picks the largest real
 * value. The same comparison is correct in both worlds, which is why real
 * engines run pooling directly on quantized data.
 */
template <typename T>
class MaxPool2D : public Layer<T> {
public:
    int pool_size;

    explicit MaxPool2D(int p = 2)
        : Layer<T>("MaxPool2D(" + std::to_string(p) + ")"), pool_size(p) {}

    Tensor<T> forward(const Tensor<T>& input) override {
        const int oh = input.height / pool_size;
        const int ow = input.width  / pool_size;

        Tensor<T> out(input.channels, oh, ow);
        out.q = input.q;   // pooling does not change what the numbers mean

        for (int c = 0; c < input.channels; ++c)
            for (int y = 0; y < oh; ++y)
                for (int x = 0; x < ow; ++x) {
                    T best = std::numeric_limits<T>::lowest();
                    for (int py = 0; py < pool_size; ++py)
                        for (int px = 0; px < pool_size; ++px) {
                            T v = input(c, y * pool_size + py, x * pool_size + px);
                            if (v > best) best = v;
                        }
                    out(c, y, x) = best;
                }
        return out;
    }
};

template <typename T>
class Flatten : public Layer<T> {
public:
    Flatten() : Layer<T>("Flatten") {}

    Tensor<T> forward(const Tensor<T>& input) override {
        Tensor<T> out(input.size());
        out.data = input.data;   // pure reshape — no arithmetic
        out.q    = input.q;
        return out;
    }
};


// =====================================================================
// SECTION 7: DENSE<T>
// =====================================================================

template <typename T>
class Dense : public Layer<T> {
public:
    int in_size, out_size;
    std::vector<T>        weights;
    std::vector<Accum<T>> biases;
    QuantParams w_q, out_q;

    Dense(int in_n, int out_n)
        : Layer<T>("Dense(" + std::to_string(in_n) + "->" + std::to_string(out_n) + ")"),
          in_size(in_n), out_size(out_n),
          weights(static_cast<size_t>(in_n) * out_n, T{}),
          biases(out_n, Accum<T>{}) {}

    Tensor<T> forward(const Tensor<T>& input) override {
        Tensor<T> out(out_size);

        float M = 1.0f;
        if constexpr (is_quantized_v<T>) {
            out.q = out_q;
            M = (input.q.scale * w_q.scale) / out_q.scale;
        }

        for (int j = 0; j < out_size; ++j) {
            Accum<T> acc = biases[j];       // float or int32, chosen by the compiler
            for (int i = 0; i < in_size; ++i) {
                if constexpr (is_quantized_v<T>) {
                    acc += (static_cast<int32_t>(input[i]) - input.q.zero_point)
                         * (static_cast<int32_t>(weights[i * out_size + j]) - w_q.zero_point);
                } else {
                    acc += input[i] * weights[i * out_size + j];
                }
            }
            if constexpr (is_quantized_v<T>)
                out[j] = requantize(acc, M, out_q.zero_point);
            else
                out[j] = acc;
        }
        return out;
    }
};


// =====================================================================
// SECTION 8: SOFTMAX<T>
// =====================================================================

/**
 * Softmax needs exp(), which has no sensible int8 form. Every real int8
 * engine does the same thing this does: dequantize, compute in float,
 * requantize. Quantization is applied where it pays (the millions of
 * multiply-accumulates), not where it does not (three exponentials).
 */
template <typename T>
class Softmax : public Layer<T> {
public:
    Softmax() : Layer<T>("Softmax") {}

    Tensor<T> forward(const Tensor<T>& input) override {
        const int n = input.size();
        std::vector<float> p(n);

        for (int i = 0; i < n; ++i) p[i] = input.real(i);   // -> real numbers

        // Subtract the max before exp() — the standard stability trick from Day 1.
        const float mx = *std::max_element(p.begin(), p.end());
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) { p[i] = std::exp(p[i] - mx); sum += p[i]; }
        for (int i = 0; i < n; ++i) p[i] /= sum;

        Tensor<T> out(n);
        if constexpr (is_quantized_v<T>) {
            // Probabilities live in [0,1]: map that range onto the int8 line.
            out.q = { 1.0f / 255.0f, -128 };
            for (int i = 0; i < n; ++i)
                out[i] = saturate_i8(
                    static_cast<int32_t>(std::lround(p[i] / out.q.scale)) + out.q.zero_point);
        } else {
            for (int i = 0; i < n; ++i) out[i] = p[i];
        }
        return out;
    }
};


// =====================================================================
// SECTION 9: CNN<T>
// =====================================================================

template <typename T>
class CNN {
public:
    std::vector<std::unique_ptr<Layer<T>>> layers;

    void add(std::unique_ptr<Layer<T>> l) { layers.push_back(std::move(l)); }

    Tensor<T> forward(const Tensor<T>& input, bool verbose = false) {
        Tensor<T> x = input;
        for (auto& l : layers) {
            x = l->forward(x);
            if (verbose)
                std::cout << "    " << std::setw(22) << std::left << l->name
                          << " -> " << x.channels << "x" << x.height << "x" << x.width
                          << "  (" << x.size() << " values)\n";
        }
        return x;
    }

    // Run the float net and record each layer's output range.
    // THIS IS CALIBRATION — the "PTQ" of Month 3, Week 9, in nine lines.
    // Real calibration uses hundreds of representative images; the principle
    // does not change.
    std::vector<std::pair<float,float>> calibrate(const std::vector<Tensor<T>>& samples) {
        std::vector<std::pair<float,float>> ranges(
            layers.size(),
            { std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest() });

        for (const auto& s : samples) {
            Tensor<T> x = s;
            for (size_t i = 0; i < layers.size(); ++i) {
                x = layers[i]->forward(x);
                for (int k = 0; k < x.size(); ++k) {
                    ranges[i].first  = std::min(ranges[i].first,  x.real(k));
                    ranges[i].second = std::max(ranges[i].second, x.real(k));
                }
            }
        }
        return ranges;
    }
};


// =====================================================================
// SECTION 10: WEIGHT GENERATION
// =====================================================================

/**
 * Deterministic on purpose — no rand().
 *
 * day4/cnn_forward.c:384 and day6/cnn_forward.cpp:321 both call srand(42) but
 * use DIFFERENT formulas, so their weights differ and their predictions differ.
 * Day 7 has to prove Python, C and C++ agree, which is impossible while any of
 * them invents its own weights.
 *
 * A closed-form function of the index is reproducible across languages,
 * compilers and machines. Day 7 replaces this with weights loaded from a file.
 */
inline std::vector<float> deterministic_weights(size_t n, float scale) {
    std::vector<float> w(n);
    for (size_t i = 0; i < n; ++i)
        w[i] = ((static_cast<float>((i * 37) % 101) / 100.0f) - 0.5f) * 2.0f * scale;
    return w;
}

// The two hand-designed edge detectors from Day 1, reused ever since.
static const std::vector<float> kHorizontalEdge = { -1,-1,-1,  0, 0, 0,  1, 1, 1 };
static const std::vector<float> kVerticalEdge   = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };


// =====================================================================
// SECTION 11: MAIN
// =====================================================================

int main() {
    std::cout << "+----------------------------------------------------------+\n";
    std::cout << "|  DAY 6 PART 2: ONE SOURCE, TWO NUMBER TYPES              |\n";
    std::cout << "|  CNN<float> and CNN<int8_t> from the same layer code     |\n";
    std::cout << "+----------------------------------------------------------+\n";

    // ── Architecture (identical to Day 1 / Day 4 / Day 6 part 1) ──
    const int  IN_CH = 1, OUT_CH = 2, KSIZE = 3, IMG = 8, POOL = 2, NCLS = 3;
    const int  FLAT  = OUT_CH * (IMG / POOL) * (IMG / POOL);   // 2*4*4 = 32
    const char* class_names[NCLS] = { "Horizontal Edge", "Vertical Edge", "Uniform" };

    // ── Shared float weights: both networks start from these ──
    std::vector<float> conv_w(static_cast<size_t>(OUT_CH) * IN_CH * KSIZE * KSIZE);
    std::copy(kHorizontalEdge.begin(), kHorizontalEdge.end(), conv_w.begin());
    std::copy(kVerticalEdge.begin(),   kVerticalEdge.end(),   conv_w.begin() + 9);

    const std::vector<float> dense_w =
        deterministic_weights(static_cast<size_t>(FLAT) * NCLS,
                              1.0f / std::sqrt(static_cast<float>(FLAT)));
    const std::vector<float> dense_b = { 0.1f, -0.1f, 0.0f };

    // ── Test images ──
    TensorF img_h(IN_CH, IMG, IMG);   // bottom half bright  -> horizontal edge
    for (int y = 4; y < 8; ++y) for (int x = 0; x < 8; ++x) img_h(0, y, x) = 10.0f;

    TensorF img_v(IN_CH, IMG, IMG);   // right half bright   -> vertical edge
    for (int y = 0; y < 8; ++y) for (int x = 4; x < 8; ++x) img_v(0, y, x) = 10.0f;

    TensorF img_u(IN_CH, IMG, IMG);   // flat                -> uniform
    for (int i = 0; i < img_u.size(); ++i) img_u[i] = 5.0f;

    const std::vector<TensorF> images = { img_h, img_v, img_u };

    // =================================================================
    // BUILD THE FLOAT NETWORK
    // =================================================================
    CNN<float> fnet;
    {
        auto conv = std::make_unique<Conv2D<float>>(IN_CH, OUT_CH, KSIZE);
        conv->weights = conv_w;
        fnet.add(std::move(conv));
        fnet.add(std::make_unique<ReLU<float>>());
        fnet.add(std::make_unique<MaxPool2D<float>>(POOL));
        fnet.add(std::make_unique<Flatten<float>>());
        auto dense = std::make_unique<Dense<float>>(FLAT, NCLS);
        dense->weights = dense_w;
        dense->biases  = dense_b;
        fnet.add(std::move(dense));
        fnet.add(std::make_unique<Softmax<float>>());
    }

    std::cout << "\n== Architecture (shared by both networks) ==\n";
    fnet.forward(images[0], true);

    // =================================================================
    // CALIBRATE, THEN BUILD THE INT8 NETWORK
    // =================================================================
    const auto ranges = fnet.calibrate(images);

    std::cout << "\n== Calibration: observed range of every layer output ==\n";
    std::cout << "   (this is Post-Training Quantization -- Month 3, Week 9)\n";
    for (size_t i = 0; i < ranges.size(); ++i)
        std::cout << "    " << std::setw(22) << std::left << fnet.layers[i]->name
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(10) << ranges[i].first << "  to "
                  << std::setw(10) << ranges[i].second << "\n";

    // Input range comes from the images themselves: 0 .. 10
    const QuantParams in_q     = qparams_from_range(0.0f, 10.0f);
    const QuantParams conv_out = qparams_from_range(ranges[0].first, ranges[0].second);
    const QuantParams relu_out = qparams_from_range(ranges[1].first, ranges[1].second);
    const QuantParams dens_out = qparams_from_range(ranges[4].first, ranges[4].second);

    const QuantParams conv_wq = qparams_symmetric(conv_w);
    const QuantParams dens_wq = qparams_symmetric(dense_w);

    CNN<int8_t> qnet;
    {
        auto conv = std::make_unique<Conv2D<int8_t>>(IN_CH, OUT_CH, KSIZE);
        conv->w_q   = conv_wq;
        conv->out_q = conv_out;
        for (size_t i = 0; i < conv_w.size(); ++i)
            conv->weights[i] = saturate_i8(
                static_cast<int32_t>(std::lround(conv_w[i] / conv_wq.scale)));
        // Bias lives in accumulator units: scale_in * scale_w, zero_point 0.
        for (int i = 0; i < OUT_CH; ++i) conv->biases[i] = 0;
        qnet.add(std::move(conv));

        qnet.add(std::make_unique<ReLU<int8_t>>());
        qnet.add(std::make_unique<MaxPool2D<int8_t>>(POOL));
        qnet.add(std::make_unique<Flatten<int8_t>>());

        auto dense = std::make_unique<Dense<int8_t>>(FLAT, NCLS);
        dense->w_q   = dens_wq;
        dense->out_q = dens_out;
        for (size_t i = 0; i < dense_w.size(); ++i)
            dense->weights[i] = saturate_i8(
                static_cast<int32_t>(std::lround(dense_w[i] / dens_wq.scale)));
        const float bias_scale = relu_out.scale * dens_wq.scale;
        for (int i = 0; i < NCLS; ++i)
            dense->biases[i] = static_cast<int32_t>(std::lround(dense_b[i] / bias_scale));
        qnet.add(std::move(dense));

        qnet.add(std::make_unique<Softmax<int8_t>>());
    }

    std::cout << "\n== Quantization parameters chosen ==\n";
    auto show = [](const char* n, const QuantParams& p) {
        std::cout << "    " << std::setw(22) << std::left << n << std::right
                  << "scale " << std::setw(10) << std::fixed << std::setprecision(6) << p.scale
                  << "   zero_point " << std::setw(5) << p.zero_point << "\n";
    };
    show("input",         in_q);
    show("conv weights",  conv_wq);
    show("conv output",   conv_out);
    show("relu output",   relu_out);
    show("dense weights", dens_wq);
    show("dense output",  dens_out);

    // =================================================================
    // RUN BOTH NETWORKS ON THE SAME IMAGES
    // =================================================================
    std::cout << "\n== float32 vs int8, same layer code, same weights ==\n\n";
    std::cout << "    image             float32 probabilities        "
                 "int8 probabilities          max err\n";
    std::cout << "    ----------------------------------------------"
                 "----------------------------------------\n";

    float worst = 0.0f;
    int   agree = 0;

    for (int i = 0; i < NCLS; ++i) {
        // float path
        TensorF fout = fnet.forward(images[i]);

        // int8 path: quantize the image, then run entirely in integers
        TensorQ qin(IN_CH, IMG, IMG);
        qin.q = in_q;
        for (int k = 0; k < images[i].size(); ++k)
            qin[k] = saturate_i8(
                static_cast<int32_t>(std::lround(images[i][k] / in_q.scale)) + in_q.zero_point);
        TensorQ qout = qnet.forward(qin);

        int fpred = 0, qpred = 0;
        for (int j = 1; j < NCLS; ++j) {
            if (fout.real(j) > fout.real(fpred)) fpred = j;
            if (qout.real(j) > qout.real(qpred)) qpred = j;
        }
        if (fpred == qpred) ++agree;

        float err = 0.0f;
        for (int j = 0; j < NCLS; ++j)
            err = std::max(err, std::fabs(fout.real(j) - qout.real(j)));
        worst = std::max(worst, err);

        std::cout << "    " << std::setw(16) << std::left << class_names[i] << std::right;
        std::cout << "  [";
        for (int j = 0; j < NCLS; ++j)
            std::cout << std::setw(6) << std::fixed << std::setprecision(1)
                      << fout.real(j) * 100.0f << (j < NCLS - 1 ? "," : "");
        std::cout << " ]  [";
        for (int j = 0; j < NCLS; ++j)
            std::cout << std::setw(6) << std::fixed << std::setprecision(1)
                      << qout.real(j) * 100.0f << (j < NCLS - 1 ? "," : "");
        std::cout << " ]   " << std::setw(6) << std::setprecision(2) << err * 100.0f << "%\n";

        std::cout << "                      -> " << std::setw(18) << std::left
                  << class_names[fpred]
                  << "        -> " << std::setw(18) << class_names[qpred]
                  << (fpred == qpred ? "   MATCH" : "   DIVERGED") << std::right << "\n";
    }

    std::cout << "\n== Result ==\n";
    std::cout << "    Predictions agreeing: " << agree << " / " << NCLS << "\n";
    std::cout << "    Largest probability error: " << std::fixed << std::setprecision(2)
              << worst * 100.0f << " percentage points\n";
    std::cout << "    Weight memory: " << conv_w.size() + dense_w.size() << " values -> "
              << (conv_w.size() + dense_w.size()) * 4 << " bytes as float32, "
              << (conv_w.size() + dense_w.size())     << " bytes as int8   (4x smaller)\n";

    // NOTE ON THE PREDICTIONS THEMSELVES
    // The conv kernels are hand-designed edge detectors, but the Dense layer is
    // generated, never trained. A random matrix maps good features to arbitrary
    // classes, so the LABELS are meaningless. That is expected: this file tests
    // that two number systems agree, not that the network is accurate. Training
    // was Day 2's job. What matters here is that the float and int8 columns
    // track each other.

    std::cout << "\n+----------------------------------------------------------+\n";
    std::cout << "|  WHAT TEMPLATES BOUGHT                                   |\n";
    std::cout << "+----------------------------------------------------------+\n";
    std::cout << "|  Six layers written once, instantiated twice.            |\n";
    std::cout << "|  AccumTraits<int8_t> forces the int32 accumulator, so    |\n";
    std::cout << "|  Day 5's overflow rule cannot be forgotten.              |\n";
    std::cout << "|  if constexpr resolves at compile time: the float build  |\n";
    std::cout << "|  contains none of the int8 code, and vice versa.         |\n";
    std::cout << "|  Zero runtime cost for the abstraction.                  |\n";
    std::cout << "+----------------------------------------------------------+\n";

    return 0;   // every vector and unique_ptr unwinds here. RAII.
}
