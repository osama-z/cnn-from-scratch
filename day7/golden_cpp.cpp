/**
 * DAY 7: THE C++ IMPLEMENTATION, LOADING THE GOLDEN MODEL
 * ========================================================
 *
 * This file used to spell out all six layer classes inline — a copy of what
 * Day 6 already contained. That copy is now gone: the engine lives in cnn.hpp
 * and this file does only what it uniquely has to do, which is wire the loaded
 * weights into a network and emit the trace verify.py compares.
 *
 * Two headers, two jobs:
 *   model_io.h  parses weights.bin   (shared with golden_c.c — SAME parser, so
 *                                     "C and C++ agree" is a claim about two
 *                                     inference engines, not two file readers)
 *   cnn.hpp     the float engine     (shared with Phase 2's ARM/NEON build)
 */

#include <cstdio>
#include <vector>
#include <memory>
#include "model_io.h"
#include "cnn.hpp"

constexpr int IN_CH = 1, OUT_CH = 2, KSIZE = 3;
constexpr int IMG = 8, POOL = 2, NCLS = 3;
constexpr int FLAT = OUT_CH * (IMG / POOL) * (IMG / POOL);   // 32
constexpr int NIMG = 4;   // 3 originals + the softmax stability probe

static void emit(int img, const std::string& layer, const cnn::Tensor& t) {
    std::printf("TENSOR %d %s %d\n", img, layer.c_str(), t.size());
    for (int i = 0; i < t.size(); ++i) std::printf("%.9g\n", t[i]);
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "weights.bin";

    Model model;
    if (model_load(&model, path) != 0) return 1;

    const MTensor* cw = model_require(&model, "conv.weight");
    const MTensor* cb = model_require(&model, "conv.bias");
    const MTensor* dw = model_require(&model, "dense.weight");
    const MTensor* db = model_require(&model, "dense.bias");

    // The golden model froze these shapes. A silent mismatch would produce
    // plausible-looking wrong numbers, which is the exact failure this whole
    // exercise exists to eliminate — so check rather than assume.
    if (cw->n_elem != OUT_CH * IN_CH * KSIZE * KSIZE || cb->n_elem != OUT_CH ||
        dw->n_elem != FLAT * NCLS || db->n_elem != NCLS) {
        std::fprintf(stderr, "golden_cpp: model shapes do not match architecture\n");
        model_free(&model);
        return 1;
    }

    // Layers borrow pointers into model.blob, so `model` must outlive `net`.
    cnn::Network net;
    net.add(std::make_unique<cnn::Conv2D>(IN_CH, OUT_CH, KSIZE, cw->data, cb->data));
    net.add(std::make_unique<cnn::ReLU>());
    net.add(std::make_unique<cnn::MaxPool2D>(POOL));
    net.add(std::make_unique<cnn::Flatten>());
    net.add(std::make_unique<cnn::Dense>(FLAT, NCLS, dw->data, db->data));
    net.add(std::make_unique<cnn::Softmax>());

    // Three test images, identical to export_weights.py's build_images().
    // Image 3 is the softmax stability probe: amplitude 60 drives the largest
    // logit to ~100, and exp(100) overflows float32 unless the max is
    // subtracted first. See the comment in export_weights.py.
    std::vector<cnn::Tensor> images(NIMG, cnn::Tensor(IN_CH, IMG, IMG));
    for (int y = 4; y < 8; ++y) for (int x = 0; x < 8; ++x) images[0](0, y, x) = 10.0f;
    for (int y = 0; y < 8; ++y) for (int x = 4; x < 8; ++x) images[1](0, y, x) = 10.0f;
    for (int i = 0; i < images[2].size(); ++i)              images[2][i]       = 5.0f;
    for (int y = 4; y < 8; ++y) for (int x = 0; x < 8; ++x) images[3](0, y, x) = 60.0f;

    for (int i = 0; i < NIMG; ++i) {
        emit(i, "input", images[i]);
        net.forward_traced(images[i],
                           [i](const std::string& name, const cnn::Tensor& t) {
                               emit(i, name, t);
                           });
    }

    model_free(&model);
    return 0;   // unique_ptrs unwind here
}
