#include "run_common.h"
#include "../engine/cnn.hpp"
#include <iostream>

struct OwnedRun {
    Run value{};
    ~OwnedRun() { run_free(&value); }
};

int main(int argc, char** argv) {
    int trace, repeats;
    if (run_options(argc, argv, &trace, &repeats)) return 1;
    OwnedRun owned;
    Run& r = owned.value;
    if (run_load(&r, argv[1], argv[2])) return 1;
    try {
        cnn::Network net;
        net.add(std::make_unique<cnn::Conv2D>(r.channels, r.filters, r.kernel, r.cw->data, r.cb->data));
        net.add(std::make_unique<cnn::ReLU>());
        net.add(std::make_unique<cnn::MaxPool2D>(2));
        net.add(std::make_unique<cnn::Flatten>());
        net.add(std::make_unique<cnn::Dense>(r.flat, r.classes, r.dw->data, r.db->data));
        net.add(std::make_unique<cnn::Softmax>());
        cnn::Tensor x(r.channels, r.h, r.w);
        auto predict = [&](int image, bool tracing) {
            std::copy_n(r.input->data + image * x.size(), x.size(), x.data.begin());
            if (!tracing) return net.forward(x);
            emit_tensor(image, "input", x.data.data(), x.size());
            return net.forward_traced(x, [image](const std::string& name, const cnn::Tensor& t) {
                emit_tensor(image, name.c_str(), t.data.data(), t.size());
            });
        };
        if (repeats) {
            for (int i = 0; i < 10; ++i) predict(0, false);
            double checksum = 0, start = now_seconds();
            for (int i = 0; i < repeats; ++i) {
                auto output = predict(i % r.n, false);
                checksum += output[i % r.classes];
            }
            double elapsed = now_seconds() - start;
            emit_benchmark(repeats, elapsed, checksum);
        } else {
            for (int i = 0; i < r.n; ++i) {
                auto output = predict(i, trace != 0);
                if (!trace) emit_tensor(i, "softmax", output.data.data(), output.size());
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "inference: " << error.what() << '\n';
        return 1;
    }
}
