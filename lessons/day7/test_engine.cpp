#include "../../engine/cnn.hpp"
#include <iostream>

template<class F> void rejects(F fn) {
    try { fn(); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid input was accepted");
}

int main() {
    float w[9] = {}, b[1] = {};
    rejects([] { cnn::Tensor t(-1, 2, 2); });
    rejects([] { cnn::Tensor t(2, 2147483647, 2); });
    rejects([&] { cnn::Conv2D c(1, 1, 2, w, b); });
    rejects([&] { cnn::Conv2D c(1, 1, 3, nullptr, b); });
    rejects([&] { cnn::Conv2D(1, 1, 3, w, b).forward(cnn::Tensor(2, 4, 4)); });
    rejects([] { cnn::MaxPool2D p(0); });
    rejects([] { cnn::MaxPool2D(4).forward(cnn::Tensor(1, 2, 2)); });
    rejects([&] { cnn::Dense(9, 1, w, b).forward(cnn::Tensor(8)); });
    rejects([] { cnn::Softmax().forward(cnn::Tensor()); });
    rejects([] { cnn::Tensor t(2); t.data.clear(); cnn::ReLU().forward(t); });
    cnn::Tensor x(3); x[0] = 1000; x[1] = 1000; x[2] = -1000;
    auto y = cnn::Softmax().forward(x);
    if (std::abs(y[0] - 0.5f) > 1e-6f || y[2] != 0)
        throw std::runtime_error("softmax stability regression");
    std::cout << "Engine validation and softmax stability passed\n";
}
