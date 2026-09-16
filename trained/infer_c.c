#include "run_common.h"
#include "../engine/cnn_c.h"

static void forward(const Run* r, const float* x, float* conv, float* pool,
                    float* output, int image, int trace) {
    int input_n = r->channels * r->h * r->w;
    int conv_n = r->filters * r->h * r->w;
    if (trace) emit_tensor(image, "input", x, input_n);
    conv2d(x, r->channels, r->h, r->w, r->cw->data, r->cb->data,
           r->filters, conv, r->kernel);
    if (trace) emit_tensor(image, "conv", conv, conv_n);
    relu(conv, conv_n);
    if (trace) emit_tensor(image, "relu", conv, conv_n);
    maxpool2d(conv, r->filters, r->h, r->w, 2, pool);
    if (trace) {
        emit_tensor(image, "pool", pool, r->flat);
        emit_tensor(image, "flatten", pool, r->flat);
    }
    dense(pool, r->flat, r->dw->data, r->db->data, r->classes, output);
    if (trace) emit_tensor(image, "dense", output, r->classes);
    softmax_inplace(output, r->classes);
}

int main(int argc, char** argv) {
    int trace, repeats;
    if (run_options(argc, argv, &trace, &repeats)) return 1;
    Run r;
    if (run_load(&r, argv[1], argv[2])) return 1;
    float* conv = (float*)malloc((size_t)r.filters * r.h * r.w * sizeof(float));
    float* pool = (float*)malloc((size_t)r.flat * sizeof(float));
    float* output = (float*)malloc((size_t)r.classes * sizeof(float));
    if (!conv || !pool || !output) {
        fprintf(stderr, "inference: allocation failed\n");
        free(conv); free(pool); free(output); run_free(&r); return 1;
    }
    int stride = r.channels * r.h * r.w;
    if (repeats) {
        for (int i = 0; i < 10; ++i) forward(&r, r.input->data, conv, pool, output, 0, 0);
        double checksum = 0, start = now_seconds();
        for (int i = 0; i < repeats; ++i) {
            forward(&r, r.input->data + (i % r.n) * stride, conv, pool, output, 0, 0);
            checksum += output[i % r.classes];
        }
        double elapsed = now_seconds() - start;
        emit_benchmark(repeats, elapsed, checksum);
    } else {
        for (int i = 0; i < r.n; ++i) {
            forward(&r, r.input->data + i * stride, conv, pool, output, i, trace);
            emit_tensor(i, "softmax", output, r.classes);
        }
    }
    free(conv); free(pool); free(output); run_free(&r);
    return 0;
}
