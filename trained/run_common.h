/* File/shape validation and CLI protocol shared by the two inference runners.
 * The arithmetic engines remain independent (cnn_c.h versus cnn.hpp).
 */
#ifndef TRAINED_RUN_COMMON_H
#define TRAINED_RUN_COMMON_H
#include "../engine/model_io.h"
#include <limits.h>
#include <math.h>
#include <time.h>
#include <errno.h>

typedef struct {
    Model weights, samples;
    const MTensor *cw, *cb, *dw, *db, *input;
    int n, channels, h, w, filters, kernel, flat, classes;
} Run;

static void run_free(Run* r) { model_free(&r->samples); model_free(&r->weights); }

static int run_load(Run* r, const char* weights, const char* samples) {
    memset(r, 0, sizeof(*r));
    if (model_load(&r->weights, weights) || model_load(&r->samples, samples)) {
        run_free(r); return 1;
    }
    r->cw = model_get(&r->weights, "conv.weight");
    r->cb = model_get(&r->weights, "conv.bias");
    r->dw = model_get(&r->weights, "dense.weight");
    r->db = model_get(&r->weights, "dense.bias");
    r->input = model_get(&r->samples, "input");
    if (!r->cw || !r->cb || !r->dw || !r->db || !r->input) goto invalid;
    if (r->cw->ndim != 4 || r->cb->ndim != 1 || r->dw->ndim != 2 ||
        r->db->ndim != 1 || r->input->ndim != 4) goto invalid;
    /* The file loader bounds element counts; additionally bound every int index
     * used for intermediate activations before allocating or multiplying. */
    r->n = (int)r->input->dims[0]; r->channels = (int)r->input->dims[1];
    r->h = (int)r->input->dims[2]; r->w = (int)r->input->dims[3];
    r->filters = (int)r->cw->dims[0]; r->kernel = (int)r->cw->dims[2];
    r->classes = (int)r->db->dims[0];
    if (r->h < 2 || r->w < 2 || r->kernel % 2 != 1 ||
        r->cw->dims[1] != (uint32_t)r->channels ||
        r->cw->dims[3] != (uint32_t)r->kernel ||
        r->cb->dims[0] != (uint32_t)r->filters ||
        r->dw->dims[1] != (uint32_t)r->classes) goto invalid;
    if ((uint64_t)r->filters * r->h * r->w > INT_MAX) goto invalid;
    r->flat = r->filters * (r->h / 2) * (r->w / 2);
    if (r->dw->dims[0] != (uint32_t)r->flat) goto invalid;
    for (uint32_t t = 0; t < r->weights.n_tensors; ++t)
        for (uint32_t i = 0; i < r->weights.tensors[t].n_elem; ++i)
            if (!isfinite(r->weights.tensors[t].data[i])) goto invalid;
    for (uint32_t i = 0; i < r->input->n_elem; ++i)
        if (!isfinite(r->input->data[i])) goto invalid;
    return 0;
invalid:
    fprintf(stderr, "inference: invalid model/input shapes or non-finite values\n");
    run_free(r); return 1;
}

static void emit_tensor(int image, const char* name, const float* data, int n) {
    printf("TENSOR %d %s %d\n", image, name, n);
    for (int i = 0; i < n; ++i) printf("%.9g\n", data[i]);
}

static int run_options(int argc, char** argv, int* trace, int* repeats) {
    *trace = 0; *repeats = 0;
    if (argc == 3) return 0;
    if (argc == 4 && strcmp(argv[3], "--trace") == 0) { *trace = 1; return 0; }
    if (argc == 5 && strcmp(argv[3], "--benchmark") == 0) {
        char* end;
        errno = 0;
        long n = strtol(argv[4], &end, 10);
        if (!errno && *end == '\0' && n >= 10 && n <= 10000000) {
            *repeats = (int)n; return 0;
        }
    }
    fprintf(stderr, "usage: %s weights.bin samples.bin [--trace | --benchmark repeats>=10]\n", argv[0]);
    return 1;
}

static double now_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) { perror("clock_gettime"); exit(1); }
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void emit_benchmark(int repeats, double seconds, double checksum) {
    printf("{\"repeats\":%d,\"seconds\":%.9g,\"latency_us\":%.9g,"
           "\"fps\":%.9g,\"checksum\":%.9g}\n", repeats, seconds,
           seconds * 1e6 / repeats, repeats / seconds, checksum);
}
#endif
