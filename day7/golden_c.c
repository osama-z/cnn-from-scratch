/**
 * DAY 7, STEP 3: THE C IMPLEMENTATION, LOADING THE GOLDEN MODEL
 * =============================================================
 *
 * This is day4/cnn_forward.c with exactly one thing changed: it no longer
 * invents its weights. It loads them.
 *
 *   Day 4:   srand(42);
 *            dense_weights[i] = ((rand()%100)/100.0f - 0.5f) * 0.1f;
 *
 *   Day 7:   const MTensor* dw = model_require(&model, "dense.weight");
 *
 * The arithmetic below is untouched -- same loops, same 1D pointer indexing,
 * same zero padding. Day 4 stays in the repo as the teaching artifact for
 * pointers and malloc; this file is the one that has to agree with NumPy,
 * C++, and eventually VHDL.
 *
 * Output goes to stdout as machine-readable TENSOR blocks so verify.py can
 * compare it layer by layer against the NumPy reference.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "model_io.h"

#define IN_CH   1
#define OUT_CH  2
#define KSIZE   3
#define IMG     8
#define POOL    2
#define NCLS    3
#define FLAT    (OUT_CH * (IMG / POOL) * (IMG / POOL))   /* 32 */

/* ---------------------------------------------------------------- */
/* Layers -- identical math to day4/cnn_forward.c                    */
/* ---------------------------------------------------------------- */

static void conv2d(const float* in, int in_ch, int H, int W,
                   const float* kern, const float* bias, int out_ch,
                   float* out)
{
    const int kh = KSIZE / 2;
    for (int oc = 0; oc < out_ch; oc++) {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                float sum = bias[oc];
                for (int ic = 0; ic < in_ch; ic++) {
                    for (int ky = -kh; ky <= kh; ky++) {
                        for (int kx = -kh; kx <= kh; kx++) {
                            int iy = y + ky, ix = x + kx;
                            if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                                int in_idx = ic * (H * W) + iy * W + ix;
                                int k_idx  = oc * (in_ch * KSIZE * KSIZE)
                                           + ic * (KSIZE * KSIZE)
                                           + (ky + kh) * KSIZE + (kx + kh);
                                sum += in[in_idx] * kern[k_idx];
                            }
                        }
                    }
                }
                out[oc * (H * W) + y * W + x] = sum;
            }
        }
    }
}

static void relu(float* d, int n) {
    for (int i = 0; i < n; i++) if (d[i] < 0.0f) d[i] = 0.0f;
}

static void maxpool2d(const float* in, int ch, int H, int W, int p, float* out) {
    int oh = H / p, ow = W / p;
    for (int c = 0; c < ch; c++)
        for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow; x++) {
                float best = in[c * (H * W) + (y * p) * W + (x * p)];
                for (int py = 0; py < p; py++)
                    for (int px = 0; px < p; px++) {
                        float v = in[c * (H * W) + (y * p + py) * W + (x * p + px)];
                        if (v > best) best = v;
                    }
                out[c * (oh * ow) + y * ow + x] = best;
            }
}

static void dense(const float* in, int in_size,
                  const float* w, const float* b, int out_size, float* out)
{
    for (int j = 0; j < out_size; j++) {
        float sum = b[j];
        for (int i = 0; i < in_size; i++)
            sum += in[i] * w[i * out_size + j];   /* layout: (in_size, out_size) */
        out[j] = sum;
    }
}

static void softmax_inplace(float* d, int n) {
    float mx = d[0];
    for (int i = 1; i < n; i++) if (d[i] > mx) mx = d[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { d[i] = expf(d[i] - mx); sum += d[i]; }
    for (int i = 0; i < n; i++) d[i] /= sum;
}

/* ---------------------------------------------------------------- */

static void emit(int img, const char* layer, const float* d, int n) {
    printf("TENSOR %d %s %d\n", img, layer, n);
    for (int i = 0; i < n; i++) printf("%.9g\n", d[i]);
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "weights.bin";

    Model model;
    if (model_load(&model, path) != 0) return 1;

    const MTensor* cw = model_require(&model, "conv.weight");
    const MTensor* cb = model_require(&model, "conv.bias");
    const MTensor* dw = model_require(&model, "dense.weight");
    const MTensor* db = model_require(&model, "dense.bias");

    /* Shape assertions. The golden model froze these conventions; a silent
     * mismatch here would produce plausible-looking wrong numbers, which is
     * the failure mode this whole exercise exists to eliminate. */
    if (cw->n_elem != OUT_CH * IN_CH * KSIZE * KSIZE ||
        cb->n_elem != OUT_CH ||
        dw->n_elem != FLAT * NCLS ||
        db->n_elem != NCLS) {
        fprintf(stderr, "golden_c: model shapes do not match this architecture\n");
        model_free(&model);
        return 1;
    }

    /* Three test images, identical to export_weights.py's build_images() */
    float images[3][IN_CH * IMG * IMG];
    memset(images, 0, sizeof(images));
    for (int y = 4; y < 8; y++) for (int x = 0; x < 8; x++) images[0][y * IMG + x] = 10.0f;
    for (int y = 0; y < 8; y++) for (int x = 4; x < 8; x++) images[1][y * IMG + x] = 10.0f;
    for (int i = 0; i < IN_CH * IMG * IMG; i++)                images[2][i]       = 5.0f;

    float conv_out[OUT_CH * IMG * IMG];
    float pool_out[FLAT];
    float dense_out[NCLS];

    for (int i = 0; i < 3; i++) {
        emit(i, "input", images[i], IN_CH * IMG * IMG);

        conv2d(images[i], IN_CH, IMG, IMG, cw->data, cb->data, OUT_CH, conv_out);
        emit(i, "conv", conv_out, OUT_CH * IMG * IMG);

        relu(conv_out, OUT_CH * IMG * IMG);
        emit(i, "relu", conv_out, OUT_CH * IMG * IMG);

        maxpool2d(conv_out, OUT_CH, IMG, IMG, POOL, pool_out);
        emit(i, "pool", pool_out, FLAT);

        /* Flatten is a no-op on a contiguous buffer -- emitted anyway so the
         * layer names line up with NumPy's, which makes diffs readable. */
        emit(i, "flatten", pool_out, FLAT);

        dense(pool_out, FLAT, dw->data, db->data, NCLS, dense_out);
        emit(i, "dense", dense_out, NCLS);

        softmax_inplace(dense_out, NCLS);
        emit(i, "softmax", dense_out, NCLS);
    }

    model_free(&model);
    return 0;
}
