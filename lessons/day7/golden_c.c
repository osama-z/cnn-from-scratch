/**
 * DAY 7, STEP 3: THE C IMPLEMENTATION, LOADING THE GOLDEN MODEL
 * =============================================================
 *
 * This is lessons/day4/cnn_forward.c with exactly one thing changed: it no longer
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
#include "../../engine/model_io.h"

#define IN_CH   1
#define OUT_CH  2
#define KSIZE   3
#define IMG     8
#define POOL    2
#define NCLS    3
#define FLAT    (OUT_CH * (IMG / POOL) * (IMG / POOL))   /* 32 */
#define NIMG    4                                        /* 3 originals + stability probe */

/* ---------------------------------------------------------------- */
/* Layers -- identical math to lessons/day4/cnn_forward.c                    */
/* ---------------------------------------------------------------- */

#include "../../engine/cnn_c.h"

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
    const uint32_t cw_shape[] = {OUT_CH, IN_CH, KSIZE, KSIZE};
    const uint32_t cb_shape[] = {OUT_CH}, dw_shape[] = {FLAT, NCLS}, db_shape[] = {NCLS};
    if (!model_has_shape(cw, 4, cw_shape) || !model_has_shape(cb, 1, cb_shape) ||
        !model_has_shape(dw, 2, dw_shape) || !model_has_shape(db, 1, db_shape)) {
        fprintf(stderr, "golden_c: model shapes do not match this architecture\n");
        model_free(&model);
        return 1;
    }

    /* Test images, identical to export_weights.py's build_images().
     * Image 3 is the softmax stability probe: amplitude 60 drives the largest
     * logit to ~100, and exp(100) overflows float32 unless the max is
     * subtracted first. See the comment in export_weights.py. */
    float images[NIMG][IN_CH * IMG * IMG];
    memset(images, 0, sizeof(images));
    for (int y = 4; y < 8; y++) for (int x = 0; x < 8; x++) images[0][y * IMG + x] = 10.0f;
    for (int y = 0; y < 8; y++) for (int x = 4; x < 8; x++) images[1][y * IMG + x] = 10.0f;
    for (int i = 0; i < IN_CH * IMG * IMG; i++)                images[2][i]       = 5.0f;
    for (int y = 4; y < 8; y++) for (int x = 0; x < 8; x++) images[3][y * IMG + x] = 60.0f;

    float conv_out[OUT_CH * IMG * IMG];
    float pool_out[FLAT];
    float dense_out[NCLS];

    for (int i = 0; i < NIMG; i++) {
        emit(i, "input", images[i], IN_CH * IMG * IMG);

        conv2d(images[i], IN_CH, IMG, IMG, cw->data, cb->data, OUT_CH, conv_out, KSIZE);
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
