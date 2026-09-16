/* Shared scalar float kernels. Callers validate shapes and own the buffers. */
#ifndef CNN_C_H
#define CNN_C_H
#include <math.h>

static inline void conv2d(const float* in, int in_ch, int H, int W,
                   const float* kern, const float* bias, int out_ch,
                   float* out, int ksize)
{
    const int kh = ksize / 2;
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
                                int k_idx  = oc * (in_ch * ksize * ksize)
                                           + ic * (ksize * ksize)
                                           + (ky + kh) * ksize + (kx + kh);
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

static inline void relu(float* d, int n) {
    for (int i = 0; i < n; i++) if (d[i] < 0.0f) d[i] = 0.0f;
}

static inline void maxpool2d(const float* in, int ch, int H, int W, int p, float* out) {
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

static inline void dense(const float* in, int in_size,
                  const float* w, const float* b, int out_size, float* out)
{
    for (int j = 0; j < out_size; j++) {
        float sum = b[j];
        for (int i = 0; i < in_size; i++)
            sum += in[i] * w[i * out_size + j];   /* layout: (in_size, out_size) */
        out[j] = sum;
    }
}

static inline void softmax_inplace(float* d, int n) {
    float mx = d[0];
    for (int i = 1; i < n; i++) if (d[i] > mx) mx = d[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { d[i] = expf(d[i] - mx); sum += d[i]; }
    for (int i = 0; i < n; i++) d[i] /= sum;
}

#endif
