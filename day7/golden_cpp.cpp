/**
 * DAY 7, STEP 4: THE C++ IMPLEMENTATION, LOADING THE GOLDEN MODEL
 * ===============================================================
 *
 * The Day 6 layer classes, now loading weights from the golden model instead
 * of generating them. It includes the SAME model_io.h that golden_c.c uses --
 * so "C and C++ agree" is a statement about two inference engines, not about
 * two copies of a parser that might differ.
 *
 * The output format is byte-for-byte the same TENSOR blocks golden_c.c emits,
 * so verify.py treats them identically.
 */

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include "model_io.h"

constexpr int IN_CH = 1, OUT_CH = 2, KSIZE = 3;
constexpr int IMG = 8, POOL = 2, NCLS = 3;
constexpr int FLAT = OUT_CH * (IMG / POOL) * (IMG / POOL);   // 32

// ---- minimal Tensor: enough to carry a shape and a buffer ----
struct Tensor {
    int channels = 0, height = 0, width = 0;
    std::vector<float> data;
    Tensor() = default;
    Tensor(int c, int h, int w) : channels(c), height(h), width(w),
                                  data(static_cast<size_t>(c) * h * w, 0.0f) {}
    explicit Tensor(int n) : channels(1), height(1), width(n), data(n, 0.0f) {}
    int  size() const { return static_cast<int>(data.size()); }
    float&       operator()(int c,int y,int x)       { return data[c*(height*width)+y*width+x]; }
    const float& operator()(int c,int y,int x) const { return data[c*(height*width)+y*width+x]; }
    float&       operator[](int i)       { return data[i]; }
    const float& operator[](int i) const { return data[i]; }
};

struct Layer {
    std::string name;
    explicit Layer(std::string n) : name(std::move(n)) {}
    virtual Tensor forward(const Tensor&) = 0;
    virtual ~Layer() = default;
};

struct Conv2D : Layer {
    int in_ch, out_ch, k;
    const float* w;   // borrowed from the Model's blob -- not owned
    const float* b;
    Conv2D(int ic,int oc,int ks,const float* wt,const float* bs)
        : Layer("conv"), in_ch(ic), out_ch(oc), k(ks), w(wt), b(bs) {}
    Tensor forward(const Tensor& in) override {
        int H = in.height, W = in.width, kh = k/2;
        Tensor out(out_ch, H, W);
        for (int oc=0; oc<out_ch; ++oc)
          for (int y=0; y<H; ++y)
            for (int x=0; x<W; ++x) {
                float sum = b[oc];
                for (int ic=0; ic<in_ch; ++ic)
                  for (int ky=-kh; ky<=kh; ++ky)
                    for (int kx=-kh; kx<=kh; ++kx) {
                        int iy=y+ky, ix=x+kx;
                        if (iy>=0&&iy<H&&ix>=0&&ix<W) {
                            int wi = oc*(in_ch*k*k)+ic*(k*k)+(ky+kh)*k+(kx+kh);
                            sum += in(ic,iy,ix)*w[wi];
                        }
                    }
                out(oc,y,x)=sum;
            }
        return out;
    }
};

struct ReLU : Layer {
    ReLU() : Layer("relu") {}
    Tensor forward(const Tensor& in) override {
        Tensor out = in;
        for (int i=0;i<out.size();++i) if (out[i]<0.0f) out[i]=0.0f;
        return out;
    }
};

struct MaxPool2D : Layer {
    int p;
    explicit MaxPool2D(int ps) : Layer("pool"), p(ps) {}
    Tensor forward(const Tensor& in) override {
        int oh=in.height/p, ow=in.width/p;
        Tensor out(in.channels, oh, ow);
        for (int c=0;c<in.channels;++c)
          for (int y=0;y<oh;++y)
            for (int x=0;x<ow;++x) {
                float best = in(c,y*p,x*p);
                for (int py=0;py<p;++py)
                  for (int px=0;px<p;++px)
                    best = std::max(best, in(c,y*p+py,x*p+px));
                out(c,y,x)=best;
            }
        return out;
    }
};

struct Flatten : Layer {
    Flatten() : Layer("flatten") {}
    Tensor forward(const Tensor& in) override {
        Tensor out(in.size()); out.data = in.data; return out;
    }
};

struct Dense : Layer {
    int in_size, out_size;
    const float* w;
    const float* b;
    Dense(int in_n,int out_n,const float* wt,const float* bs)
        : Layer("dense"), in_size(in_n), out_size(out_n), w(wt), b(bs) {}
    Tensor forward(const Tensor& in) override {
        Tensor out(out_size);
        for (int j=0;j<out_size;++j) {
            float sum = b[j];
            for (int i=0;i<in_size;++i) sum += in[i]*w[i*out_size+j];
            out[j]=sum;
        }
        return out;
    }
};

struct Softmax : Layer {
    Softmax() : Layer("softmax") {}
    Tensor forward(const Tensor& in) override {
        Tensor out = in;
        float mx = *std::max_element(out.data.begin(), out.data.end());
        float sum=0.0f;
        for (int i=0;i<out.size();++i){ out[i]=std::exp(out[i]-mx); sum+=out[i]; }
        for (int i=0;i<out.size();++i) out[i]/=sum;
        return out;
    }
};

static void emit(int img, const std::string& layer, const Tensor& t) {
    std::printf("TENSOR %d %s %d\n", img, layer.c_str(), t.size());
    for (int i=0;i<t.size();++i) std::printf("%.9g\n", t[i]);
}

int main(int argc, char** argv) {
    const char* path = (argc>1) ? argv[1] : "weights.bin";

    Model model;
    if (model_load(&model, path) != 0) return 1;

    const MTensor* cw = model_require(&model, "conv.weight");
    const MTensor* cb = model_require(&model, "conv.bias");
    const MTensor* dw = model_require(&model, "dense.weight");
    const MTensor* db = model_require(&model, "dense.bias");

    if (cw->n_elem != OUT_CH*IN_CH*KSIZE*KSIZE || cb->n_elem != OUT_CH ||
        dw->n_elem != FLAT*NCLS || db->n_elem != NCLS) {
        std::fprintf(stderr, "golden_cpp: model shapes do not match architecture\n");
        model_free(&model);
        return 1;
    }

    // Layers hold BORROWED pointers into model.blob, so the model must outlive
    // them. unique_ptr still owns the Layer objects themselves -- RAII frees
    // those; model_free frees the weight blob. Two distinct ownerships, neither
    // leaked. (Verified under ASan in the Makefile.)
    std::vector<std::unique_ptr<Layer>> net;
    net.push_back(std::make_unique<Conv2D>(IN_CH, OUT_CH, KSIZE, cw->data, cb->data));
    net.push_back(std::make_unique<ReLU>());
    net.push_back(std::make_unique<MaxPool2D>(POOL));
    net.push_back(std::make_unique<Flatten>());
    net.push_back(std::make_unique<Dense>(FLAT, NCLS, dw->data, db->data));
    net.push_back(std::make_unique<Softmax>());

    // Three test images, matching export_weights.py exactly.
    std::vector<Tensor> images(3, Tensor(IN_CH, IMG, IMG));
    for (int y=4;y<8;++y) for (int x=0;x<8;++x) images[0](0,y,x)=10.0f;
    for (int y=0;y<8;++y) for (int x=4;x<8;++x) images[1](0,y,x)=10.0f;
    for (int i=0;i<images[2].size();++i) images[2][i]=5.0f;

    for (int i=0;i<3;++i) {
        emit(i, "input", images[i]);
        Tensor x = images[i];
        for (auto& layer : net) {
            x = layer->forward(x);
            emit(i, layer->name, x);
        }
    }

    model_free(&model);
    return 0;   // unique_ptrs unwind here
}
