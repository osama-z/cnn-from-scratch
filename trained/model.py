"""Conv(same) -> ReLU -> MaxPool(2) -> Dense -> Softmax, in NCHW layout.

All derivatives are explicit. NumPy handles array operations; no autograd or
neural-network framework is used. Training and inference share these weights.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def conv_forward(x, w, b):
    k = w.shape[-1]
    pad = k // 2
    padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    windows = sliding_window_view(padded, (k, k), axis=(2, 3))
    out = np.einsum("nchwij,ocij->nohw", windows, w, optimize=True)
    return out + b[None, :, None, None], (x.shape, windows, w)


def conv_backward(dout, cache):
    shape, windows, w = cache
    k = w.shape[-1]
    pad = k // 2
    h, width = shape[2:]
    dw = np.einsum("nohw,nchwij->ocij", dout, windows, optimize=True)
    db = dout.sum(axis=(0, 2, 3))
    dx_pad = np.zeros((shape[0], shape[1], h + 2 * pad, width + 2 * pad), dtype=dout.dtype)
    for ky in range(k):
        for kx in range(k):
            dx_pad[:, :, ky:ky+h, kx:kx+width] += np.einsum(
                "nohw,oc->nchw", dout, w[:, :, ky, kx], optimize=True)
    return dx_pad[:, :, pad:pad+h, pad:pad+width], dw, db


def pool_forward(x):
    n, c, h, w = x.shape
    oh, ow = h // 2, w // 2
    blocks = x[:, :, :2*oh, :2*ow].reshape(n, c, oh, 2, ow, 2)
    blocks = blocks.transpose(0, 1, 2, 4, 3, 5).reshape(n, c, oh, ow, 4)
    indices = blocks.argmax(axis=-1)  # First maximum wins ties consistently.
    return blocks.max(axis=-1), (x.shape, indices)


def pool_backward(dout, cache):
    shape, indices = cache
    dx = np.zeros(shape, dtype=dout.dtype)
    oh, ow = dout.shape[2:]
    for p in range(4):
        dx[:, :, p//2:2*oh:2, p%2:2*ow:2] = dout * (indices == p)
    return dx


class CNN:
    def __init__(self, height=28, width=28, filters=4, classes=10, seed=42, dtype=np.float32):
        if min(height, width, filters, classes) <= 0 or min(height, width) < 2:
            raise ValueError("positive dimensions and images of at least 2x2 required")
        self.height, self.width = height, width
        rng = np.random.default_rng(seed)
        flat = filters * (height // 2) * (width // 2)
        self.params = {
            "conv.weight": (rng.normal(size=(filters, 1, 3, 3)) * np.sqrt(2/9)).astype(dtype),
            "conv.bias": np.zeros(filters, dtype=dtype),
            "dense.weight": (rng.normal(size=(flat, classes)) * np.sqrt(1/flat)).astype(dtype),
            "dense.bias": np.zeros(classes, dtype=dtype),
        }

    def forward(self, x, trace=False):
        if x.ndim != 4 or x.shape[1:] != (1, self.height, self.width) or len(x) == 0:
            raise ValueError("input must have shape (N, 1, height, width)")
        p = self.params
        conv, conv_cache = conv_forward(x, p["conv.weight"], p["conv.bias"])
        relu = np.maximum(conv, 0)
        pool, pool_cache = pool_forward(relu)
        flat = pool.reshape(len(x), -1)
        logits = flat @ p["dense.weight"] + p["dense.bias"]
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        self.cache = (conv, conv_cache, pool_cache, flat, logits, probs)
        if trace:
            return {"input": x, "conv": conv, "relu": relu, "pool": pool,
                    "flatten": flat, "dense": logits, "softmax": probs}
        return probs

    def loss_and_grads(self, x, labels):
        labels = np.asarray(labels)
        if labels.shape != (len(x),) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("one integer label per image required")
        classes = self.params["dense.bias"].size
        if np.any(labels < 0) or np.any(labels >= classes):
            raise ValueError("label out of range")
        probs = self.forward(x)
        conv, conv_cache, pool_cache, flat, logits, _ = self.cache
        shifted = logits - logits.max(axis=1, keepdims=True)
        loss = np.mean(np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(x)), labels])
        dlogits = probs.copy()
        dlogits[np.arange(len(x)), labels] -= 1
        dlogits /= len(x)
        dflat = dlogits @ self.params["dense.weight"].T
        dpool = dflat.reshape(pool_cache[1].shape)
        dconv = pool_backward(dpool, pool_cache) * (conv > 0)
        dx, dw, db = conv_backward(dconv, conv_cache)
        grads = {"conv.weight": dw, "conv.bias": db,
                 "dense.weight": flat.T @ dlogits, "dense.bias": dlogits.sum(axis=0)}
        return float(loss), grads, dx

    def save(self, path):
        np.savez(path, height=self.height, width=self.width, **self.params)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            model = cls(int(data["height"]), int(data["width"]),
                        data["conv.bias"].size, data["dense.bias"].size)
            for key, expected in model.params.items():
                value = data[key]
                if value.shape != expected.shape or not np.isfinite(value).all():
                    raise ValueError(f"invalid checkpoint tensor: {key}")
                model.params[key] = value.astype(np.float32)
        return model


class Adam:
    def __init__(self, params, lr=0.001):
        self.lr, self.step = lr, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def update(self, params, grads):
        self.step += 1
        for key in params:
            g = grads[key]
            self.m[key] = 0.9 * self.m[key] + 0.1 * g
            self.v[key] = 0.999 * self.v[key] + 0.001 * g * g
            m = self.m[key] / (1 - 0.9**self.step)
            v = self.v[key] / (1 - 0.999**self.step)
            params[key] -= self.lr * m / (np.sqrt(v) + 1e-8)
