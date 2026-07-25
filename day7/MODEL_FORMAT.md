# `.bin` Model Format — Specification v1

> A hand-rolled binary format for storing a neural network's weights.
> This is a miniature of what ONNX and TFLite do. Writing one by hand first
> means those formats will look obvious in Month 3 instead of magical.

All integers are **little-endian unsigned 32-bit**. All offsets are **4-byte aligned**.

---

## Layout

```text
┌──────────────────────────────────────────────┐
│ HEADER (16 bytes)                             │
├──────────────┬───────────────────────────────┤
│ u32 magic    │ 0x574E4E43  ('C','N','N','W')  │
│ u32 version  │ 1                              │
│ u32 n_tensors│ number of tensor records       │
│ u32 checksum │ sum of all body bytes mod 2^32 │
├──────────────┴───────────────────────────────┤
│ BODY: n_tensors records, back to back         │
│                                               │
│  ┌─────────────────────────────────────────┐ │
│  │ u32  name_len                           │ │
│  │ u8   name[name_len]                     │ │
│  │ u8   pad to 4-byte boundary             │ │
│  │ u32  dtype        (0 = float32)         │ │
│  │ u32  ndim                               │ │
│  │ u32  dims[ndim]                         │ │
│  │ u32  n_bytes      (= product(dims) * 4) │ │
│  │ u8   data[n_bytes]                      │ │
│  │ u8   pad to 4-byte boundary             │ │
│  └─────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
```

---

## Field notes — why each piece exists

**`magic`** — the first four bytes. If a loader is handed a JPEG, a truncated
download, or a file from a future format, it fails on byte 0 with a clear
message instead of interpreting random bytes as tensor dimensions and allocating
gigabytes. Every real format starts with one (`\x89PNG`, `PK` for zip,
`TFL3` for TFLite).

**`version`** — lets a v2 loader refuse a v1 file, or handle it deliberately.
The one field you always regret omitting.

**`checksum`** — a plain byte sum. Not cryptographic; it catches the failure
mode that actually happens on embedded hardware: a flash write interrupted by a
brownout, leaving a file that still parses and then predicts garbage. Cheap
insurance against silent corruption.

**4-byte alignment** — every field before a tensor's `data` is a multiple of 4
bytes, so `data` always starts 4-aligned. The loader then points a
`const float*` directly into the buffer — **zero-copy**. On x86 an unaligned
float load is just slow; on some ARM cores it faults. Formats meant to be
`mmap`'d on embedded targets (TFLite via FlatBuffers) are strict about this for
exactly this reason.

**little-endian, stated explicitly** — x86 and ARM are both little-endian, so
this is free today. But a format that does not name its byte order is a bug
waiting for the first big-endian target. The loader reads LE by explicit byte
assembly, so it is correct regardless of host endianness.

---

## Tensor layout conventions

The format stores raw numbers; these conventions say what they mean. Every
implementation (NumPy, C, C++, VHDL) obeys them:

| Tensor | Shape | Flat index |
|---|---|---|
| `conv.weight` | `(out_ch, in_ch, kh, kw)` | `oc*(in_ch*9) + ic*9 + ky*3 + kx` |
| `conv.bias` | `(out_ch,)` | `oc` |
| `dense.weight` | `(in_size, out_size)` | `i*out_size + j` |
| `dense.bias` | `(out_size,)` | `j` |

`dense.weight` is row-major `(in, out)` because that is what `day4`'s
`weights[i * out_size + j]` already assumed. The golden model **freezes an
existing convention**; it does not invent a new one.

---

## Reader / writer

- **Writer:** `export_weights.py` → `write_model()`
- **Reader:** `model_io.h` → `model_load()` (header-only C, also valid C++; both
  `golden_c.c` and `golden_cpp.cpp` include it, so they provably parse identical
  bytes)

The current model is 636 bytes: a 16-byte header + 4 tensors (119 floats).
