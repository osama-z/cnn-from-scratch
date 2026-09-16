# Binary tensor format, version 1

[Engine guide](../../engine/README.md) · [Documentation index](../README.md)

This format stores named float32 tensors. It does not encode a general network
graph; the inference runners define the layer sequence.

## Byte layout

All integer fields are unsigned 32-bit little-endian values. Float payloads are
little-endian IEEE binary32. Records begin at four-byte boundaries.

| Header field | Bytes | Meaning |
|---|---:|---|
| magic | 4 | `0x574E4E43`, bytes spelling `CNNW` |
| version | 4 | `1` |
| n_tensors | 4 | Number of records in the body |
| checksum | 4 | Sum of body bytes modulo 2^32 |

Each tensor record contains, in order:

| Field | Size |
|---|---|
| name_len | 4 bytes |
| name | name_len bytes; writer uses ASCII |
| name padding | 0–3 bytes to the next four-byte boundary |
| dtype | 4 bytes; 0 means float32 |
| ndim | 4 bytes |
| dims | ndim × 4 bytes |
| n_bytes | 4 bytes; must equal product(dims) × 4 |
| data | n_bytes bytes, contiguous row-major values |
| data padding | Padding to four bytes; float32 payloads already align |

The checksum catches some accidental corruption. Different changes can have the
same sum, so it is not authentication and cannot replace structural checks.

## Tensor conventions

| Name | Shape |
|---|---|
| `conv.weight` | `(output_channels, input_channels, kernel_height, kernel_width)` |
| `conv.bias` | `(output_channels,)` |
| `dense.weight` | `(input_features, output_features)` |
| `dense.bias` | `(output_features,)` |
| `input`, in a samples file | `(batch, channels, height, width)` |

For convolution weights the flat offset is:

```text
((output_channel × input_channels + input_channel) × kernel_height + ky)
    × kernel_width + kx
```

For Dense weights it is `input_index × output_features + output_index`.

## Writer, reader, and validation

- [engine/model_format.py](../../engine/model_format.py): `write_model()`.
- [engine/model_io.h](../../engine/model_io.h): `model_load()` and access helpers.
- [trained/run_common.h](../../trained/run_common.h): architecture-specific shape
  and finite-value validation.

The reader accepts 1–16 tensors, unique nonempty names without embedded NULs,
and 1–4 positive dimensions per tensor. It checks byte counts and overflow,
truncation, checksum, dtype/version, and trailing data.

The loaded payload is borrowed directly from the owned file buffer. Its use as
float pointers requires a little-endian IEEE binary32 host; unsupported hosts
are rejected. Keep the Model alive while using those pointers.

The writer is a serialization utility for trusted NumPy tensors, not a substitute
for reader validation. Malformed-file tests deliberately create invalid records.

## Two model sizes

The frozen [Lesson 7](../../lessons/day7/README.md) model contains 119 parameters:
476 payload bytes and a total file size of 636 bytes.

The default [trained CNN](../../trained/README.md) contains 7,890 parameters:
31,560 payload bytes and a total file size of 31,720 bytes.

The trained runners use same-padded odd square convolution kernels and 2×2
pooling, infer dimensions from metadata, and reject incompatible shapes.
