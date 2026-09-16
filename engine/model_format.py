"""Version 1 tensor-file writer shared by the lessons and trained model.

See docs/reference/model-format.md for the byte layout and loader limits.
"""
from pathlib import Path
import struct
import numpy as np

MAGIC   = 0x574E4E43        # bytes 'C','N','N','W' read as little-endian u32
VERSION = 1
DTYPE_FLOAT32 = 0

# Explicit little-endian fields make the file layout independent of the writer's host.
U32 = "<I"


def _pad4(n: int) -> int:
    """Bytes needed to round n up to a 4-byte boundary."""
    return (-n) & 3


def write_model(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Serialize named tensors as contiguous little-endian float32 arrays.

    Padding keeps payload offsets aligned to four bytes. The native reader
    validates the file and host representation before borrowing float pointers.
    """
    body = bytearray()

    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr, dtype="<f4")   # force LE float32
        raw = arr.tobytes()
        nb  = name.encode("ascii")

        body += struct.pack(U32, len(nb))
        body += nb
        body += b"\x00" * _pad4(len(nb))               # realign after the name

        body += struct.pack(U32, DTYPE_FLOAT32)
        body += struct.pack(U32, arr.ndim)
        for d in arr.shape:
            body += struct.pack(U32, d)

        body += struct.pack(U32, len(raw))
        body += raw
        body += b"\x00" * _pad4(len(raw))              # float32 is already 4-aligned

    # Detect some accidental corruption; this byte sum is not authentication.
    checksum = sum(body) & 0xFFFFFFFF

    header = struct.pack(U32, MAGIC)
    header += struct.pack(U32, VERSION)
    header += struct.pack(U32, len(tensors))
    header += struct.pack(U32, checksum)               # header is exactly 16 bytes

    path.write_bytes(header + bytes(body))
    print(f"  wrote {path.name}: {len(header) + len(body)} bytes, "
          f"{len(tensors)} tensors, checksum 0x{checksum:08X}")
