"""Corrupt files must be rejected normally by BOTH inference executables."""
import struct
import argparse
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_checks(binaries=None):
    binaries = binaries or [HERE / "golden_c", HERE / "golden_cpp"]
    original = (HERE / "weights.bin").read_bytes()

    def changed(offset, value):
        data = bytearray(original)
        struct.pack_into("<I", data, offset, value)
        return data

    cases = {
        "short header": original[:8],
        "truncated data": original[:-4],
        "bad magic": changed(0, 0),
        "future version": changed(4, 99),
        "no tensors": changed(8, 0),
        "too many tensors": changed(8, 17),
        "empty name": changed(16, 0),
        "unsupported dtype": changed(32, 1),
        "empty shape": changed(36, 0),
        "too many dimensions": changed(36, 5),
        "zero dimension": changed(40, 0),
        "overflowing dimension": changed(40, 0x80000002),
        "wrong byte count": changed(56, 4),
        "trailing bytes": original + b"junk",
        "duplicate tensor": changed(8, 5) + original[16:132],
    }
    wrong_shape = bytearray(original)
    struct.pack_into("<IIII", wrong_shape, 40, 1, 2, 3, 3)
    cases["wrong shape with same element count"] = wrong_shape
    with tempfile.TemporaryDirectory(prefix="cnn-model-test-") as directory:
        path = Path(directory) / "model.bin"
        for executable in binaries:
            path.write_bytes(original)
            result = subprocess.run([str(executable), str(path)], capture_output=True)
            if result.returncode != 0:
                raise AssertionError(f"{executable}: valid model rejected")
            for name, raw in cases.items():
                data = bytearray(raw)
                # Recompute checksum to exercise parsing, not just integrity.
                if len(data) >= 16:
                    struct.pack_into("<I", data, 12, sum(data[16:]) & 0xFFFFFFFF)
                path.write_bytes(data)
                result = subprocess.run([str(executable), str(path)], capture_output=True)
                if result.returncode != 1:
                    raise AssertionError(f"{executable}: {name}: expected exit 1, got {result.returncode}")
            data = bytearray(original)
            data[100] ^= 0xFF
            path.write_bytes(data)
            result = subprocess.run([str(executable), str(path)], capture_output=True)
            if result.returncode != 1:
                raise AssertionError(f"{executable}: checksum corruption not rejected")
            print(f"{executable.name}: valid model accepted; {len(cases) + 1} invalid models rejected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binaries", nargs="+", type=Path)
    args = parser.parse_args()
    run_checks([path.resolve() for path in args.binaries] if args.binaries else None)
