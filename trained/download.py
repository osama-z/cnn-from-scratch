"""Download the four MNIST archives and verify their SHA-256 checksums."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

from .verify import ROOT

ARCHIVES = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "lessons/day3/mnist_data")
    args = parser.parse_args()
    args.data.mkdir(parents=True, exist_ok=True)
    for name, expected in ARCHIVES.items():
        path = args.data / name
        if path.exists():
            raw = path.read_bytes()
        else:
            with urllib.request.urlopen("https://ossci-datasets.s3.amazonaws.com/mnist/" + name, timeout=60) as response:
                raw = response.read()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise RuntimeError(f"checksum mismatch: {path}; existing files are not overwritten")
        if not path.exists():
            path.write_bytes(raw)
        print(f"Verified {name}")


if __name__ == "__main__":
    main()
