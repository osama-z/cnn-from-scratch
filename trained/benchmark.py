"""Benchmark the verified deployed model, excluding file loading and printing."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import numpy as np

from .verify import ROOT, verify_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "build/trained")
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 10 or args.trials < 1:
        parser.error("at least 10 repeats and one trial required")
    # Rebuild so the recorded commands describe the binaries actually timed.
    subprocess.run(["make", "-B", "-C", str(ROOT / "trained"), "all"], check=True)
    verify_bundle(args.out)
    report = {"platform": platform.platform(), "machine": platform.machine(),
              "processor": platform.processor(), "trials": args.trials,
              "weights_sha256": hashlib.sha256((args.out / "weights.bin").read_bytes()).hexdigest(),
              "timing": "CLOCK_MONOTONIC; 10 warm-ups; excludes model/input loading and printing",
              "c_scope": "preallocated activation buffers; scalar float inference",
              "cpp_scope": "input copy and Tensor allocations included; scalar float inference",
              "results": {}}
    with np.load(args.out / "samples.npz", allow_pickle=False) as data:
        report["sample_shape"] = list(data["images"].shape)
    with np.load(args.out / "checkpoint.npz", allow_pickle=False) as data:
        report["parameter_payload_bytes"] = sum(data[key].nbytes for key in (
            "conv.weight", "conv.bias", "dense.weight", "dense.bias"))
    report["model_file_bytes"] = (args.out / "weights.bin").stat().st_size
    if Path("/proc/cpuinfo").exists():
        report["cpu_model"] = next((line.split(":", 1)[1].strip()
                                   for line in Path("/proc/cpuinfo").read_text().splitlines()
                                   if line.startswith("model name")), "unknown")
    for tool in ("gcc", "g++"):
        if shutil.which(tool):
            report[tool] = subprocess.check_output([tool, "--version"], text=True).splitlines()[0]
    report["build_commands"] = subprocess.check_output(
        ["make", "-B", "-n", "-C", str(ROOT / "trained"), "all"], text=True).splitlines()
    for name in ("infer_c", "infer_cpp"):
        binary = ROOT / "trained" / name
        trials = []
        for _ in range(args.trials):
            raw = subprocess.check_output([str(binary), str(args.out / "weights.bin"),
                                           str(args.out / "samples.bin"), "--benchmark", str(args.repeats)], text=True)
            trials.append(json.loads(raw))
        median = statistics.median(t["latency_us"] for t in trials)
        report["results"][name] = {"median_latency_us": median, "median_fps": 1e6 / median,
                                    "trials": trials,
                                    "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
        print(f"{name}: median {median:.2f} us/image ({1e6/median:.0f} images/s)")
    (args.out / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
