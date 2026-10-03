#!/usr/bin/env python3
"""
validate.py — Language-agnostic mel spectrogram validation harness.

Usage:
    uv run python validate.py <program_path> [--runs N]

Examples:
    uv run python validate.py ref/python/melspec.py
    uv run python validate.py ./my_melspec_binary
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

TEST_CASES = [
    {
        "name": "default",
        "params": {"sr": 44100, "n_fft": 2048, "hop_length": 512,
                   "n_mels": 128, "f_min": 0.0, "f_max": 22050.0},
        "expected": "data/expected/default.bin",
        "shape": (128, 22099),
    },
    {
        "name": "small_mels",
        "params": {"sr": 44100, "n_fft": 2048, "hop_length": 512,
                   "n_mels": 32, "f_min": 0.0, "f_max": 22050.0},
        "expected": "data/expected/small_mels.bin",
        "shape": (32, 22099),
    },
    {
        "name": "large_fft",
        "params": {"sr": 44100, "n_fft": 4096, "hop_length": 512,
                   "n_mels": 128, "f_min": 0.0, "f_max": 22050.0},
        "expected": "data/expected/large_fft.bin",
        "shape": (128, 22095),
    },
    {
        "name": "narrow_band",
        "params": {"sr": 44100, "n_fft": 2048, "hop_length": 512,
                   "n_mels": 128, "f_min": 300.0, "f_max": 8000.0},
        "expected": "data/expected/narrow_band.bin",
        "shape": (128, 22099),
    },
]


def build_args(params):
    """Convert params dict to CLI argument list."""
    args = []
    for key, value in params.items():
        args.extend([f"--{key}", str(value)])
    return args


def build_librosa_kwargs(params):
    """Convert params to librosa keyword arguments (fmin/fmax not f_min/f_max)."""
    return {
        "n_fft": params["n_fft"],
        "hop_length": params["hop_length"],
        "n_mels": params["n_mels"],
        "fmin": params["f_min"],
        "fmax": params["f_max"],
    }


def run_and_time(cmd, runs=3):
    """
    Run a command multiple times, return (min_time_seconds, output_bytes_from_first_run).

    cmd must have a None placeholder at the output_bin position.
    Uses a temp file for output; cleans up in finally block.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".bin")
    os.close(tmp_fd)

    # Replace None placeholder with the actual temp file path
    full_cmd = [tmp_path if part is None else part for part in cmd]

    out_bytes = None
    times = []
    try:
        for i in range(runs):
            t0 = time.perf_counter()
            subprocess.run(full_cmd, check=True, capture_output=True)
            t1 = time.perf_counter()
            times.append(t1 - t0)

            if i == 0:
                with open(tmp_path, "rb") as f:
                    out_bytes = f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return min(times), out_bytes


def time_librosa(params, runs=3):
    """
    Time librosa's mel spectrogram computation.

    Returns min wall-clock time over `runs` runs, or None if librosa is unavailable.
    """
    import librosa

    y = np.fromfile("data/audio.pcm", dtype="<i2").astype(np.float32) / 32768.0
    librosa_kwargs = build_librosa_kwargs(params)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        librosa.feature.melspectrogram(y=y, sr=44100, center=False, htk=True, **librosa_kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return min(times)


def validate_program(program_path, runs=3, rtol=1e-3, atol=1e-4):
    """
    Validate a mel spectrogram program against all 4 test cases.

    Returns list of result dicts.
    """
    is_python = program_path.endswith(".py")
    results = []

    skip_librosa = False
    try:
        import librosa
    except ImportError:
        print(f"Warning: Librosa not available, skipping librosa timing.")
        skip_librosa = True

    for tc in TEST_CASES:
        print(f"Running test case: {tc['name']}...")
        name = tc["name"]
        params = tc["params"]
        expected_path = tc["expected"]
        shape = tc["shape"]

        # Build command: Python scripts use 'uv run python', binaries run directly
        if is_python:
            cmd = [sys.executable, program_path, "data/audio.pcm", None] + build_args(params)
        else:
            cmd = [program_path, "data/audio.pcm", None] + build_args(params)

        prog_time, out_bytes = run_and_time(cmd, runs)

        # Load program output
        result_arr = np.frombuffer(out_bytes, dtype="<f4").reshape(shape)

        # Load expected ground truth
        expected_arr = np.fromfile(expected_path, dtype="<f4").reshape(shape)

        # Compute error metrics
        max_abs_err = np.max(np.abs(result_arr - expected_arr))
        passed = bool(np.allclose(result_arr, expected_arr, rtol=rtol, atol=atol))

        # Time librosa for speedup comparison
        if not skip_librosa:
            librosa_time = time_librosa(params, runs)
        else:
            librosa_time = None
        if librosa_time is not None:
            speedup = librosa_time / prog_time
        else:
            speedup = None

        results.append({
            "name": name,
            "params": params,
            "max_abs_err": max_abs_err,
            "passed": passed,
            "prog_time": prog_time,
            "librosa_time": librosa_time,
            "speedup": speedup,
        })

    return results


def print_results(results):
    """Print results as a formatted table."""
    # Header
    header = (
        f"{'Test Case':<16}| {'n_fft':>5} | {'hop':>4} | {'mels':>4} | "
        f"{'f_min':>6} | {'f_max':>6} | {'Max Abs Err':>11} | {'Status':>6} | "
        f"{'Time (s)':>8} | {'Speedup':>7}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    passed_count = 0
    for r in results:
        p = r["params"]
        status = "PASS" if r["passed"] else "FAIL"
        if r["passed"]:
            passed_count += 1

        speedup_str = f"{r['speedup']:.3f}x" if r["speedup"] is not None else "N/A"

        row = (
            f"{r['name']:<16}| {p['n_fft']:>5} | {p['hop_length']:>4} | {p['n_mels']:>4} | "
            f"{p['f_min']:>6.0f} | {p['f_max']:>6.0f} | {r['max_abs_err']:>11.2e} | "
            f"{status:>6} | {r['prog_time']:>8.3f} | {speedup_str:>7}"
        )
        print(row)

    print(sep)
    print(f"\nResult: {passed_count}/{len(results)} PASSED")


def main():
    parser = argparse.ArgumentParser(
        description="Validate a mel spectrogram program against ground-truth binaries."
    )
    parser.add_argument(
        "program_path",
        help="Path to the program to validate (Python script or compiled binary)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of timing runs per test case (default: 3)",
    )
    args = parser.parse_args()

    results = validate_program(args.program_path, runs=args.runs)
    print_results(results)


if __name__ == "__main__":
    main()
