#!/usr/bin/env python3
"""
Feature-by-feature benchmark comparison between rsona and librosa.

This script measures individual feature extraction performance to identify
where performance differences occur.

Usage:
    python feature_benchmark.py [audio_file] [--iterations N]
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import librosa
import librosa.feature.rhythm
import numpy as np


def time_librosa_features(audio_path: str) -> Dict[str, float]:
    """Benchmark individual librosa features."""
    y, sr = librosa.load(audio_path, mono=True, sr=None)

    timings = {}

    # STFT
    t0 = time.perf_counter()
    S = librosa.stft(y, n_fft=2048, hop_length=512)
    timings["stft"] = time.perf_counter() - t0

    # Mel Spectrogram
    t0 = time.perf_counter()
    mel = librosa.feature.melspectrogram(
        S=np.abs(S) ** 2, sr=sr, n_fft=2048, hop_length=512
    )
    timings["mel"] = time.perf_counter() - t0

    # MFCC
    t0 = time.perf_counter()
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel))
    timings["mfcc"] = time.perf_counter() - t0

    # RMS
    t0 = time.perf_counter()
    rms = librosa.feature.rms(y=y)
    timings["rms"] = time.perf_counter() - t0

    # Onset Strength
    t0 = time.perf_counter()
    onset = librosa.onset.onset_strength(S=mel, sr=sr)
    timings["onset"] = time.perf_counter() - t0

    # Tempo
    t0 = time.perf_counter()
    tempo = librosa.feature.rhythm.tempo(onset_envelope=onset, sr=sr)[0]
    timings["tempo"] = time.perf_counter() - t0

    # Chroma
    t0 = time.perf_counter()
    chroma = librosa.feature.chroma_stft(S=np.abs(S), sr=sr)
    timings["chroma"] = time.perf_counter() - t0

    # Spectral Centroid
    t0 = time.perf_counter()
    centroid = librosa.feature.spectral_centroid(S=np.abs(S), sr=sr)
    timings["spectral_centroid"] = time.perf_counter() - t0

    # Zero Crossing Rate
    t0 = time.perf_counter()
    zcr = librosa.feature.zero_crossing_rate(y)
    timings["zcr"] = time.perf_counter() - t0

    # Total pipeline
    t0 = time.perf_counter()
    _ = librosa.stft(y, n_fft=2048, hop_length=512)
    S_full = librosa.stft(y, n_fft=2048, hop_length=512)
    mel_full = librosa.feature.melspectrogram(
        S=np.abs(S_full) ** 2, sr=sr, n_fft=2048, hop_length=512
    )
    _ = librosa.feature.mfcc(S=librosa.power_to_db(mel_full))
    onset_full = librosa.onset.onset_strength(S=mel_full, sr=sr)
    _ = librosa.feature.rhythm.tempo(onset_envelope=onset_full, sr=sr)[0]
    timings["total"] = time.perf_counter() - t0

    return timings


def time_rsona_features(audio_path: str, bench_dir: Path) -> Dict[str, float]:
    """Benchmark individual rsona features."""
    result = subprocess.run(
        [
            "cargo",
            "run",
            "--release",
            "--bin",
            "rsona_bench",
            "--",
            audio_path,
            "--detailed",
        ],
        cwd=bench_dir.parent,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"rsona benchmark failed: {result.stderr}")

    # Parse JSON output
    for line in result.stdout.strip().split("\n"):
        try:
            data = json.loads(line)
            if "timings" in data:
                # Convert ms to seconds
                timings = {
                    "load": data["timings"]["load_ms"] / 1000.0,
                    "frame": data["timings"]["frame_ms"] / 1000.0,
                    "stft": data["timings"]["stft_ms"] / 1000.0,
                    "mel": data["timings"]["mel_ms"] / 1000.0,
                    "mfcc": data["timings"]["mfcc_ms"] / 1000.0,
                    "rms": data["timings"]["rms_ms"] / 1000.0,
                    "onset": data["timings"]["onset_ms"] / 1000.0,
                    "tempo": data["timings"]["tempo_ms"] / 1000.0,
                    "total": data["time_sec"],
                }
                return timings
        except (json.JSONDecodeError, KeyError):
            continue

    raise RuntimeError("Could not parse rsona benchmark output")


def run_feature_benchmarks(
    audio_path: str, iterations: int, bench_dir: Path, warmup: int = 1
) -> Dict[str, List[Dict[str, float]]]:
    """Run feature benchmarks multiple times."""
    print(f"Running feature benchmarks on: {audio_path}")
    print(f"Iterations: {iterations} (+ {warmup} warmup)\n")

    # Warmup
    if warmup > 0:
        print(f"Warmup ({warmup} iteration(s))...", end=" ", flush=True)
        for _ in range(warmup):
            try:
                time_librosa_features(audio_path)
                time_rsona_features(audio_path, bench_dir)
            except Exception:
                pass
        print("done\n")

    librosa_results = []
    rsona_results = []

    for i in range(iterations):
        print(f"Iteration {i + 1}/{iterations}...", end=" ", flush=True)

        # Run librosa
        try:
            lib_timings = time_librosa_features(audio_path)
            librosa_results.append(lib_timings)
            print(f"librosa: {lib_timings['total']:.4f}s", end=" | ")
        except Exception as e:
            print(f"librosa failed: {e}")
            continue

        # Run rsona
        try:
            rs_timings = time_rsona_features(audio_path, bench_dir)
            rsona_results.append(rs_timings)
            print(f"rsona: {rs_timings['total']:.4f}s")
        except Exception as e:
            print(f"rsona failed: {e}")
            continue

    print()
    return {
        "librosa": librosa_results,
        "rsona": rsona_results,
    }


def analyze_feature_results(
    results: Dict[str, List[Dict[str, float]]],
) -> Dict[str, Dict[str, Any]]:
    """Analyze feature benchmark results."""
    # Get common features
    lib_features = set(results["librosa"][0].keys()) if results["librosa"] else set()
    rs_features = set(results["rsona"][0].keys()) if results["rsona"] else set()

    # Map rsona features to librosa features for comparison
    feature_map = {
        "stft": "stft",
        "mel": "mel",
        "mfcc": "mfcc",
        "rms": "rms",
        "onset": "onset",
        "tempo": "tempo",
        "total": "total",
    }

    analysis = {}

    for rs_name, lib_name in feature_map.items():
        if lib_name in lib_features and rs_name in rs_features:
            lib_times = [r[lib_name] for r in results["librosa"]]
            rs_times = [r[rs_name] for r in results["rsona"]]

            lib_mean = statistics.mean(lib_times)
            rs_mean = statistics.mean(rs_times)
            speedup = lib_mean / rs_mean if rs_mean > 0 else 0

            analysis[lib_name] = {
                "librosa": {
                    "mean": lib_mean,
                    "median": statistics.median(lib_times),
                    "stdev": statistics.stdev(lib_times) if len(lib_times) > 1 else 0.0,
                },
                "rsona": {
                    "mean": rs_mean,
                    "median": statistics.median(rs_times),
                    "stdev": statistics.stdev(rs_times) if len(rs_times) > 1 else 0.0,
                },
                "speedup": speedup,
            }

    # Add librosa-only features
    for feat in ["chroma", "spectral_centroid", "zcr"]:
        if feat in lib_features:
            lib_times = [r[feat] for r in results["librosa"]]
            analysis[feat] = {
                "librosa": {
                    "mean": statistics.mean(lib_times),
                    "median": statistics.median(lib_times),
                    "stdev": statistics.stdev(lib_times) if len(lib_times) > 1 else 0.0,
                },
                "rsona": None,
                "speedup": None,
            }

    return analysis


def format_feature_report(analysis: Dict[str, Dict[str, Any]], audio_path: str) -> str:
    """Generate markdown report."""
    report = []
    report.append("# Feature-by-Feature Benchmark: rsona vs librosa")
    report.append("")
    report.append(f"**Audio File:** `{Path(audio_path).name}`")
    report.append("")

    # Performance comparison table
    report.append("## Performance Comparison")
    report.append("")
    report.append("| Feature | librosa (ms) | rsona (ms) | Speedup |")
    report.append("|---------|--------------|------------|---------|")

    feature_order = ["stft", "mel", "mfcc", "rms", "onset", "tempo", "total"]

    for feat in feature_order:
        if feat in analysis:
            data = analysis[feat]
            lib_ms = data["librosa"]["mean"] * 1000

            if data["rsona"]:
                rs_ms = data["rsona"]["mean"] * 1000
                speedup = data["speedup"]
                report.append(
                    f"| {feat.upper()} | {lib_ms:.2f} | {rs_ms:.2f} | **{speedup:.2f}x** |"
                )
            else:
                report.append(f"| {feat.upper()} | {lib_ms:.2f} | - | - |")

    report.append("")

    # Librosa-only features
    librosa_only = ["chroma", "spectral_centroid", "zcr"]
    if any(feat in analysis for feat in librosa_only):
        report.append("## Additional librosa Features")
        report.append("")
        report.append("| Feature | Time (ms) |")
        report.append("|---------|-----------|")

        for feat in librosa_only:
            if feat in analysis:
                lib_ms = analysis[feat]["librosa"]["mean"] * 1000
                report.append(f"| {feat.replace('_', ' ').title()} | {lib_ms:.2f} |")

        report.append("")

    # Summary with highlights
    report.append("## Summary")
    report.append("")

    if "total" in analysis and analysis["total"]["speedup"]:
        speedup = analysis["total"]["speedup"]
        report.append(f"🚀 **Overall Pipeline: {speedup:.2f}x faster**")
        report.append("")

    # Find best speedups
    speedups = [
        (k, v["speedup"])
        for k, v in analysis.items()
        if v["speedup"] is not None and k != "total"
    ]
    speedups.sort(key=lambda x: x[1], reverse=True)

    if speedups:
        report.append("### Top Speedups")
        for feat, speedup in speedups[:3]:
            report.append(f"- **{feat.upper()}**: {speedup:.2f}x faster")

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="Feature-by-feature benchmark comparison"
    )
    parser.add_argument(
        "audio",
        nargs="?",
        default=None,
        help="Path to audio file",
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=5,
        help="Number of iterations (default: 5)",
    )
    parser.add_argument(
        "-w",
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup iterations (default: 1)",
    )
    parser.add_argument(
        "-o",
        "--output",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "-f",
        "--file",
        help="Output file (default: stdout)",
    )

    args = parser.parse_args()

    # Determine audio path
    if args.audio:
        audio_path = args.audio
    else:
        bench_dir = Path(__file__).parent
        audio_files = list((bench_dir / "audio").glob("*.mp3"))
        if not audio_files:
            print("Error: No audio files found in bench/audio/", file=sys.stderr)
            sys.exit(1)
        audio_path = str(audio_files[0])

    if not Path(audio_path).exists():
        print(f"Error: Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    bench_dir = Path(__file__).parent

    # Build rsona if needed
    print("Building rsona...")
    subprocess.run(
        ["cargo", "build", "--release", "--bin", "rsona_bench"],
        cwd=bench_dir.parent,
        capture_output=True,
    )
    print()

    # Run benchmarks
    results = run_feature_benchmarks(
        audio_path, args.iterations, bench_dir, args.warmup
    )

    # Analyze results
    analysis = analyze_feature_results(results)

    # Format output
    if args.output == "markdown":
        output = format_feature_report(analysis, audio_path)
    else:
        output = json.dumps(analysis, indent=2)

    # Write output
    if args.file:
        with open(args.file, "w") as f:
            f.write(output)
        print(f"Results written to: {args.file}")
    else:
        print(output)


if __name__ == "__main__":
    main()
