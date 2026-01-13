#!/usr/bin/env python3
"""
Benchmark comparison script between rsona (Rust) and librosa (Python).

This script runs both implementations multiple times and compares:
- Execution time (mean, median, std dev)
- Accuracy (tempo estimation, frame counts)
- Speedup factor

Usage:
    python compare_bench.py [audio_file] [--iterations N] [--output FORMAT]
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import librosa
import librosa.feature.rhythm
import numpy as np


def run_librosa_benchmark(audio_path: str) -> Dict[str, Any]:
    """Run librosa benchmark and return results."""
    y, sr = librosa.load(audio_path, mono=True, sr=None)

    t0 = time.perf_counter()

    S = librosa.stft(y, n_fft=2048, hop_length=512)
    mel = librosa.feature.melspectrogram(
        S=np.abs(S) ** 2, sr=sr, n_fft=2048, hop_length=512
    )
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel))
    onset = librosa.onset.onset_strength(S=mel, sr=sr)
    tempo = librosa.feature.rhythm.tempo(onset_envelope=onset, sr=sr)[0]

    t1 = time.perf_counter()

    return {
        "time_sec": t1 - t0,
        "tempo_bpm": float(tempo),
        "n_frames": mfcc.shape[1],
    }


def run_rsona_benchmark(audio_path: str, bench_dir: Path) -> Dict[str, Any]:
    """Run rsona benchmark via cargo and return results."""
    result = subprocess.run(
        ["cargo", "run", "--release", "--bin", "rsona_bench", "--", audio_path],
        cwd=bench_dir.parent,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"rsona benchmark failed: {result.stderr}")

    # Parse JSON output from the last line
    for line in reversed(result.stdout.strip().split("\n")):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue

    raise RuntimeError("Could not parse rsona benchmark output")


def run_benchmarks(
    audio_path: str, iterations: int, bench_dir: Path, warmup: int = 1
) -> Dict[str, List[Dict[str, Any]]]:
    """Run both benchmarks multiple times."""
    print(f"Running benchmarks on: {audio_path}")
    print(f"Iterations: {iterations} (+ {warmup} warmup)\n")

    # Warmup runs to handle caching and JIT compilation
    if warmup > 0:
        print(
            f"Warmup ({warmup} iteration{'s' if warmup > 1 else ''})...",
            end=" ",
            flush=True,
        )
        for _ in range(warmup):
            try:
                run_librosa_benchmark(audio_path)
                run_rsona_benchmark(audio_path, bench_dir)
            except Exception:
                pass
        print("done\n")

    librosa_results = []
    rsona_results = []

    for i in range(iterations):
        print(f"Iteration {i + 1}/{iterations}...", end=" ", flush=True)

        # Run librosa
        try:
            librosa_res = run_librosa_benchmark(audio_path)
            librosa_results.append(librosa_res)
            print(f"librosa: {librosa_res['time_sec']:.4f}s", end=" | ")
        except Exception as e:
            print(f"librosa failed: {e}")
            continue

        # Run rsona
        try:
            rsona_res = run_rsona_benchmark(audio_path, bench_dir)
            rsona_results.append(rsona_res)
            print(f"rsona: {rsona_res['time_sec']:.4f}s")
        except Exception as e:
            print(f"rsona failed: {e}")
            continue

    print()
    return {
        "librosa": librosa_results,
        "rsona": rsona_results,
    }


def calculate_stats(values: List[float]) -> Dict[str, float]:
    """Calculate statistical measures for a list of values."""
    if not values:
        return {}

    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def analyze_results(results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Analyze and compare benchmark results."""
    librosa_times = [r["time_sec"] for r in results["librosa"]]
    rsona_times = [r["time_sec"] for r in results["rsona"]]

    librosa_tempos = [r["tempo_bpm"] for r in results["librosa"]]
    rsona_tempos = [r["tempo_bpm"] for r in results["rsona"]]

    librosa_frames = [r["n_frames"] for r in results["librosa"]]
    rsona_frames = [r["n_frames"] for r in results["rsona"]]

    analysis = {
        "librosa": {
            "time": calculate_stats(librosa_times),
            "tempo": calculate_stats(librosa_tempos),
            "frames": librosa_frames[0] if librosa_frames else 0,
        },
        "rsona": {
            "time": calculate_stats(rsona_times),
            "tempo": calculate_stats(rsona_tempos),
            "frames": rsona_frames[0] if rsona_frames else 0,
        },
    }

    # Calculate speedup
    if librosa_times and rsona_times:
        analysis["speedup"] = {
            "mean": statistics.mean(librosa_times) / statistics.mean(rsona_times),
            "median": statistics.median(librosa_times) / statistics.median(rsona_times),
        }

    # Calculate tempo difference
    if librosa_tempos and rsona_tempos:
        tempo_diff = abs(
            statistics.mean(librosa_tempos) - statistics.mean(rsona_tempos)
        )
        tempo_diff_pct = (tempo_diff / statistics.mean(librosa_tempos)) * 100
        analysis["tempo_difference"] = {
            "absolute": tempo_diff,
            "percent": tempo_diff_pct,
        }

    return analysis


def format_markdown_report(analysis: Dict[str, Any], audio_path: str) -> str:
    """Generate a markdown-formatted report."""
    report = []
    report.append("# Benchmark Comparison: rsona vs librosa")
    report.append("")
    report.append(f"**Audio File:** `{Path(audio_path).name}`")
    report.append("")

    # Timing Results
    report.append("## Execution Time")
    report.append("")
    report.append("| Metric | librosa (s) | rsona (s) | Speedup |")
    report.append("|--------|-------------|-----------|---------|")

    lib_time = analysis["librosa"]["time"]
    rs_time = analysis["rsona"]["time"]

    if "mean" in lib_time and "mean" in rs_time:
        speedup_mean = analysis["speedup"]["mean"]
        report.append(
            f"| Mean   | {lib_time['mean']:.4f} | {rs_time['mean']:.4f} | {speedup_mean:.2f}x |"
        )
        report.append(
            f"| Median | {lib_time['median']:.4f} | {rs_time['median']:.4f} | {analysis['speedup']['median']:.2f}x |"
        )
        report.append(
            f"| Std Dev| {lib_time['stdev']:.4f} | {rs_time['stdev']:.4f} | - |"
        )
        report.append(f"| Min    | {lib_time['min']:.4f} | {rs_time['min']:.4f} | - |")
        report.append(f"| Max    | {lib_time['max']:.4f} | {rs_time['max']:.4f} | - |")

    report.append("")

    # Accuracy Results
    report.append("## Tempo Estimation")
    report.append("")
    report.append("| Metric | librosa (BPM) | rsona (BPM) | Difference |")
    report.append("|--------|---------------|-------------|------------|")

    lib_tempo = analysis["librosa"]["tempo"]
    rs_tempo = analysis["rsona"]["tempo"]

    if "mean" in lib_tempo and "mean" in rs_tempo:
        tempo_diff = analysis["tempo_difference"]
        report.append(
            f"| Mean   | {lib_tempo['mean']:.2f} | {rs_tempo['mean']:.2f} | {tempo_diff['absolute']:.2f} ({tempo_diff['percent']:.2f}%) |"
        )
        report.append(
            f"| Median | {lib_tempo['median']:.2f} | {rs_tempo['median']:.2f} | - |"
        )
        report.append(
            f"| Std Dev| {lib_tempo['stdev']:.2f} | {rs_tempo['stdev']:.2f} | - |"
        )

    report.append("")

    # Frame Count
    report.append("## Frame Count")
    report.append("")
    report.append(f"- **librosa:** {analysis['librosa']['frames']} frames")
    report.append(f"- **rsona:** {analysis['rsona']['frames']} frames")

    if analysis["librosa"]["frames"] != analysis["rsona"]["frames"]:
        diff = abs(analysis["librosa"]["frames"] - analysis["rsona"]["frames"])
        report.append(f"- **Difference:** {diff} frames")

    report.append("")

    # Summary
    if "speedup" in analysis:
        report.append("## Summary")
        report.append("")
        speedup = analysis["speedup"]["mean"]
        if speedup > 1:
            report.append(f"🚀 **rsona is {speedup:.2f}x faster than librosa**")
        elif speedup < 1:
            report.append(f"⚠️ **librosa is {1 / speedup:.2f}x faster than rsona**")
        else:
            report.append("⚖️ **Both implementations have similar performance**")

        if "tempo_difference" in analysis:
            tempo_diff_pct = analysis["tempo_difference"]["percent"]
            if tempo_diff_pct < 1:
                report.append(
                    f"✅ Tempo estimation is highly accurate (< 1% difference)"
                )
            elif tempo_diff_pct < 5:
                report.append(
                    f"✓ Tempo estimation is accurate ({tempo_diff_pct:.2f}% difference)"
                )
            else:
                report.append(f"⚠️ Tempo estimation differs by {tempo_diff_pct:.2f}%")

    return "\n".join(report)


def format_json_report(analysis: Dict[str, Any]) -> str:
    """Generate a JSON-formatted report."""
    return json.dumps(analysis, indent=2)


def format_csv_report(results: Dict[str, List[Dict[str, Any]]]) -> str:
    """Generate a CSV-formatted report."""
    lines = []
    lines.append("implementation,iteration,time_sec,tempo_bpm,n_frames")

    for i, result in enumerate(results["librosa"]):
        lines.append(
            f"librosa,{i + 1},{result['time_sec']},{result['tempo_bpm']},{result['n_frames']}"
        )

    for i, result in enumerate(results["rsona"]):
        lines.append(
            f"rsona,{i + 1},{result['time_sec']},{result['tempo_bpm']},{result['n_frames']}"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark comparison between rsona and librosa"
    )
    parser.add_argument(
        "audio",
        nargs="?",
        default=None,
        help="Path to audio file (default: bench/audio/*.mp3)",
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
        choices=["markdown", "json", "csv"],
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

    # Run benchmarks
    results = run_benchmarks(audio_path, args.iterations, bench_dir, args.warmup)

    # Analyze results
    analysis = analyze_results(results)

    # Format output
    if args.output == "markdown":
        output = format_markdown_report(analysis, audio_path)
    elif args.output == "json":
        output = format_json_report(analysis)
    elif args.output == "csv":
        output = format_csv_report(results)

    # Write output
    if args.file:
        with open(args.file, "w") as f:
            f.write(output)
        print(f"Results written to: {args.file}")
    else:
        print(output)


if __name__ == "__main__":
    main()
