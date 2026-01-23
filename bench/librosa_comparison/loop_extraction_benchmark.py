#!/usr/bin/env python3
"""
Loop/Intro/Outro Extraction Benchmark: rsona vs librosa

Compares loop point detection and intro/outro segmentation between
rsona (Rust) and librosa_loopfinder (Python).

Usage:
    python loop_extraction_benchmark.py --audio path/to/audio.wav
    python loop_extraction_benchmark.py --audio audio.wav --runs 5 --output results.json
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import librosa
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required library: {e}", file=sys.stderr)
    print("\nInstall with:", file=sys.stderr)
    print("  pip install librosa numpy", file=sys.stderr)
    sys.exit(1)

# Try to import librosa_loopfinder (optional)
LIBROSA_LOOPFINDER_AVAILABLE = False
try:
    from librosa_loopfinder import BeatFeaturesGenerator, find_loop_points
    from sklearn.metrics import DistanceMetric

    LIBROSA_LOOPFINDER_AVAILABLE = True
except ImportError:
    print("Warning: librosa_loopfinder not available. Install with:", file=sys.stderr)
    print(
        "  pip install git+https://github.com/Kexanone/librosa_loopfinder.git scikit-learn",
        file=sys.stderr,
    )


class LoopExtractionBenchmark:
    """Benchmark loop extraction between rsona and librosa."""

    def __init__(self, audio_path: str, runs: int = 3, warmup: int = 1):
        self.audio_path = audio_path
        self.runs = runs
        self.warmup = warmup
        self.audio_duration = None
        self.sample_rate = None

    def load_audio_info(self):
        """Load basic audio information."""
        print(f"Loading audio: {self.audio_path}")
        y, sr = librosa.load(self.audio_path, sr=None, mono=True)
        self.audio_duration = len(y) / sr
        self.sample_rate = sr
        print(f"  Duration: {self.audio_duration:.2f}s")
        print(f"  Sample rate: {sr} Hz")
        return y, sr

    def benchmark_librosa_loopfinder(self, y: np.ndarray, sr: int) -> Dict:
        """
        Benchmark librosa_loopfinder for loop detection.

        Uses the librosa_loopfinder library to find loop points based on
        beat-synchronized features.
        """
        if not LIBROSA_LOOPFINDER_AVAILABLE:
            return {"error": "librosa_loopfinder not available", "available": False}

        print("\nBenchmarking librosa_loopfinder...")

        # Parameters (match rsona defaults where possible)
        win_seconds = 8.0  # Feature window
        min_loop_seconds = 10.0
        max_loop_seconds = None

        win_length = librosa.time_to_samples(win_seconds, sr=sr)
        min_length = librosa.time_to_samples(min_loop_seconds, sr=sr)
        max_length = (
            np.inf
            if max_loop_seconds is None
            else librosa.time_to_samples(max_loop_seconds, sr=sr)
        )

        # Initialize
        get_beat_features = BeatFeaturesGenerator(n_pc=12, n_chroma=12, n_mels=128)
        distance_metric = DistanceMetric.get_metric("manhattan")

        # Warmup
        if self.warmup > 0:
            print(f"  Warmup runs: {self.warmup}")
            for _ in range(self.warmup):
                _ = find_loop_points(
                    y=y,
                    sr=sr,
                    win_length=win_length,
                    min_length=min_length,
                    max_length=max_length,
                    get_beat_features=get_beat_features,
                    distance_metric=distance_metric,
                )

        # Timed runs
        print(f"  Benchmark runs: {self.runs}")
        timings = []
        all_results = []

        for run in range(self.runs):
            start = time.perf_counter()

            loop_candidates = find_loop_points(
                y=y,
                sr=sr,
                win_length=win_length,
                min_length=min_length,
                max_length=max_length,
                get_beat_features=get_beat_features,
                distance_metric=distance_metric,
            )

            end = time.perf_counter()
            runtime_ms = (end - start) * 1000
            timings.append(runtime_ms)
            all_results.append(loop_candidates)
            print(f"    Run {run + 1}: {runtime_ms:.2f} ms")

        # Extract best result
        best_result = None
        intro_seconds = 0.0
        loop_seconds = 0.0
        outro_seconds = 0.0

        if all_results[0]:  # Use first run's results
            best = all_results[0][0]  # Best candidate (tuple)
            # Tuple format: (loop_begin_samples, loop_end_samples, distance)
            loop_begin_samples = int(best[0])
            loop_end_samples = int(best[1])
            distance = float(best[2])

            loop_begin_seconds = librosa.samples_to_time(loop_begin_samples, sr=sr)
            loop_end_seconds = librosa.samples_to_time(loop_end_samples, sr=sr)

            intro_seconds = loop_begin_seconds
            loop_seconds = loop_end_seconds - loop_begin_seconds
            outro_seconds = self.audio_duration - loop_end_seconds

            best_result = {
                "loop_begin_seconds": float(loop_begin_seconds),
                "loop_end_seconds": float(loop_end_seconds),
                "loop_duration_seconds": float(loop_seconds),
                "distance": distance,
                "loop_begin_samples": loop_begin_samples,
                "loop_end_samples": loop_end_samples,
            }

        return {
            "available": True,
            "runtime_ms": {
                "mean": float(np.mean(timings)),
                "median": float(np.median(timings)),
                "std": float(np.std(timings)),
                "min": float(np.min(timings)),
                "max": float(np.max(timings)),
            },
            "best_result": best_result,
            "segmentation": {
                "intro_seconds": float(intro_seconds),
                "loop_seconds": float(loop_seconds),
                "outro_seconds": float(outro_seconds),
            },
            "num_candidates": len(all_results[0]) if all_results[0] else 0,
        }

    def benchmark_rsona_loopfinder(self) -> Dict:
        """
        Benchmark rsona's loop finder.

        Calls the rsona loopfinder_bench_beats binary to get results.
        """
        print("\nBenchmarking rsona loop finder...")

        # Check if binary exists
        rsona_bench_path = (
            Path.home() / ".cargo-target" / "release" / "loopfinder_bench_beats"
        )
        if not rsona_bench_path.exists():
            # Try building it
            print("  Building rsona loopfinder benchmark...")
            build_cmd = [
                "cargo",
                "build",
                "--release",
                "--bin",
                "loopfinder_bench_beats",
            ]
            try:
                subprocess.run(build_cmd, check=True, cwd="../..", capture_output=True)
            except subprocess.CalledProcessError as e:
                return {
                    "error": f"Failed to build rsona benchmark: {e}",
                    "available": False,
                }

        # Run benchmark
        timings = []
        all_results = []

        # Warmup + runs
        total_runs = self.warmup + self.runs
        print(f"  Total runs (warmup + benchmark): {total_runs}")

        for run in range(total_runs):
            try:
                cmd = [str(rsona_bench_path), self.audio_path]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)

                # Parse JSON output
                output = json.loads(result.stdout)

                if run >= self.warmup:  # Skip warmup runs
                    runtime_ms = output["timing"]["total_time_ms"]
                    timings.append(runtime_ms)
                    all_results.append(output)
                    print(f"    Run {run - self.warmup + 1}: {runtime_ms:.2f} ms")
                else:
                    print(f"    Warmup {run + 1}")

            except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
                return {
                    "error": f"Failed to run rsona benchmark: {e}",
                    "available": False,
                }

        # Extract results from first benchmark run
        if not all_results:
            return {"error": "No results collected", "available": False}

        result = all_results[0]

        return {
            "available": True,
            "runtime_ms": {
                "mean": float(np.mean(timings)),
                "median": float(np.median(timings)),
                "std": float(np.std(timings)),
                "min": float(np.min(timings)),
                "max": float(np.max(timings)),
            },
            "best_result": result.get("best_result"),
            "segmentation": result.get("segmentation"),
            "timing_breakdown": result.get("timing"),
            "parameters": result.get("parameters"),
        }

    def compare_results(self, librosa_result: Dict, rsona_result: Dict) -> Dict:
        """Compare results between librosa and rsona."""

        comparison = {
            "audio_file": self.audio_path,
            "audio_duration_seconds": self.audio_duration,
            "sample_rate": self.sample_rate,
            "librosa_available": librosa_result.get("available", False),
            "rsona_available": rsona_result.get("available", False),
        }

        # Performance comparison
        if librosa_result.get("available") and rsona_result.get("available"):
            lib_mean = librosa_result["runtime_ms"]["mean"]
            rsona_mean = rsona_result["runtime_ms"]["mean"]
            speedup = lib_mean / rsona_mean if rsona_mean > 0 else 0

            comparison["performance"] = {
                "librosa_ms": lib_mean,
                "rsona_ms": rsona_mean,
                "speedup": speedup,
                "time_saved_ms": lib_mean - rsona_mean,
            }

            # Loop point comparison
            lib_best = librosa_result.get("best_result")
            rsona_best = rsona_result.get("best_result")

            if lib_best and rsona_best:
                loop_begin_diff = abs(
                    lib_best["loop_begin_seconds"] - rsona_best["loop_begin_seconds"]
                )
                loop_end_diff = abs(
                    lib_best["loop_end_seconds"] - rsona_best["loop_end_seconds"]
                )
                duration_diff = abs(
                    lib_best["loop_duration_seconds"]
                    - rsona_best["loop_duration_seconds"]
                )

                comparison["loop_points"] = {
                    "loop_begin_diff_seconds": loop_begin_diff,
                    "loop_end_diff_seconds": loop_end_diff,
                    "duration_diff_seconds": duration_diff,
                    "librosa": {
                        "begin": lib_best["loop_begin_seconds"],
                        "end": lib_best["loop_end_seconds"],
                        "duration": lib_best["loop_duration_seconds"],
                    },
                    "rsona": {
                        "begin": rsona_best["loop_begin_seconds"],
                        "end": rsona_best["loop_end_seconds"],
                        "duration": rsona_best["loop_duration_seconds"],
                    },
                }

            # Segmentation comparison
            lib_seg = librosa_result.get("segmentation", {})
            rsona_seg = rsona_result.get("segmentation", {})

            comparison["segmentation"] = {
                "intro_diff_seconds": abs(
                    lib_seg.get("intro_seconds", 0) - rsona_seg.get("intro_seconds", 0)
                ),
                "loop_diff_seconds": abs(
                    lib_seg.get("loop_seconds", 0) - rsona_seg.get("loop_seconds", 0)
                ),
                "outro_diff_seconds": abs(
                    lib_seg.get("outro_seconds", 0) - rsona_seg.get("outro_seconds", 0)
                ),
                "librosa": lib_seg,
                "rsona": rsona_seg,
            }

        return comparison

    def run(self) -> Dict:
        """Run complete benchmark suite."""
        print("=" * 80)
        print("Loop/Intro/Outro Extraction Benchmark")
        print("=" * 80)

        # Load audio
        y, sr = self.load_audio_info()

        # Benchmark librosa
        librosa_result = self.benchmark_librosa_loopfinder(y, sr)

        # Benchmark rsona
        rsona_result = self.benchmark_rsona_loopfinder()

        # Compare
        comparison = self.compare_results(librosa_result, rsona_result)

        # Combine all results
        results = {
            "benchmark_type": "loop_extraction",
            "librosa": librosa_result,
            "rsona": rsona_result,
            "comparison": comparison,
        }

        return results

    def print_summary(self, results: Dict):
        """Print human-readable summary."""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 80)

        comp = results["comparison"]

        if "performance" in comp:
            perf = comp["performance"]
            print(f"\nPerformance:")
            print(f"  librosa: {perf['librosa_ms']:.2f} ms")
            print(f"  rsona:   {perf['rsona_ms']:.2f} ms")
            print(f"  Speedup: {perf['speedup']:.2f}x")
            print(f"  Time saved: {perf['time_saved_ms']:.2f} ms")

        if "loop_points" in comp:
            lp = comp["loop_points"]
            print(f"\nLoop Points Comparison:")
            print(f"  Begin difference: {lp['loop_begin_diff_seconds']:.3f}s")
            print(f"  End difference:   {lp['loop_end_diff_seconds']:.3f}s")
            print(f"  Duration difference: {lp['duration_diff_seconds']:.3f}s")

        if "segmentation" in comp:
            seg = comp["segmentation"]
            print(f"\nSegmentation Comparison:")
            print(f"  Intro difference:  {seg['intro_diff_seconds']:.3f}s")
            print(f"  Loop difference:   {seg['loop_diff_seconds']:.3f}s")
            print(f"  Outro difference:  {seg['outro_diff_seconds']:.3f}s")

            print(f"\nSegmentation Details:")
            lib_seg = seg["librosa"]
            rsona_seg = seg["rsona"]
            print(f"  {'Segment':<10} {'librosa':<12} {'rsona':<12}")
            print(f"  {'-' * 34}")
            print(
                f"  {'Intro':<10} {lib_seg.get('intro_seconds', 0):>10.2f}s  {rsona_seg.get('intro_seconds', 0):>10.2f}s"
            )
            print(
                f"  {'Loop':<10} {lib_seg.get('loop_seconds', 0):>10.2f}s  {rsona_seg.get('loop_seconds', 0):>10.2f}s"
            )
            print(
                f"  {'Outro':<10} {lib_seg.get('outro_seconds', 0):>10.2f}s  {rsona_seg.get('outro_seconds', 0):>10.2f}s"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark loop/intro/outro extraction: rsona vs librosa"
    )
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument(
        "--runs", type=int, default=3, help="Number of benchmark runs (default: 3)"
    )
    parser.add_argument(
        "--warmup", type=int, default=1, help="Number of warmup runs (default: 1)"
    )
    parser.add_argument(
        "--output", help="Output JSON file path (default: print to stdout)"
    )
    parser.add_argument(
        "--summary", action="store_true", help="Print human-readable summary"
    )

    args = parser.parse_args()

    # Validate audio file exists
    if not Path(args.audio).exists():
        print(f"Error: Audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    # Run benchmark
    benchmark = LoopExtractionBenchmark(
        audio_path=args.audio, runs=args.runs, warmup=args.warmup
    )

    try:
        results = benchmark.run()

        # Save or print results
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")
        else:
            print("\n" + json.dumps(results, indent=2))

        # Print summary if requested
        if args.summary or not args.output:
            benchmark.print_summary(results)

    except Exception as e:
        print(f"Error during benchmark: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
