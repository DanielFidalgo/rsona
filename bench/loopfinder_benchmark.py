#!/usr/bin/env python3
"""
Benchmark librosa_loopfinder against test audio files.

This script uses the librosa_loopfinder library to find loop points
and outputs results in JSON format for comparison with rsona.
"""

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import librosa
    import numpy as np
    from librosa_loopfinder import BeatFeaturesGenerator, find_loop_points
    from sklearn.metrics import DistanceMetric
except ImportError as e:
    print(f"Error: Missing required library: {e}", file=sys.stderr)
    print("\nInstall with:", file=sys.stderr)
    print("  pip install librosa scikit-learn", file=sys.stderr)
    print(
        "  pip install git+https://github.com/Kexanone/librosa_loopfinder.git",
        file=sys.stderr,
    )
    sys.exit(1)


def benchmark_loopfinder(
    audio_path,
    win_seconds=8.0,
    min_loop_seconds=10.0,
    max_loop_seconds=None,
    n_results=5,
    iterations=1,
):
    """
    Run librosa_loopfinder on an audio file and return timing + results.

    Args:
        audio_path: Path to audio file
        win_seconds: Feature window length in seconds (default: 8.0)
        min_loop_seconds: Minimum loop duration in seconds (default: 10.0)
        max_loop_seconds: Maximum loop duration in seconds (default: None = infinite)
        n_results: Number of top loop candidates to return (default: 5)
        iterations: Number of times to run for timing (default: 1)

    Returns:
        Dictionary with results and timing information
    """

    # Load audio
    load_start = time.perf_counter()
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    load_time = time.perf_counter() - load_start

    duration_seconds = len(y) / sr

    # Convert time parameters to samples
    win_length = librosa.time_to_samples(win_seconds, sr=sr)
    min_length = librosa.time_to_samples(min_loop_seconds, sr=sr)
    max_length = (
        np.inf
        if max_loop_seconds is None
        else librosa.time_to_samples(max_loop_seconds, sr=sr)
    )

    # Initialize feature generator
    beat_features_gen = BeatFeaturesGenerator(n_pc=12, n_chroma=12, n_mels=128)
    distance_metric = DistanceMetric.get_metric("manhattan")

    # Warm-up run if multiple iterations
    if iterations > 1:
        _ = find_loop_points(
            y=y,
            sr=sr,
            win_length=win_length,
            min_length=min_length,
            max_length=max_length,
            get_beat_features=beat_features_gen,
            distance_metric=distance_metric,
        )

    # Timed runs
    timings = []
    all_results = None

    for i in range(iterations):
        start = time.perf_counter()

        results = find_loop_points(
            y=y,
            sr=sr,
            win_length=win_length,
            min_length=min_length,
            max_length=max_length,
            get_beat_features=beat_features_gen,
            distance_metric=distance_metric,
        )

        elapsed = time.perf_counter() - start
        timings.append(elapsed)

        if i == 0:
            all_results = results

    # Extract top N results
    loop_candidates = []
    for i, (loop_begin, loop_end, score) in enumerate(all_results[:n_results]):
        loop_begin_sec = librosa.samples_to_time(loop_begin, sr=sr)
        loop_end_sec = librosa.samples_to_time(loop_end, sr=sr)
        loop_duration_sec = loop_end_sec - loop_begin_sec

        loop_candidates.append(
            {
                "rank": i + 1,
                "loop_begin_sample": int(loop_begin),
                "loop_end_sample": int(loop_end),
                "loop_begin_seconds": float(loop_begin_sec),
                "loop_end_seconds": float(loop_end_sec),
                "loop_duration_seconds": float(loop_duration_sec),
                "score": float(score),
                "normalized_score": float(
                    score
                ),  # Already normalized by librosa_loopfinder
            }
        )

    # Best result
    if loop_candidates:
        best = loop_candidates[0]
        intro_duration = best["loop_begin_seconds"]
        outro_duration = duration_seconds - best["loop_end_seconds"]
    else:
        best = None
        intro_duration = 0.0
        outro_duration = 0.0

    return {
        "audio_file": str(audio_path),
        "duration_seconds": float(duration_seconds),
        "sample_rate": int(sr),
        "n_samples": len(y),
        "parameters": {
            "win_seconds": win_seconds,
            "min_loop_seconds": min_loop_seconds,
            "max_loop_seconds": max_loop_seconds,
            "n_results": n_results,
        },
        "timing": {
            "load_time_ms": load_time * 1000.0,
            "mean_time_ms": float(np.mean(timings)) * 1000.0,
            "median_time_ms": float(np.median(timings)) * 1000.0,
            "std_time_ms": float(np.std(timings)) * 1000.0,
            "min_time_ms": float(np.min(timings)) * 1000.0,
            "max_time_ms": float(np.max(timings)) * 1000.0,
            "iterations": iterations,
        },
        "best_result": best,
        "segmentation": {
            "intro_seconds": intro_duration,
            "loop_seconds": best["loop_duration_seconds"] if best else 0.0,
            "outro_seconds": outro_duration,
        }
        if best
        else None,
        "top_candidates": loop_candidates,
        "total_candidates_found": len(all_results),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark librosa_loopfinder on audio files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio_path", type=str, help="Path to audio file")
    parser.add_argument(
        "-w",
        "--win-seconds",
        type=float,
        default=8.0,
        help="Feature window length in seconds",
    )
    parser.add_argument(
        "-m",
        "--min-loop-seconds",
        type=float,
        default=10.0,
        help="Minimum loop duration in seconds",
    )
    parser.add_argument(
        "-M",
        "--max-loop-seconds",
        type=float,
        default=None,
        help="Maximum loop duration in seconds",
    )
    parser.add_argument(
        "-n",
        "--n-results",
        type=int,
        default=5,
        help="Number of top loop candidates to return",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations for timing",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output JSON file (default: stdout)",
    )
    parser.add_argument(
        "--pretty", action="store_true", help="Pretty-print JSON output"
    )

    args = parser.parse_args()

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Benchmarking: {audio_path.name}", file=sys.stderr)
    print(
        f"Parameters: win={args.win_seconds}s, min_loop={args.min_loop_seconds}s, iterations={args.iterations}",
        file=sys.stderr,
    )
    print("", file=sys.stderr)

    try:
        results = benchmark_loopfinder(
            audio_path=audio_path,
            win_seconds=args.win_seconds,
            min_loop_seconds=args.min_loop_seconds,
            max_loop_seconds=args.max_loop_seconds,
            n_results=args.n_results,
            iterations=args.iterations,
        )

        # Print summary to stderr
        if results["best_result"]:
            best = results["best_result"]
            timing = results["timing"]
            print(f"Best Loop Found:", file=sys.stderr)
            print(
                f"  Start:    {best['loop_begin_seconds']:.2f}s (sample {best['loop_begin_sample']})",
                file=sys.stderr,
            )
            print(
                f"  End:      {best['loop_end_seconds']:.2f}s (sample {best['loop_end_sample']})",
                file=sys.stderr,
            )
            print(f"  Duration: {best['loop_duration_seconds']:.2f}s", file=sys.stderr)
            print(f"  Score:    {best['score']:.4f}", file=sys.stderr)
            print(f"", file=sys.stderr)
            print(
                f"Timing: {timing['mean_time_ms']:.2f}ms (±{timing['std_time_ms']:.2f}ms)",
                file=sys.stderr,
            )
            print(
                f"Total candidates found: {results['total_candidates_found']}",
                file=sys.stderr,
            )
        else:
            print("No valid loop points found!", file=sys.stderr)

        # Output JSON
        indent = 2 if args.pretty else None
        json_output = json.dumps(results, indent=indent)

        if args.output:
            with open(args.output, "w") as f:
                f.write(json_output)
            print(f"\nResults written to: {args.output}", file=sys.stderr)
        else:
            print(json_output)

    except Exception as e:
        print(f"Error during benchmarking: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
