#!/usr/bin/env python3
"""
Compare librosa_loopfinder vs rsona loop finding performance and accuracy.

This script runs both implementations, compares results, and generates
a comprehensive comparison report.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import numpy as np
except ImportError:
    print("Error: numpy is required", file=sys.stderr)
    print("Install with: pip install numpy", file=sys.stderr)
    sys.exit(1)


def run_librosa_loopfinder(
    audio_path: Path,
    iterations: int = 1,
    win_seconds: float = 8.0,
    min_loop_seconds: float = 10.0,
) -> Optional[Dict[str, Any]]:
    """Run librosa_loopfinder benchmark."""
    script_path = Path(__file__).parent / "loopfinder_benchmark.py"

    if not script_path.exists():
        print(f"Error: {script_path} not found", file=sys.stderr)
        return None

    cmd = [
        "python3",
        str(script_path),
        str(audio_path),
        "-i",
        str(iterations),
        "-w",
        str(win_seconds),
        "-m",
        str(min_loop_seconds),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running librosa_loopfinder: {e}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing librosa_loopfinder output: {e}", file=sys.stderr)
        return None


def run_rsona_loopfinder(audio_path: Path) -> Optional[Dict[str, Any]]:
    """Run rsona loop finding benchmark."""
    import os

    # Check CARGO_TARGET_DIR first, then fall back to default location
    cargo_target_dir = os.environ.get("CARGO_TARGET_DIR")
    if cargo_target_dir:
        binary_path = Path(cargo_target_dir) / "release" / "loopfinder_bench"
    else:
        binary_path = (
            Path(__file__).parent.parent / "target" / "release" / "loopfinder_bench"
        )

    if not binary_path.exists():
        print(f"Error: {binary_path} not found", file=sys.stderr)
        print(
            "Build with: cargo build --release --bin loopfinder_bench", file=sys.stderr
        )
        if cargo_target_dir:
            print(f"Note: Using CARGO_TARGET_DIR={cargo_target_dir}", file=sys.stderr)
        return None

    cmd = [str(binary_path), str(audio_path)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running rsona loopfinder: {e}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing rsona output: {e}", file=sys.stderr)
        return None


def compare_results(
    librosa_result: Dict[str, Any], rsona_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare librosa and rsona results."""

    # Extract timing
    librosa_time = librosa_result["timing"]["mean_time_ms"]
    rsona_time = rsona_result["timing"]["total_time_ms"]

    # Extract loop points
    librosa_best = librosa_result["best_result"]
    rsona_best = rsona_result["best_result"]

    # Calculate differences
    start_diff_seconds = abs(
        librosa_best["loop_begin_seconds"] - rsona_best["loop_begin_seconds"]
    )
    end_diff_seconds = abs(
        librosa_best["loop_end_seconds"] - rsona_best["loop_end_seconds"]
    )
    duration_diff_seconds = abs(
        librosa_best["loop_duration_seconds"] - rsona_best["loop_duration_seconds"]
    )

    # Calculate percentage differences
    duration = rsona_result["duration_seconds"]
    start_diff_pct = (start_diff_seconds / duration) * 100 if duration > 0 else 0
    end_diff_pct = (end_diff_seconds / duration) * 100 if duration > 0 else 0
    duration_diff_pct = (
        (duration_diff_seconds / librosa_best["loop_duration_seconds"]) * 100
        if librosa_best["loop_duration_seconds"] > 0
        else 0
    )

    # Speedup calculation
    speedup = librosa_time / rsona_time if rsona_time > 0 else 0

    return {
        "timing_comparison": {
            "librosa_ms": librosa_time,
            "rsona_ms": rsona_time,
            "speedup": speedup,
            "faster": "rsona" if speedup > 1 else "librosa",
        },
        "loop_point_comparison": {
            "start": {
                "librosa_seconds": librosa_best["loop_begin_seconds"],
                "rsona_seconds": rsona_best["loop_begin_seconds"],
                "difference_seconds": start_diff_seconds,
                "difference_percent": start_diff_pct,
            },
            "end": {
                "librosa_seconds": librosa_best["loop_end_seconds"],
                "rsona_seconds": rsona_best["loop_end_seconds"],
                "difference_seconds": end_diff_seconds,
                "difference_percent": end_diff_pct,
            },
            "duration": {
                "librosa_seconds": librosa_best["loop_duration_seconds"],
                "rsona_seconds": rsona_best["loop_duration_seconds"],
                "difference_seconds": duration_diff_seconds,
                "difference_percent": duration_diff_pct,
            },
        },
        "segmentation_comparison": {
            "intro": {
                "librosa_seconds": librosa_result["segmentation"]["intro_seconds"]
                if librosa_result["segmentation"]
                else 0,
                "rsona_seconds": rsona_result["segmentation"]["intro_seconds"],
            },
            "loop": {
                "librosa_seconds": librosa_result["segmentation"]["loop_seconds"]
                if librosa_result["segmentation"]
                else 0,
                "rsona_seconds": rsona_result["segmentation"]["loop_seconds"],
            },
            "outro": {
                "librosa_seconds": librosa_result["segmentation"]["outro_seconds"]
                if librosa_result["segmentation"]
                else 0,
                "rsona_seconds": rsona_result["segmentation"]["outro_seconds"],
            },
        },
        "accuracy_metrics": {
            "start_within_1s": start_diff_seconds < 1.0,
            "end_within_1s": end_diff_seconds < 1.0,
            "duration_within_5pct": duration_diff_pct < 5.0,
            "overall_match": start_diff_seconds < 1.0
            and end_diff_seconds < 1.0
            and duration_diff_pct < 5.0,
        },
    }


def format_report(
    audio_path: Path,
    librosa_result: Dict[str, Any],
    rsona_result: Dict[str, Any],
    comparison: Dict[str, Any],
) -> str:
    """Format comparison results as a markdown report."""

    lines = ["# Loop Finder Comparison Report", ""]

    # Audio info
    lines.append("## Audio File")
    lines.append(f"- **File**: `{audio_path.name}`")
    lines.append(f"- **Duration**: {rsona_result['duration_seconds']:.2f}s")
    lines.append(f"- **Sample Rate**: {rsona_result['sample_rate']} Hz")
    lines.append(f"- **Samples**: {rsona_result['n_samples']:,}")
    lines.append("")

    # Performance comparison
    timing = comparison["timing_comparison"]
    lines.append("## Performance Comparison")
    lines.append("")
    lines.append("| Implementation | Time (ms) | Speedup |")
    lines.append("|----------------|-----------|---------|")
    lines.append(
        f"| librosa_loopfinder | {timing['librosa_ms']:.2f} | 1.00x (baseline) |"
    )
    lines.append(
        f"| **rsona** | **{timing['rsona_ms']:.2f}** | **{timing['speedup']:.2f}x** {'⚡' if timing['speedup'] > 2 else '✓' if timing['speedup'] > 1 else ''} |"
    )
    lines.append("")

    if timing["speedup"] > 1:
        lines.append(
            f"✅ **rsona is {timing['speedup']:.2f}x faster** ({timing['rsona_ms']:.2f}ms vs {timing['librosa_ms']:.2f}ms)"
        )
    else:
        lines.append(
            f"⚠️ librosa is {1 / timing['speedup']:.2f}x faster ({timing['librosa_ms']:.2f}ms vs {timing['rsona_ms']:.2f}ms)"
        )
    lines.append("")

    # Loop point comparison
    loop_comp = comparison["loop_point_comparison"]
    lines.append("## Loop Point Accuracy")
    lines.append("")
    lines.append("### Loop Start")
    lines.append(
        f"- **librosa**: {loop_comp['start']['librosa_seconds']:.3f}s (sample {librosa_result['best_result']['loop_begin_sample']})"
    )
    lines.append(
        f"- **rsona**: {loop_comp['start']['rsona_seconds']:.3f}s (sample {rsona_result['best_result']['loop_begin_sample']})"
    )
    lines.append(
        f"- **Difference**: {loop_comp['start']['difference_seconds']:.3f}s ({loop_comp['start']['difference_percent']:.2f}% of total)"
    )
    lines.append("")

    lines.append("### Loop End")
    lines.append(
        f"- **librosa**: {loop_comp['end']['librosa_seconds']:.3f}s (sample {librosa_result['best_result']['loop_end_sample']})"
    )
    lines.append(
        f"- **rsona**: {loop_comp['end']['rsona_seconds']:.3f}s (sample {rsona_result['best_result']['loop_end_sample']})"
    )
    lines.append(
        f"- **Difference**: {loop_comp['end']['difference_seconds']:.3f}s ({loop_comp['end']['difference_percent']:.2f}% of total)"
    )
    lines.append("")

    lines.append("### Loop Duration")
    lines.append(f"- **librosa**: {loop_comp['duration']['librosa_seconds']:.3f}s")
    lines.append(f"- **rsona**: {loop_comp['duration']['rsona_seconds']:.3f}s")
    lines.append(
        f"- **Difference**: {loop_comp['duration']['difference_seconds']:.3f}s ({loop_comp['duration']['difference_percent']:.2f}%)"
    )
    lines.append("")

    # Accuracy assessment
    accuracy = comparison["accuracy_metrics"]
    lines.append("### Accuracy Assessment")
    lines.append("")
    lines.append(
        f"- Start within 1s: {'✅ Yes' if accuracy['start_within_1s'] else '❌ No'}"
    )
    lines.append(
        f"- End within 1s: {'✅ Yes' if accuracy['end_within_1s'] else '❌ No'}"
    )
    lines.append(
        f"- Duration within 5%: {'✅ Yes' if accuracy['duration_within_5pct'] else '❌ No'}"
    )
    lines.append(
        f"- **Overall Match**: {'✅ Yes' if accuracy['overall_match'] else '❌ No'}"
    )
    lines.append("")

    # Segmentation comparison
    seg_comp = comparison["segmentation_comparison"]
    lines.append("## Segmentation Breakdown")
    lines.append("")
    lines.append("| Segment | librosa (s) | rsona (s) | Difference (s) |")
    lines.append("|---------|-------------|-----------|----------------|")
    intro_diff = abs(
        seg_comp["intro"]["librosa_seconds"] - seg_comp["intro"]["rsona_seconds"]
    )
    loop_diff = abs(
        seg_comp["loop"]["librosa_seconds"] - seg_comp["loop"]["rsona_seconds"]
    )
    outro_diff = abs(
        seg_comp["outro"]["librosa_seconds"] - seg_comp["outro"]["rsona_seconds"]
    )
    lines.append(
        f"| Intro | {seg_comp['intro']['librosa_seconds']:.2f} | {seg_comp['intro']['rsona_seconds']:.2f} | {intro_diff:.2f} |"
    )
    lines.append(
        f"| Loop | {seg_comp['loop']['librosa_seconds']:.2f} | {seg_comp['loop']['rsona_seconds']:.2f} | {loop_diff:.2f} |"
    )
    lines.append(
        f"| Outro | {seg_comp['outro']['librosa_seconds']:.2f} | {seg_comp['outro']['rsona_seconds']:.2f} | {outro_diff:.2f} |"
    )
    lines.append("")

    # rsona-specific features
    if rsona_result.get("confidence"):
        conf = rsona_result["confidence"]
        lines.append("## rsona-Specific Metrics")
        lines.append("")
        lines.append(f"- **Loop Confidence**: {conf['loop_confidence']:.2f}")
        if conf.get("tempo_bpm"):
            lines.append(f"- **Detected Tempo**: {conf['tempo_bpm']:.1f} BPM")
        lines.append(f"- **Section Boundaries**: {conf['n_section_boundaries']}")
        if conf.get("n_beat_frames"):
            lines.append(f"- **Beat Frames**: {conf['n_beat_frames']}")
        lines.append("")

    # Algorithm differences
    lines.append("## Algorithm Differences")
    lines.append("")
    lines.append("### librosa_loopfinder")
    lines.append("- Uses beat-synchronized features")
    lines.append("- Computes pairwise distance matrix over beat features")
    lines.append("- PCA-reduced feature space (12 components)")
    lines.append("- Manhattan distance metric")
    lines.append(f"- Found {librosa_result['total_candidates_found']} total candidates")
    lines.append("")
    lines.append("### rsona")
    lines.append("- Uses frame-level MFCC features")
    lines.append("- Self-similarity matrix with lag energy estimation")
    lines.append("- Repeat phase detection for loop start")
    lines.append("- Beat-snapped boundaries (when available)")
    lines.append("- Novelty-based section detection")
    lines.append("")

    # Detailed timing breakdown for rsona
    timing_rsona = rsona_result["timing"]
    lines.append("## Detailed Timing Breakdown (rsona)")
    lines.append("")
    lines.append("| Stage | Time (ms) | % of Total |")
    lines.append("|-------|-----------|------------|")
    total = timing_rsona["total_time_ms"]
    lines.append(
        f"| Audio Load | {timing_rsona['load_time_ms']:.2f} | {timing_rsona['load_time_ms'] / total * 100:.1f}% |"
    )
    lines.append(
        f"| Framing | {timing_rsona['frame_time_ms']:.2f} | {timing_rsona['frame_time_ms'] / total * 100:.1f}% |"
    )
    lines.append(
        f"| STFT | {timing_rsona['stft_time_ms']:.2f} | {timing_rsona['stft_time_ms'] / total * 100:.1f}% |"
    )
    lines.append(
        f"| Mel Spectrogram | {timing_rsona['mel_time_ms']:.2f} | {timing_rsona['mel_time_ms'] / total * 100:.1f}% |"
    )
    lines.append(
        f"| MFCC | {timing_rsona['mfcc_time_ms']:.2f} | {timing_rsona['mfcc_time_ms'] / total * 100:.1f}% |"
    )
    lines.append(
        f"| Self-Similarity | {timing_rsona['similarity_time_ms']:.2f} | {timing_rsona['similarity_time_ms'] / total * 100:.1f}% |"
    )
    lines.append(
        f"| Onset Strength | {timing_rsona['onset_time_ms']:.2f} | {timing_rsona['onset_time_ms'] / total * 100:.1f}% |"
    )
    lines.append(
        f"| Segmentation | {timing_rsona['segmentation_time_ms']:.2f} | {timing_rsona['segmentation_time_ms'] / total * 100:.1f}% |"
    )
    lines.append(f"| **Total** | **{total:.2f}** | **100%** |")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare librosa_loopfinder vs rsona loop finding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio_path", type=str, help="Path to audio file")
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations for librosa timing",
    )
    parser.add_argument(
        "-w",
        "--win-seconds",
        type=float,
        default=8.0,
        help="Feature window length (seconds)",
    )
    parser.add_argument(
        "-m",
        "--min-loop-seconds",
        type=float,
        default=10.0,
        help="Minimum loop duration (seconds)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output markdown file (default: print to stdout)",
    )
    parser.add_argument(
        "--json", action="store_true", help="Also save JSON comparison data"
    )

    args = parser.parse_args()

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Comparing loop finders on: {audio_path.name}", file=sys.stderr)
    print("", file=sys.stderr)

    # Run librosa_loopfinder
    print("Running librosa_loopfinder...", file=sys.stderr)
    librosa_result = run_librosa_loopfinder(
        audio_path,
        iterations=args.iterations,
        win_seconds=args.win_seconds,
        min_loop_seconds=args.min_loop_seconds,
    )
    if not librosa_result:
        sys.exit(1)

    # Run rsona
    print("Running rsona loopfinder...", file=sys.stderr)
    rsona_result = run_rsona_loopfinder(audio_path)
    if not rsona_result:
        sys.exit(1)

    print("", file=sys.stderr)

    # Compare results
    comparison = compare_results(librosa_result, rsona_result)

    # Format report
    report = format_report(audio_path, librosa_result, rsona_result, comparison)

    # Output report
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to: {args.output}", file=sys.stderr)
    else:
        print(report)

    # Optionally save JSON
    if args.json:
        json_output = {
            "audio_file": str(audio_path),
            "librosa_result": librosa_result,
            "rsona_result": rsona_result,
            "comparison": comparison,
        }
        json_path = (
            args.output.replace(".md", ".json") if args.output else "comparison.json"
        )
        with open(json_path, "w") as f:
            json.dump(json_output, f, indent=2)
        print(f"JSON data written to: {json_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
