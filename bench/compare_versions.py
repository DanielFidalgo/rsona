#!/usr/bin/env python3
"""
Comprehensive benchmark comparison between baseline and current rsona versions.

This script runs feature-by-feature benchmarks comparing baseline (git tag or commit)
against current version, and generates a detailed markdown report.

Usage:
    python compare_versions.py [audio_file] [--baseline TAG] [--iterations N] [--output FILE]
"""

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class BenchmarkRunner:
    """Runs benchmarks and generates comparison reports."""

    def __init__(self, audio_path: str, iterations: int = 5, warmup: int = 1):
        self.audio_path = audio_path
        self.iterations = iterations
        self.warmup = warmup
        self.baseline_binary = None
        self.current_binary = None

    def build_binary(self, output_name: str) -> Path:
        """Build rsona_bench binary."""
        print(f"Building {output_name}...")
        result = subprocess.run(
            ["cargo", "build", "--release", "--bin", "rsona_bench"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Error building {output_name}:")
            print(result.stderr)
            sys.exit(1)

        # Copy binary to unique name
        binary_src = Path("target/release/rsona_bench")
        binary_dst = Path(f"target/release/{output_name}")
        subprocess.run(["cp", str(binary_src), str(binary_dst)], check=True)
        return binary_dst

    def run_benchmark(self, binary: Path) -> Dict[str, Any]:
        """Run benchmark binary and parse JSON output."""
        results = []

        # Warmup runs
        for _ in range(self.warmup):
            subprocess.run(
                [str(binary), self.audio_path],
                capture_output=True,
                check=True,
            )

        # Actual benchmark runs
        for _ in range(self.iterations):
            result = subprocess.run(
                [str(binary), self.audio_path],
                capture_output=True,
                text=True,
                check=True,
            )
            data = json.loads(result.stdout)
            results.append(data)

        return self.aggregate_results(results)

    def aggregate_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate multiple benchmark runs with statistics."""
        if not results:
            return {}

        # Extract timing data
        timing_keys = results[0].get("timing", {}).keys()
        aggregated = {
            "timing": {},
            "audio_info": results[0].get("audio_info", {}),
            "parameters": results[0].get("parameters", {}),
        }

        for key in timing_keys:
            values = [r["timing"][key] for r in results]
            aggregated["timing"][key] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                "min": min(values),
                "max": max(values),
            }

        return aggregated

    def compare_results(
        self, baseline: Dict[str, Any], current: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare baseline and current results."""
        comparison = {}

        for feature, baseline_stats in baseline.get("timing", {}).items():
            current_stats = current.get("timing", {}).get(feature, {})
            if not current_stats:
                continue

            baseline_mean = baseline_stats["mean"]
            current_mean = current_stats["mean"]

            if baseline_mean > 0:
                speedup = baseline_mean / current_mean
                improvement_pct = ((baseline_mean - current_mean) / baseline_mean) * 100
            else:
                speedup = 1.0
                improvement_pct = 0.0

            comparison[feature] = {
                "baseline_mean": baseline_mean,
                "baseline_stdev": baseline_stats["stdev"],
                "current_mean": current_mean,
                "current_stdev": current_stats["stdev"],
                "speedup": speedup,
                "improvement_pct": improvement_pct,
                "regression": speedup < 0.95,  # >5% slower is regression
            }

        return comparison

    def generate_markdown_report(
        self,
        comparison: Dict[str, Any],
        baseline_info: Dict[str, Any],
        current_info: Dict[str, Any],
        baseline_version: str,
        output_file: Optional[str] = None,
    ) -> str:
        """Generate detailed markdown report."""
        lines = []

        # Header
        lines.append("# rsona Benchmark Report")
        lines.append("")
        lines.append(
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
        lines.append("")

        # Version Info
        lines.append("## Version Information")
        lines.append("")
        lines.append(f"- **Baseline:** `{baseline_version}`")
        lines.append(f"- **Current:** `HEAD`")
        lines.append("")

        # Test Configuration
        lines.append("## Test Configuration")
        lines.append("")
        audio_info = current_info.get("audio_info", {})
        lines.append(f"- **Audio File:** `{self.audio_path}`")
        lines.append(f"- **Duration:** {audio_info.get('duration_seconds', 0):.2f}s")
        lines.append(f"- **Sample Rate:** {audio_info.get('sample_rate', 0)} Hz")
        lines.append(f"- **Samples:** {audio_info.get('n_samples', 0):,}")
        lines.append(f"- **Iterations:** {self.iterations}")
        lines.append(f"- **Warmup Runs:** {self.warmup}")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")

        total_baseline = comparison.get("total_time_ms", {}).get("baseline_mean", 0)
        total_current = comparison.get("total_time_ms", {}).get("current_mean", 0)
        total_speedup = comparison.get("total_time_ms", {}).get("speedup", 1.0)

        if total_speedup >= 1.1:
            emoji = "🚀 **FASTER**"
        elif total_speedup >= 1.05:
            emoji = "⚡ **Slightly Faster**"
        elif total_speedup >= 0.95:
            emoji = "➡️ **Similar Performance**"
        else:
            emoji = "⚠️ **REGRESSION**"

        lines.append(f"### Overall Performance: {emoji}")
        lines.append("")
        lines.append(f"- **Baseline Total:** {total_baseline:.2f} ms")
        lines.append(f"- **Current Total:** {total_current:.2f} ms")
        lines.append(f"- **Speedup:** {total_speedup:.2f}x")
        lines.append(
            f"- **Improvement:** {comparison.get('total_time_ms', {}).get('improvement_pct', 0):.1f}%"
        )
        lines.append("")

        # Check for regressions
        regressions = [
            (feature, data)
            for feature, data in comparison.items()
            if data.get("regression", False)
        ]
        if regressions:
            lines.append("### ⚠️ Performance Regressions Detected")
            lines.append("")
            for feature, data in regressions:
                lines.append(f"- **{feature}**: {data['speedup']:.2f}x (slower)")
            lines.append("")

        # Feature-by-Feature Comparison
        lines.append("## Feature-by-Feature Results")
        lines.append("")
        lines.append("| Feature | Baseline (ms) | Current (ms) | Speedup | Change |")
        lines.append("|---------|---------------|--------------|---------|--------|")

        # Sort by speedup (best improvements first)
        sorted_features = sorted(
            comparison.items(),
            key=lambda x: x[1].get("speedup", 1.0),
            reverse=True,
        )

        for feature, data in sorted_features:
            baseline = data["baseline_mean"]
            current = data["current_mean"]
            speedup = data["speedup"]
            improvement = data["improvement_pct"]

            # Format feature name
            feature_name = feature.replace("_", " ").title().replace("Ms", "")

            # Determine emoji
            if speedup >= 2.0:
                change_emoji = "🚀"
            elif speedup >= 1.5:
                change_emoji = "⚡"
            elif speedup >= 1.1:
                change_emoji = "✓"
            elif speedup >= 0.95:
                change_emoji = "→"
            else:
                change_emoji = "⚠️"

            lines.append(
                f"| {feature_name} | {baseline:.2f} ± {data['baseline_stdev']:.2f} | "
                f"{current:.2f} ± {data['current_stdev']:.2f} | "
                f"**{speedup:.2f}x** | {change_emoji} {improvement:+.1f}% |"
            )

        lines.append("")

        # Detailed Timing Breakdown
        lines.append("## Detailed Timing Breakdown")
        lines.append("")
        lines.append("### Baseline Timings")
        lines.append("")
        lines.append("| Feature | Mean | Median | Std Dev | Min | Max |")
        lines.append("|---------|------|--------|---------|-----|-----|")

        for feature in sorted(comparison.keys()):
            data = comparison[feature]
            baseline = data["baseline_mean"]
            stdev = data["baseline_stdev"]

            feature_name = feature.replace("_", " ").title().replace("Ms", "")
            lines.append(
                f"| {feature_name} | {baseline:.2f} | - | {stdev:.2f} | - | - |"
            )

        lines.append("")
        lines.append("### Current Timings")
        lines.append("")
        lines.append("| Feature | Mean | Median | Std Dev | Min | Max |")
        lines.append("|---------|------|--------|---------|-----|-----|")

        for feature in sorted(comparison.keys()):
            data = comparison[feature]
            current = data["current_mean"]
            stdev = data["current_stdev"]

            feature_name = feature.replace("_", " ").title().replace("Ms", "")
            lines.append(
                f"| {feature_name} | {current:.2f} | - | {stdev:.2f} | - | - |"
            )

        lines.append("")

        # Performance Categories
        lines.append("## Performance Categories")
        lines.append("")

        massive_improvements = [
            (f, d) for f, d in comparison.items() if d["speedup"] >= 2.0
        ]
        good_improvements = [
            (f, d) for f, d in comparison.items() if 1.5 <= d["speedup"] < 2.0
        ]
        minor_improvements = [
            (f, d) for f, d in comparison.items() if 1.1 <= d["speedup"] < 1.5
        ]
        similar = [(f, d) for f, d in comparison.items() if 0.95 <= d["speedup"] < 1.1]
        regressions = [(f, d) for f, d in comparison.items() if d["speedup"] < 0.95]

        if massive_improvements:
            lines.append("### 🚀 Massive Improvements (≥2x faster)")
            lines.append("")
            for feature, data in massive_improvements:
                lines.append(f"- **{feature}**: {data['speedup']:.2f}x faster")
            lines.append("")

        if good_improvements:
            lines.append("### ⚡ Good Improvements (1.5x - 2x faster)")
            lines.append("")
            for feature, data in good_improvements:
                lines.append(f"- **{feature}**: {data['speedup']:.2f}x faster")
            lines.append("")

        if minor_improvements:
            lines.append("### ✓ Minor Improvements (1.1x - 1.5x faster)")
            lines.append("")
            for feature, data in minor_improvements:
                lines.append(f"- **{feature}**: {data['speedup']:.2f}x faster")
            lines.append("")

        if similar:
            lines.append("### → Similar Performance (within 5%)")
            lines.append("")
            for feature, data in similar:
                lines.append(f"- **{feature}**: {data['speedup']:.2f}x")
            lines.append("")

        if regressions:
            lines.append("### ⚠️ Regressions (>5% slower)")
            lines.append("")
            for feature, data in regressions:
                lines.append(
                    f"- **{feature}**: {data['speedup']:.2f}x (needs investigation)"
                )
            lines.append("")

        # Recommendations
        lines.append("## Recommendations")
        lines.append("")

        if total_speedup >= 1.1:
            lines.append(
                "✅ **Performance improvements detected!** Current version shows measurable speedups."
            )
        elif total_speedup >= 0.95:
            lines.append(
                "✓ **Performance maintained.** No significant changes detected."
            )
        else:
            lines.append(
                "⚠️ **Performance regression detected!** Investigate before merging."
            )

        if regressions:
            lines.append("")
            lines.append("**Action Items:**")
            for feature, data in regressions:
                lines.append(
                    f"- Investigate `{feature}` regression ({data['speedup']:.2f}x slower)"
                )

        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("*Generated by rsona benchmark suite*")

        report = "\n".join(lines)

        # Save to file if specified
        if output_file:
            with open(output_file, "w") as f:
                f.write(report)
            print(f"\n✓ Report saved to: {output_file}")

        return report


def main():
    parser = argparse.ArgumentParser(
        description="Compare rsona benchmark performance between versions"
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        default="test_audio.wav",
        help="Audio file to benchmark (default: test_audio.wav)",
    )
    parser.add_argument(
        "--baseline",
        default="",
        help="Baseline git tag/commit (default: latest tag)",
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=5,
        help="Number of benchmark iterations (default: 5)",
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
        default="BENCHMARK_RESULTS.md",
        help="Output file for markdown report (default: BENCHMARK_RESULTS.md)",
    )

    args = parser.parse_args()

    # Check if audio file exists
    if not Path(args.audio_file).exists():
        print(f"Error: Audio file not found: {args.audio_file}")
        print("\nGenerate test audio with:")
        print("  ./scripts/download_test_audio.sh")
        sys.exit(1)

    print("=" * 60)
    print("rsona Benchmark Comparison")
    print("=" * 60)
    print()

    runner = BenchmarkRunner(args.audio_file, args.iterations, args.warmup)

    # Determine baseline version
    if args.baseline:
        baseline_version = args.baseline
    else:
        # Get latest tag
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            capture_output=True,
            text=True,
        )
        baseline_version = result.stdout.strip() if result.returncode == 0 else "HEAD~1"

    if not baseline_version:
        baseline_version = "HEAD~1"

    print(f"Baseline version: {baseline_version}")
    print(f"Current version:  HEAD")
    print()

    # Save current state
    subprocess.run(["git", "stash"], capture_output=True)

    try:
        # Build and run baseline
        print("Building baseline version...")
        subprocess.run(
            ["git", "checkout", baseline_version], check=True, capture_output=True
        )
        baseline_binary = runner.build_binary("rsona_bench_baseline")

        print(f"Running baseline benchmarks ({args.iterations} iterations)...")
        baseline_results = runner.run_benchmark(baseline_binary)

        # Restore current version
        print("\nBuilding current version...")
        subprocess.run(["git", "checkout", "-"], check=True, capture_output=True)
        subprocess.run(["git", "stash", "pop"], capture_output=True)

        current_binary = runner.build_binary("rsona_bench_current")

        print(f"Running current benchmarks ({args.iterations} iterations)...")
        current_results = runner.run_benchmark(current_binary)

        # Compare results
        print("\nAnalyzing results...")
        comparison = runner.compare_results(baseline_results, current_results)

        # Generate report
        print("Generating markdown report...")
        report = runner.generate_markdown_report(
            comparison,
            baseline_results,
            current_results,
            baseline_version,
            args.output,
        )

        # Print summary to console
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        total_speedup = comparison.get("total_time_ms", {}).get("speedup", 1.0)
        total_improvement = comparison.get("total_time_ms", {}).get(
            "improvement_pct", 0
        )

        print(f"\nOverall Speedup: {total_speedup:.2f}x ({total_improvement:+.1f}%)")

        regressions = [
            (f, d) for f, d in comparison.items() if d.get("regression", False)
        ]
        if regressions:
            print("\n⚠️ REGRESSIONS DETECTED:")
            for feature, data in regressions:
                print(f"  - {feature}: {data['speedup']:.2f}x")
            sys.exit(1)
        else:
            print("\n✓ No performance regressions detected")

    except subprocess.CalledProcessError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        subprocess.run(["git", "checkout", "-"], capture_output=True)
        subprocess.run(["git", "stash", "pop"], capture_output=True)


if __name__ == "__main__":
    main()
