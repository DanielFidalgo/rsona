#!/usr/bin/env python3
"""
Performance Analysis and Artifact Generator for rsona vs librosa

This tool analyzes benchmark results and generates visualization artifacts
including charts, CSV exports, and feature parity matrices.

Usage:
    python analyze_performance.py --rsona rsona.json --librosa librosa.json --output-dir artifacts/
    python analyze_performance.py --rsona rsona.json --librosa librosa.json --spec benchmark_spec.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Use non-interactive backend for headless environments
matplotlib.use("Agg")


class PerformanceAnalyzer:
    """Analyzes and compares rsona vs librosa performance."""

    def __init__(self, spec_path: Optional[str] = None):
        """Initialize with optional benchmark spec."""
        self.spec = None
        if spec_path:
            with open(spec_path) as f:
                self.spec = json.load(f)

    def calculate_speedup(self, librosa_time_ms: float, rsona_time_ms: float) -> float:
        """Calculate speedup factor (librosa / rsona)."""
        if rsona_time_ms == 0:
            return float("inf")
        return librosa_time_ms / rsona_time_ms

    def calculate_real_time_factor(
        self, processing_time_ms: float, audio_duration_ms: float
    ) -> float:
        """Calculate real-time factor (processing_time / audio_duration)."""
        if audio_duration_ms == 0:
            return float("inf")
        return processing_time_ms / audio_duration_ms

    def analyze_feature(
        self, feature: str, rsona_data: Dict[str, Any], librosa_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze performance for a single feature."""
        rsona_runtime = rsona_data.get("runtime_ms", {})
        librosa_runtime = librosa_data.get("runtime_ms", {})

        if isinstance(rsona_runtime, dict):
            rsona_mean = rsona_runtime.get("mean", 0)
            rsona_std = rsona_runtime.get("std", 0)
            rsona_median = rsona_runtime.get("median", 0)
        else:
            rsona_mean = rsona_runtime
            rsona_std = 0
            rsona_median = rsona_runtime

        if isinstance(librosa_runtime, dict):
            librosa_mean = librosa_runtime.get("mean", 0)
            librosa_std = librosa_runtime.get("std", 0)
            librosa_median = librosa_runtime.get("median", 0)
        else:
            librosa_mean = librosa_runtime
            librosa_std = 0
            librosa_median = librosa_runtime

        speedup_mean = self.calculate_speedup(librosa_mean, rsona_mean)
        speedup_median = self.calculate_speedup(librosa_median, rsona_median)

        return {
            "feature": feature,
            "rsona": {
                "mean_ms": rsona_mean,
                "median_ms": rsona_median,
                "std_ms": rsona_std,
            },
            "librosa": {
                "mean_ms": librosa_mean,
                "median_ms": librosa_median,
                "std_ms": librosa_std,
            },
            "speedup": {
                "mean": speedup_mean,
                "median": speedup_median,
            },
            "performance_ratio": speedup_mean,
            "absolute_improvement_ms": librosa_mean - rsona_mean,
        }

    def analyze_suite(
        self, rsona_results: Dict[str, Any], librosa_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze performance for a full suite."""
        analyses = {}

        for feature in rsona_results.keys():
            if feature not in librosa_results:
                continue

            analysis = self.analyze_feature(
                feature, rsona_results[feature], librosa_results[feature]
            )
            analyses[feature] = analysis

        return {
            "analysis_type": "full_suite",
            "features": analyses,
            "summary": self._generate_summary(analyses),
        }

    def _generate_summary(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics from analyses."""
        speedups = [a["speedup"]["mean"] for a in analyses.values()]
        rsona_times = [a["rsona"]["mean_ms"] for a in analyses.values()]
        librosa_times = [a["librosa"]["mean_ms"] for a in analyses.values()]

        return {
            "total_features": len(analyses),
            "speedup": {
                "mean": float(np.mean(speedups)),
                "median": float(np.median(speedups)),
                "min": float(np.min(speedups)),
                "max": float(np.max(speedups)),
                "geometric_mean": float(np.exp(np.mean(np.log(speedups)))),
            },
            "total_time": {
                "rsona_ms": sum(rsona_times),
                "librosa_ms": sum(librosa_times),
                "improvement_ms": sum(librosa_times) - sum(rsona_times),
                "improvement_pct": (
                    (sum(librosa_times) - sum(rsona_times)) / sum(librosa_times) * 100
                    if sum(librosa_times) > 0
                    else 0
                ),
            },
            "fastest_features": self._identify_fastest_features(analyses),
            "slowest_features": self._identify_slowest_features(analyses),
        }

    def _identify_fastest_features(
        self, analyses: Dict[str, Any], top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """Identify features where rsona has the biggest speedup."""
        sorted_features = sorted(
            analyses.items(), key=lambda x: x[1]["speedup"]["mean"], reverse=True
        )
        return [
            {
                "feature": f,
                "speedup": a["speedup"]["mean"],
                "rsona_ms": a["rsona"]["mean_ms"],
                "librosa_ms": a["librosa"]["mean_ms"],
            }
            for f, a in sorted_features[:top_n]
        ]

    def _identify_slowest_features(
        self, analyses: Dict[str, Any], top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """Identify features where rsona is slowest (smallest speedup)."""
        sorted_features = sorted(
            analyses.items(), key=lambda x: x[1]["speedup"]["mean"]
        )
        return [
            {
                "feature": f,
                "speedup": a["speedup"]["mean"],
                "rsona_ms": a["rsona"]["mean_ms"],
                "librosa_ms": a["librosa"]["mean_ms"],
            }
            for f, a in sorted_features[:top_n]
        ]

    def generate_feature_parity_matrix(self) -> Dict[str, Any]:
        """Generate feature parity matrix from spec."""
        if not self.spec:
            return {"error": "No spec provided"}

        matrix = []

        for suite in self.spec.get("test_suites", []):
            if suite["name"] == "isolated_features":
                for test in suite["tests"]:
                    matrix.append(
                        {
                            "feature": test["feature"],
                            "rsona": test["rsona_available"],
                            "librosa": test["librosa_available"],
                            "status": test["status"],
                            "benchmark_ready": test["status"] == "ready",
                        }
                    )

        # Count statuses
        status_counts = {}
        for item in matrix:
            status = item["status"]
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "matrix": matrix,
            "summary": {
                "total_features": len(matrix),
                "ready": status_counts.get("ready", 0),
                "coming_soon": status_counts.get("coming_soon", 0),
                "parity_percentage": (
                    status_counts.get("ready", 0) / len(matrix) * 100 if matrix else 0
                ),
            },
        }

    def export_to_csv(self, analyses: Dict[str, Any], output_path: str) -> None:
        """Export analysis results to CSV."""
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Feature",
                    "rsona_mean_ms",
                    "rsona_std_ms",
                    "librosa_mean_ms",
                    "librosa_std_ms",
                    "speedup_mean",
                    "speedup_median",
                    "improvement_ms",
                ]
            )

            for feature, analysis in analyses.items():
                writer.writerow(
                    [
                        feature,
                        analysis["rsona"]["mean_ms"],
                        analysis["rsona"]["std_ms"],
                        analysis["librosa"]["mean_ms"],
                        analysis["librosa"]["std_ms"],
                        analysis["speedup"]["mean"],
                        analysis["speedup"]["median"],
                        analysis["absolute_improvement_ms"],
                    ]
                )

    def generate_runtime_comparison_chart(
        self, analyses: Dict[str, Any], output_path: str
    ) -> None:
        """Generate bar chart comparing runtimes."""
        features = list(analyses.keys())
        rsona_times = [analyses[f]["rsona"]["mean_ms"] for f in features]
        librosa_times = [analyses[f]["librosa"]["mean_ms"] for f in features]

        x = np.arange(len(features))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(
            x - width / 2, rsona_times, width, label="rsona", color="#2E86AB"
        )
        bars2 = ax.bar(
            x + width / 2, librosa_times, width, label="librosa", color="#A23B72"
        )

        ax.set_ylabel("Runtime (ms)", fontsize=12)
        ax.set_title(
            "Runtime Comparison: rsona vs librosa", fontsize=14, fontweight="bold"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    def generate_speedup_chart(
        self, analyses: Dict[str, Any], output_path: str
    ) -> None:
        """Generate bar chart showing speedup factors."""
        features = list(analyses.keys())
        speedups = [analyses[f]["speedup"]["mean"] for f in features]

        fig, ax = plt.subplots(figsize=(12, 6))

        # Color bars based on speedup (green if >1, red if <1)
        colors = ["#06A77D" if s >= 1 else "#D81159" for s in speedups]

        bars = ax.bar(features, speedups, color=colors, alpha=0.8)

        ax.set_ylabel("Speedup Factor", fontsize=12)
        ax.set_title(
            "rsona Speedup vs librosa (>1 = faster)", fontsize=14, fontweight="bold"
        )
        ax.set_xticklabels(features, rotation=45, ha="right")
        ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}x",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    def generate_parity_matrix_chart(
        self, parity_data: Dict[str, Any], output_path: str
    ) -> None:
        """Generate visual feature parity matrix."""
        matrix = parity_data["matrix"]

        features = [item["feature"] for item in matrix]
        rsona = [1 if item["rsona"] else 0 for item in matrix]
        librosa = [1 if item["librosa"] else 0 for item in matrix]
        ready = [1 if item["status"] == "ready" else 0.3 for item in matrix]

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(features))
        width = 0.25

        ax.bar(
            x - width, rsona, width, label="rsona Available", color="#2E86AB", alpha=0.8
        )
        ax.bar(x, librosa, width, label="librosa Available", color="#A23B72", alpha=0.8)
        ax.bar(
            x + width, ready, width, label="Benchmark Ready", color="#06A77D", alpha=0.8
        )

        ax.set_ylabel("Status", fontsize=12)
        ax.set_title("Feature Parity Matrix", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45, ha="right")
        ax.set_ylim(0, 1.2)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Add percentage annotation
        summary = parity_data["summary"]
        ax.text(
            0.02,
            0.98,
            f"Parity: {summary['parity_percentage']:.1f}%\n"
            f"Ready: {summary['ready']}/{summary['total_features']}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    def generate_all_artifacts(
        self,
        rsona_data: Dict[str, Any],
        librosa_data: Dict[str, Any],
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Generate all analysis artifacts."""
        output_dir.mkdir(parents=True, exist_ok=True)

        artifacts = {}

        # Analyze performance
        if rsona_data["benchmark_type"] == "full_suite":
            analysis = self.analyze_suite(
                rsona_data["results"], librosa_data["results"]
            )

            # Save JSON analysis
            analysis_path = output_dir / "performance_analysis.json"
            with open(analysis_path, "w") as f:
                json.dump(analysis, f, indent=2)
            artifacts["analysis_json"] = str(analysis_path)

            # Export CSV
            csv_path = output_dir / "performance_comparison.csv"
            self.export_to_csv(analysis["features"], csv_path)
            artifacts["csv"] = str(csv_path)

            # Generate charts
            runtime_chart = output_dir / "runtime_comparison.png"
            self.generate_runtime_comparison_chart(analysis["features"], runtime_chart)
            artifacts["runtime_chart"] = str(runtime_chart)

            speedup_chart = output_dir / "speedup_comparison.png"
            self.generate_speedup_chart(analysis["features"], speedup_chart)
            artifacts["speedup_chart"] = str(speedup_chart)

        # Generate parity matrix
        if self.spec:
            parity = self.generate_feature_parity_matrix()
            parity_path = output_dir / "feature_parity.json"
            with open(parity_path, "w") as f:
                json.dump(parity, f, indent=2)
            artifacts["parity_json"] = str(parity_path)

            parity_chart = output_dir / "feature_parity_matrix.png"
            self.generate_parity_matrix_chart(parity, parity_chart)
            artifacts["parity_chart"] = str(parity_chart)

        return artifacts


def main():
    parser = argparse.ArgumentParser(
        description="Analyze performance and generate artifacts for rsona vs librosa"
    )
    parser.add_argument(
        "--rsona",
        type=str,
        required=True,
        help="Path to rsona benchmark results JSON",
    )
    parser.add_argument(
        "--librosa",
        type=str,
        required=True,
        help="Path to librosa benchmark results JSON",
    )
    parser.add_argument(
        "--spec",
        type=str,
        help="Path to benchmark specification JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Output directory for artifacts (default: artifacts/)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv", "charts", "all"],
        default="all",
        help="Output format (default: all)",
    )

    args = parser.parse_args()

    # Load data
    with open(args.rsona) as f:
        rsona_data = json.load(f)

    with open(args.librosa) as f:
        librosa_data = json.load(f)

    # Initialize analyzer
    analyzer = PerformanceAnalyzer(args.spec)

    output_dir = Path(args.output_dir)

    # Generate artifacts
    print(f"Generating artifacts in {output_dir}...", file=sys.stderr)
    artifacts = analyzer.generate_all_artifacts(rsona_data, librosa_data, output_dir)

    # Print artifact paths
    print("\nGenerated artifacts:", file=sys.stderr)
    for name, path in artifacts.items():
        print(f"  {name}: {path}", file=sys.stderr)

    # Output summary JSON to stdout
    summary = {
        "artifacts": artifacts,
        "output_directory": str(output_dir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
