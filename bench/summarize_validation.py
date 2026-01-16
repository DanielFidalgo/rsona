#!/usr/bin/env python3
"""
Summarize validation results from rsona vs librosa benchmark.
"""

import json
import sys
from pathlib import Path


def load_validation_results(filepath):
    """Load validation results from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def format_metric(value, decimals=4):
    """Format a metric value, handling N/A cases."""
    if value is None or value == "N/A":
        return "N/A".ljust(8)
    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}".ljust(8)
    return str(value).ljust(8)


def print_feature_results(features):
    """Print detailed results for each feature."""
    print("\n" + "=" * 80)
    print("FEATURE VALIDATION RESULTS")
    print("=" * 80)
    print(f"{'Feature':<20} {'Status':<10} {'Correlation':<12} {'MAE':<12} {'Notes'}")
    print("-" * 80)

    for feature_name, feature_data in sorted(features.items()):
        # Determine pass/fail status
        passed_data = feature_data.get("passed", {})
        if isinstance(passed_data, bool):
            overall_pass = passed_data
        elif isinstance(passed_data, dict):
            overall_pass = passed_data.get("overall", False)
        else:
            overall_pass = False

        status = "✅ PASS" if overall_pass else "❌ FAIL"

        # Get metrics
        correlation = feature_data.get("correlation", "N/A")
        mae = feature_data.get("mae", "N/A")

        # Format correlation and MAE
        corr_str = format_metric(correlation)
        mae_str = format_metric(mae)

        # Determine notes
        notes = []
        if isinstance(passed_data, dict):
            if not passed_data.get("mae", True):
                notes.append("MAE too high")
            if not passed_data.get("correlation", True):
                notes.append("Low correlation")
        notes_str = ", ".join(notes) if notes else ""

        print(
            f"{feature_name:<20} {status:<10} {corr_str:<12} {mae_str:<12} {notes_str}"
        )


def print_summary(summary):
    """Print overall summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total = summary.get("total_features", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    pass_rate = summary.get("pass_rate", 0) * 100

    print(f"Total Features:  {total}")
    print(f"Passed:          {passed} ({pass_rate:.1f}%)")
    print(f"Failed:          {failed}")

    # MAE statistics
    mae_stats = summary.get("mae", {})
    if mae_stats:
        print(f"\nMAE Statistics:")
        print(f"  Mean:   {mae_stats.get('mean', 0):.4f}")
        print(f"  Median: {mae_stats.get('median', 0):.4f}")
        print(f"  Min:    {mae_stats.get('min', 0):.4f}")
        print(f"  Max:    {mae_stats.get('max', 0):.4f}")

    # Correlation statistics
    corr_stats = summary.get("correlation", {})
    if corr_stats:
        print(f"\nCorrelation Statistics:")
        print(f"  Mean:   {corr_stats.get('mean', 0):.4f}")
        print(f"  Median: {corr_stats.get('median', 0):.4f}")
        print(f"  Min:    {corr_stats.get('min', 0):.4f}")
        print(f"  Max:    {corr_stats.get('max', 0):.4f}")


def print_failed_details(features):
    """Print detailed information about failed features."""
    failed = {
        name: data
        for name, data in features.items()
        if not (
            data.get("passed", {}).get("overall", False)
            if isinstance(data.get("passed", {}), dict)
            else data.get("passed", False)
        )
    }

    if not failed:
        print("\n🎉 All features passed validation!")
        return

    print("\n" + "=" * 80)
    print("FAILED FEATURES DETAILS")
    print("=" * 80)

    for feature_name, feature_data in failed.items():
        print(f"\n{feature_name}:")
        print("-" * 40)

        correlation = feature_data.get("correlation", "N/A")
        mae = feature_data.get("mae", "N/A")
        relative_error = feature_data.get("relative_error", "N/A")

        print(f"  Correlation:    {format_metric(correlation).strip()}")
        print(f"  MAE:            {format_metric(mae).strip()}")
        if relative_error != "N/A":
            print(f"  Relative Error: {format_metric(relative_error).strip()}%")

        # Show thresholds
        thresholds = feature_data.get("thresholds", {})
        if thresholds:
            print(f"\n  Thresholds:")
            for key, value in thresholds.items():
                print(f"    {key}: {value}")

        # Show what failed
        passed_data = feature_data.get("passed", {})
        if isinstance(passed_data, dict):
            print(f"\n  Failed checks:")
            for check, result in passed_data.items():
                if check != "overall" and not result:
                    print(f"    ❌ {check}")


def main():
    """Main entry point."""
    # Default path
    default_path = (
        Path(__file__).parent
        / "librosa_comparison"
        / "artifacts"
        / "validation_results.json"
    )

    # Check if path provided as argument
    if len(sys.argv) > 1:
        filepath = Path(sys.argv[1])
    else:
        filepath = default_path

    if not filepath.exists():
        print(f"Error: Validation results file not found: {filepath}")
        print(f"\nUsage: python {sys.argv[0]} [path/to/validation_results.json]")
        sys.exit(1)

    # Load and display results
    try:
        results = load_validation_results(filepath)

        # Print feature results
        features = results.get("features", {})
        print_feature_results(features)

        # Print summary
        summary = results.get("summary", {})
        print_summary(summary)

        # Print failed details
        print_failed_details(features)

        print("\n" + "=" * 80)

    except Exception as e:
        print(f"Error processing validation results: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
