#!/usr/bin/env python3
"""
Correctness Validation Tool for rsona vs librosa Comparison

This tool compares the numerical outputs of rsona and librosa to ensure
data correctness alongside performance benchmarking.

Usage:
    python validate_correctness.py --rsona rsona_output.json --librosa librosa_output.json
    python validate_correctness.py --rsona rsona_output.json --librosa librosa_output.json --spec benchmark_spec.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr


class CorrectnessValidator:
    """Validates correctness of rsona outputs against librosa."""

    def __init__(self, spec_path: Optional[str] = None):
        """Initialize with optional benchmark spec for thresholds."""
        self.spec = None
        self.thresholds = {}

        if spec_path:
            with open(spec_path) as f:
                self.spec = json.load(f)
                self._load_thresholds()

    def _load_thresholds(self):
        """Load correctness thresholds from spec."""
        if not self.spec:
            return

        for suite in self.spec.get("test_suites", []):
            if suite["name"] == "isolated_features":
                for test in suite["tests"]:
                    feature = test["feature"]
                    if "correctness" in test:
                        self.thresholds[feature] = test["correctness"]

    def calculate_mae(self, arr1: np.ndarray, arr2: np.ndarray) -> float:
        """Calculate Mean Absolute Error between two arrays."""
        if arr1.shape != arr2.shape:
            # Try to handle different shapes (e.g., (1, N) vs (N,))
            arr1 = np.asarray(arr1).flatten()
            arr2 = np.asarray(arr2).flatten()

        if arr1.shape != arr2.shape:
            # Truncate to shorter length
            min_len = min(len(arr1), len(arr2))
            print(
                f"Warning: Shape mismatch {arr1.shape} vs {arr2.shape}. "
                f"Truncating to {min_len} elements for comparison.",
                file=sys.stderr,
            )
            arr1 = arr1[:min_len]
            arr2 = arr2[:min_len]

        return float(np.mean(np.abs(arr1 - arr2)))

    def calculate_correlation(
        self, arr1: np.ndarray, arr2: np.ndarray
    ) -> Tuple[float, float]:
        """Calculate Pearson correlation between two arrays."""
        arr1 = np.asarray(arr1).flatten()
        arr2 = np.asarray(arr2).flatten()

        if len(arr1) != len(arr2):
            raise ValueError(
                f"Length mismatch: {len(arr1)} vs {len(arr2)}. Cannot compute correlation."
            )

        if len(arr1) < 2:
            return 0.0, 1.0

        # Remove NaN values
        mask = ~(np.isnan(arr1) | np.isnan(arr2))
        if not np.any(mask):
            return 0.0, 1.0

        arr1_clean = arr1[mask]
        arr2_clean = arr2[mask]

        if len(arr1_clean) < 2:
            return 0.0, 1.0

        try:
            corr, p_value = pearsonr(arr1_clean, arr2_clean)
            return float(corr), float(p_value)
        except Exception:
            return 0.0, 1.0

    def calculate_rmse(self, arr1: np.ndarray, arr2: np.ndarray) -> float:
        """Calculate Root Mean Square Error between two arrays."""
        arr1 = np.asarray(arr1).flatten()
        arr2 = np.asarray(arr2).flatten()

        if arr1.shape != arr2.shape:
            min_len = min(len(arr1), len(arr2))
            arr1 = arr1[:min_len]
            arr2 = arr2[:min_len]

        return float(np.sqrt(np.mean((arr1 - arr2) ** 2)))

    def calculate_relative_error(self, arr1: np.ndarray, arr2: np.ndarray) -> float:
        """Calculate mean relative error between two arrays."""
        arr1 = np.asarray(arr1).flatten()
        arr2 = np.asarray(arr2).flatten()

        if arr1.shape != arr2.shape:
            min_len = min(len(arr1), len(arr2))
            arr1 = arr1[:min_len]
            arr2 = arr2[:min_len]

        # Avoid division by zero
        mask = arr2 != 0
        if not np.any(mask):
            return 0.0

        relative_errors = np.abs((arr1[mask] - arr2[mask]) / arr2[mask])
        return float(np.mean(relative_errors))

    def validate_tempo(
        self, rsona_tempo: float, librosa_tempo: float
    ) -> Dict[str, Any]:
        """Validate tempo estimation."""
        mae = abs(rsona_tempo - librosa_tempo)
        relative_error = abs(rsona_tempo - librosa_tempo) / librosa_tempo

        threshold = self.thresholds.get("tempo", {}).get("mae_threshold", 2.0)
        passed = mae <= threshold

        return {
            "feature": "tempo",
            "mae": mae,
            "relative_error": relative_error * 100.0,  # as percentage
            "rsona_value": rsona_tempo,
            "librosa_value": librosa_tempo,
            "threshold": threshold,
            "passed": passed,
            "metrics": {
                "mae": mae,
                "relative_error_pct": relative_error * 100.0,
            },
        }

    def validate_array_feature(
        self, feature: str, rsona_output: List[float], librosa_output: List[float]
    ) -> Dict[str, Any]:
        """Validate array-based feature (e.g., spectral features)."""
        rsona_arr = np.array(rsona_output)
        librosa_arr = np.array(librosa_output)

        # Calculate metrics
        try:
            mae = self.calculate_mae(rsona_arr, librosa_arr)
            rmse = self.calculate_rmse(rsona_arr, librosa_arr)
            correlation, p_value = self.calculate_correlation(rsona_arr, librosa_arr)
            relative_error = self.calculate_relative_error(rsona_arr, librosa_arr)
        except Exception as e:
            return {
                "feature": feature,
                "error": f"Failed to calculate metrics: {str(e)}",
                "shape_rsona": list(rsona_arr.shape),
                "shape_librosa": list(librosa_arr.shape),
                "passed": {"overall": False},
            }

        # Check thresholds
        thresholds = self.thresholds.get(feature, {})
        mae_threshold = thresholds.get("mae_threshold", float("inf"))
        corr_threshold = thresholds.get("correlation_threshold", 0.0)

        mae_passed = mae <= mae_threshold
        corr_passed = (
            correlation >= corr_threshold if not np.isnan(correlation) else True
        )

        passed = mae_passed and corr_passed

        return {
            "feature": feature,
            "mae": mae,
            "rmse": rmse,
            "correlation": correlation,
            "correlation_p_value": p_value,
            "relative_error": relative_error * 100.0,
            "thresholds": {
                "mae": mae_threshold,
                "correlation": corr_threshold,
            },
            "passed": {
                "mae": mae_passed,
                "correlation": corr_passed,
                "overall": passed,
            },
            "metrics": {
                "mae": mae,
                "rmse": rmse,
                "correlation": correlation,
                "relative_error_pct": relative_error * 100.0,
            },
        }

    def validate_matrix_feature(
        self,
        feature: str,
        rsona_output: List[List[float]],
        librosa_output: List[List[float]],
    ) -> Dict[str, Any]:
        """Validate matrix-based feature (e.g., MFCC, chroma)."""
        rsona_arr = np.array(rsona_output)
        librosa_arr = np.array(librosa_output)

        # librosa often returns (n_features, n_frames), rsona may return (n_frames, n_features)
        # Try to align shapes
        if rsona_arr.shape != librosa_arr.shape:
            if rsona_arr.shape == librosa_arr.shape[::-1]:
                rsona_arr = rsona_arr.T
            else:
                # Flatten and compare
                rsona_arr = rsona_arr.flatten()
                librosa_arr = librosa_arr.flatten()

        try:
            mae = self.calculate_mae(rsona_arr, librosa_arr)
            rmse = self.calculate_rmse(rsona_arr, librosa_arr)
            correlation, p_value = self.calculate_correlation(rsona_arr, librosa_arr)
            relative_error = self.calculate_relative_error(rsona_arr, librosa_arr)
        except Exception as e:
            return {
                "feature": feature,
                "error": f"Failed to calculate metrics: {str(e)}",
                "shape_rsona": list(rsona_arr.shape),
                "shape_librosa": list(librosa_arr.shape),
                "passed": {"overall": False},
            }

        # Check thresholds
        thresholds = self.thresholds.get(feature, {})
        mae_threshold = thresholds.get("mae_threshold", float("inf"))
        corr_threshold = thresholds.get("correlation_threshold", 0.0)

        mae_passed = mae <= mae_threshold
        corr_passed = (
            correlation >= corr_threshold if not np.isnan(correlation) else True
        )

        passed = mae_passed and corr_passed

        return {
            "feature": feature,
            "mae": mae,
            "rmse": rmse,
            "correlation": correlation,
            "correlation_p_value": p_value,
            "relative_error": relative_error * 100.0,
            "shape_rsona": list(rsona_arr.shape),
            "shape_librosa": list(librosa_arr.shape),
            "thresholds": {
                "mae": mae_threshold,
                "correlation": corr_threshold,
            },
            "passed": {
                "mae": mae_passed,
                "correlation": corr_passed,
                "overall": passed,
            },
            "metrics": {
                "mae": mae,
                "rmse": rmse,
                "correlation": correlation,
                "relative_error_pct": relative_error * 100.0,
            },
        }

    def validate_single_feature(
        self, feature: str, rsona_data: Dict[str, Any], librosa_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate a single feature comparison."""
        rsona_output = rsona_data.get("output")
        librosa_output = librosa_data.get("output")

        if rsona_output is None or librosa_output is None:
            return {
                "feature": feature,
                "error": "Missing output data",
                "passed": {"overall": False},
            }

        # Handle tempo (scalar)
        if feature == "tempo":
            return self.validate_tempo(rsona_output, librosa_output)

        # Check if rsona has shape metadata (for 2D features that are flattened in JSON)
        rsona_shape = None
        if "result_summary" in rsona_data and "shape" in rsona_data["result_summary"]:
            rsona_shape = tuple(rsona_data["result_summary"]["shape"])

        librosa_shape = None
        if (
            "result_summary" in librosa_data
            and "shape" in librosa_data["result_summary"]
        ):
            librosa_shape = tuple(librosa_data["result_summary"]["shape"])

        # Handle arrays
        if isinstance(rsona_output, list):
            # Check if this should be a matrix based on shape metadata
            if rsona_shape and len(rsona_shape) == 2:
                # Reshape rsona output
                rsona_output = np.array(rsona_output).reshape(rsona_shape)
                # Also reshape librosa if needed
                if librosa_shape and len(librosa_shape) == 2:
                    librosa_output = np.array(librosa_output).reshape(librosa_shape)
                elif not isinstance(librosa_output[0], list):
                    # librosa is flat but should be 2D
                    librosa_output = np.array(librosa_output)
                    # Try to infer shape or use rsona's transposed shape
                    if librosa_output.size == rsona_output.size:
                        # Assume librosa might be transposed
                        try_shape = (rsona_shape[1], rsona_shape[0])
                        if np.prod(try_shape) == librosa_output.size:
                            librosa_output = librosa_output.reshape(try_shape)

                return self.validate_matrix_feature(
                    feature,
                    rsona_output.tolist(),
                    librosa_output.tolist()
                    if isinstance(librosa_output, np.ndarray)
                    else librosa_output,
                )
            elif isinstance(rsona_output[0], list):
                # Matrix
                return self.validate_matrix_feature(
                    feature, rsona_output, librosa_output
                )
            else:
                # 1D array
                return self.validate_array_feature(
                    feature, rsona_output, librosa_output
                )

        return {
            "feature": feature,
            "error": "Unknown output type",
            "passed": {"overall": False},
        }

    def _get_passed_status(self, validation: Dict[str, Any]) -> bool:
        """Extract passed status from validation result, handling both dict and bool formats."""
        passed = validation.get("passed", False)
        if isinstance(passed, dict):
            return passed.get("overall", False)
        return bool(passed)

    def validate_suite(
        self, rsona_results: Dict[str, Any], librosa_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate a full suite of features."""
        validations = {}
        all_passed = True

        for feature in rsona_results.keys():
            if feature not in librosa_results:
                validations[feature] = {
                    "feature": feature,
                    "error": "Not found in librosa results",
                    "passed": {"overall": False},
                }
                all_passed = False
                continue

            rsona_data = rsona_results[feature]
            librosa_data = librosa_results[feature]

            validation = self.validate_single_feature(feature, rsona_data, librosa_data)
            validations[feature] = validation

            if not self._get_passed_status(validation):
                all_passed = False

        return {
            "validation_type": "full_suite",
            "total_features": len(validations),
            "passed": all_passed,
            "features": validations,
            "summary": self._generate_summary(validations),
        }

    def _generate_summary(self, validations: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics from validations."""
        total = len(validations)
        passed = sum(1 for v in validations.values() if self._get_passed_status(v))
        failed = total - passed

        # Aggregate metrics
        maes = []
        correlations = []

        for v in validations.values():
            if "mae" in v:
                maes.append(v["mae"])
            if "correlation" in v:
                correlations.append(v["correlation"])

        summary = {
            "total_features": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0.0,
        }

        if maes:
            summary["mae"] = {
                "mean": float(np.mean(maes)),
                "median": float(np.median(maes)),
                "max": float(np.max(maes)),
            }

        if correlations:
            correlations = [c for c in correlations if not np.isnan(c)]
            if correlations:
                summary["correlation"] = {
                    "mean": float(np.mean(correlations)),
                    "median": float(np.median(correlations)),
                    "min": float(np.min(correlations)),
                }

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Validate correctness of rsona vs librosa outputs"
    )
    parser.add_argument(
        "--rsona",
        type=str,
        required=True,
        help="Path to rsona output JSON",
    )
    parser.add_argument(
        "--librosa",
        type=str,
        required=True,
        help="Path to librosa output JSON",
    )
    parser.add_argument(
        "--spec",
        type=str,
        help="Path to benchmark specification JSON (for thresholds)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file (default: stdout)",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit with non-zero status if validation fails",
    )

    args = parser.parse_args()

    # Load data
    with open(args.rsona) as f:
        rsona_data = json.load(f)

    with open(args.librosa) as f:
        librosa_data = json.load(f)

    # Initialize validator
    validator = CorrectnessValidator(args.spec)

    # Determine validation type
    rsona_type = rsona_data.get("benchmark_type")
    librosa_type = librosa_data.get("benchmark_type")

    if rsona_type != librosa_type:
        print(
            f"Error: Benchmark types don't match: {rsona_type} vs {librosa_type}",
            file=sys.stderr,
        )
        sys.exit(1)

    if rsona_type == "single_feature":
        # Single feature validation
        feature = rsona_data["feature"]
        result = validator.validate_single_feature(
            feature, rsona_data["statistics"], librosa_data["statistics"]
        )
        output_data = {
            "validation_type": "single_feature",
            "feature": feature,
            "result": result,
            "passed": result.get("passed", {}).get("overall", False),
        }
    elif rsona_type == "full_suite":
        # Full suite validation
        result = validator.validate_suite(
            rsona_data["results"], librosa_data["results"]
        )
        output_data = result
    else:
        print(f"Error: Unknown benchmark type: {rsona_type}", file=sys.stderr)
        sys.exit(1)

    # Output results
    output_json = json.dumps(output_data, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"Validation results written to {args.output}", file=sys.stderr)
    else:
        print(output_json)

    # Exit with error if validation failed and --fail-on-error is set
    if args.fail_on_error and not output_data.get("passed", True):
        sys.exit(1)


if __name__ == "__main__":
    main()
