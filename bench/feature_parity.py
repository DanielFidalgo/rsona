#!/usr/bin/env python3
"""
Feature Parity Validation: rsona vs librosa

Validates that rsona produces accurate results compared to librosa reference
implementation. Checks feature-by-feature accuracy, not just performance.

Usage:
    python feature_parity.py [audio_file] [--tolerance FLOAT] [--output FILE]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np
from scipy.stats import pearsonr


class ParityValidator:
    """Validates feature parity between rsona and librosa."""

    def __init__(self, tolerance: float = 0.01):
        """
        Initialize validator.

        Args:
            tolerance: Relative tolerance for comparisons (e.g., 0.01 = 1%)
        """
        self.tolerance = tolerance
        self.results = {}

    def calculate_mae(self, arr1: np.ndarray, arr2: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return np.mean(np.abs(arr1 - arr2))

    def calculate_mse(self, arr1: np.ndarray, arr2: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        return np.mean((arr1 - arr2) ** 2)

    def calculate_correlation(self, arr1: np.ndarray, arr2: np.ndarray) -> float:
        """Calculate Pearson correlation coefficient."""
        if arr1.size != arr2.size:
            return 0.0
        flat1 = arr1.flatten()
        flat2 = arr2.flatten()
        if np.std(flat1) == 0 or np.std(flat2) == 0:
            return 1.0 if np.allclose(flat1, flat2) else 0.0
        corr, _ = pearsonr(flat1, flat2)
        return corr

    def validate_shapes(self, name: str, arr1: np.ndarray, arr2: np.ndarray) -> bool:
        """Validate that arrays have the same shape."""
        if arr1.shape != arr2.shape:
            self.results[name] = {
                "pass": False,
                "reason": f"Shape mismatch: rsona {arr1.shape} vs librosa {arr2.shape}",
                "rsona_shape": arr1.shape,
                "librosa_shape": arr2.shape,
            }
            return False
        return True

    def validate_feature(
        self,
        name: str,
        rsona_data: np.ndarray,
        librosa_data: np.ndarray,
        description: str = "",
    ) -> Dict[str, Any]:
        """
        Validate a single feature.

        Returns validation metrics and pass/fail status.
        """
        if not self.validate_shapes(name, rsona_data, librosa_data):
            return self.results[name]

        mae = self.calculate_mae(rsona_data, librosa_data)
        mse = self.calculate_mse(rsona_data, librosa_data)
        correlation = self.calculate_correlation(rsona_data, librosa_data)
        max_diff = np.max(np.abs(rsona_data - librosa_data))

        # Calculate relative error
        mean_val = np.mean(np.abs(librosa_data))
        relative_error = mae / mean_val if mean_val > 1e-10 else mae

        # Determine pass/fail
        passed = (
            correlation > 0.99
            and relative_error < self.tolerance
            and max_diff < mean_val * self.tolerance * 10
        )

        result = {
            "pass": passed,
            "description": description,
            "shape": rsona_data.shape,
            "mae": float(mae),
            "mse": float(mse),
            "correlation": float(correlation),
            "max_diff": float(max_diff),
            "relative_error": float(relative_error),
            "relative_error_pct": float(relative_error * 100),
        }

        self.results[name] = result
        return result


def run_librosa_pipeline(audio_path: str) -> Dict[str, Any]:
    """Run librosa pipeline and extract all features."""
    print("Running librosa pipeline...")

    # Load audio
    y, sr = librosa.load(audio_path, mono=True, sr=None)

    # Parameters matching rsona defaults
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    n_mfcc = 20

    # STFT
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_mag = np.abs(S)
    S_power = S_mag**2

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        S=S_power, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )

    # MFCC
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=n_mfcc)

    # RMS
    rms = librosa.feature.rms(y=y, hop_length=hop_length)

    # Onset Strength
    onset_env = librosa.onset.onset_strength(S=mel, sr=sr, hop_length=hop_length)

    # Tempo
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length)[
        0
    ]

    return {
        "sample_rate": sr,
        "n_samples": len(y),
        "stft_mag": S_mag,
        "stft_power": S_power,
        "mel": mel,
        "mfcc": mfcc,
        "rms": rms,
        "onset": onset_env,
        "tempo_bpm": float(tempo),
        "n_frames": S_mag.shape[1],
        "n_frequency_bins": S_mag.shape[0],
    }


def run_rsona_pipeline(audio_path: str) -> Dict[str, Any]:
    """Run rsona pipeline and extract all features using --features flag."""
    print("Running rsona pipeline...")

    bench_dir = Path(__file__).parent
    project_root = bench_dir.parent

    # Use rsona_bench with --features flag to get full feature arrays
    result = subprocess.run(
        [
            "cargo",
            "run",
            "--release",
            "--bin",
            "rsona_bench",
            "--",
            audio_path,
            "--features",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"rsona_bench failed: {result.stderr}")

    # Parse JSON from output
    data = json.loads(result.stdout)

    # Extract features and convert to numpy arrays
    features = data["features"]
    audio_info = data["audio_info"]

    # Convert nested lists to numpy arrays
    # Features are in [frames, bins] format from rsona
    stft_mag = np.array(
        features["stft_magnitude"], dtype=np.float32
    ).T  # Transpose to [bins, frames]
    mel = np.array(
        features["mel_spectrogram"], dtype=np.float32
    ).T  # Transpose to [mels, frames]
    mfcc = np.array(
        features["mfcc"], dtype=np.float32
    ).T  # Transpose to [coeffs, frames]
    rms = np.array(features["rms"], dtype=np.float32)
    onset = np.array(features["onset_envelope"], dtype=np.float32)

    return {
        "sample_rate": audio_info["sample_rate"],
        "n_samples": audio_info["n_samples"],
        "stft_mag": stft_mag,
        "mel": mel,
        "mfcc": mfcc,
        "rms": rms,
        "onset": onset,
        "tempo_bpm": features["tempo_bpm"],
        "n_frames": stft_mag.shape[1],
        "n_frequency_bins": stft_mag.shape[0],
    }


def generate_report(
    validator: ParityValidator, audio_path: str, output_file: str = None
) -> str:
    """Generate markdown report of parity validation."""
    lines = []

    lines.append("# rsona Feature Parity Report")
    lines.append("")
    lines.append(f"**Audio File:** `{audio_path}`")
    lines.append(f"**Tolerance:** {validator.tolerance * 100:.2f}%")
    lines.append("")

    # Summary
    total = len(validator.results)
    passed = sum(1 for r in validator.results.values() if r.get("pass", False))

    if passed == total:
        lines.append(f"## ✅ Overall Status: PASS ({passed}/{total})")
    else:
        lines.append(f"## ⚠️ Overall Status: FAIL ({passed}/{total} passed)")
    lines.append("")

    # Feature-by-feature results
    lines.append("## Feature-by-Feature Results")
    lines.append("")
    lines.append("| Feature | Status | Correlation | MAE | Relative Error |")
    lines.append("|---------|--------|-------------|-----|----------------|")

    for name, result in validator.results.items():
        if "reason" in result:
            # Shape mismatch
            lines.append(f"| {name} | ❌ | - | - | {result['reason']} |")
        else:
            status = "✅" if result["pass"] else "⚠️"
            corr = f"{result['correlation']:.4f}"
            mae = f"{result['mae']:.6f}"
            rel_err = f"{result['relative_error_pct']:.2f}%"
            lines.append(f"| {name} | {status} | {corr} | {mae} | {rel_err} |")

    lines.append("")

    # Detailed results
    lines.append("## Detailed Metrics")
    lines.append("")

    for name, result in validator.results.items():
        if "reason" in result:
            lines.append(f"### ❌ {name}")
            lines.append(f"**Issue:** {result['reason']}")
        else:
            status = "✅" if result["pass"] else "⚠️"
            lines.append(f"### {status} {name}")
            if result.get("description"):
                lines.append(f"*{result['description']}*")
            lines.append("")
            lines.append(f"- **Shape:** {result['shape']}")
            lines.append(f"- **Correlation:** {result['correlation']:.6f}")
            lines.append(f"- **MAE:** {result['mae']:.6f}")
            lines.append(f"- **MSE:** {result['mse']:.6f}")
            lines.append(f"- **Max Difference:** {result['max_diff']:.6f}")
            lines.append(f"- **Relative Error:** {result['relative_error_pct']:.2f}%")

            if not result["pass"]:
                lines.append("")
                lines.append("**Why Failed:**")
                if result["correlation"] <= 0.99:
                    lines.append(f"- Correlation {result['correlation']:.4f} ≤ 0.99")
                if result["relative_error"] >= validator.tolerance:
                    lines.append(
                        f"- Relative error {result['relative_error_pct']:.2f}% ≥ {validator.tolerance * 100}%"
                    )

        lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")

    if passed == total:
        lines.append("✅ **All features pass parity validation!**")
        lines.append("")
        lines.append("rsona produces results that match librosa within tolerance.")
    else:
        lines.append("⚠️ **Some features need investigation:**")
        lines.append("")
        for name, result in validator.results.items():
            if not result.get("pass", False):
                if "reason" in result:
                    lines.append(f"- **{name}**: {result['reason']}")
                else:
                    lines.append(
                        f"- **{name}**: {result['relative_error_pct']:.2f}% error (limit: {validator.tolerance * 100}%)"
                    )

    lines.append("")
    lines.append("---")
    lines.append("*Generated by rsona feature parity validation*")

    report = "\n".join(lines)

    if output_file:
        with open(output_file, "w") as f:
            f.write(report)
        print(f"\n✓ Report saved to: {output_file}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Validate feature parity between rsona and librosa"
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        default="test_audio.wav",
        help="Audio file to test (default: test_audio.wav)",
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        type=float,
        default=0.01,
        help="Relative error tolerance (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="FEATURE_PARITY.md",
        help="Output markdown file (default: FEATURE_PARITY.md)",
    )

    args = parser.parse_args()

    if not Path(args.audio_file).exists():
        print(f"Error: Audio file not found: {args.audio_file}")
        sys.exit(1)

    print("=" * 60)
    print("rsona Feature Parity Validation")
    print("=" * 60)
    print()

    validator = ParityValidator(tolerance=args.tolerance)

    # Run both pipelines
    librosa_data = run_librosa_pipeline(args.audio_file)
    print()

    rsona_data = run_rsona_pipeline(args.audio_file)
    print()

    # Show what was extracted
    print("Librosa extracted:")
    print(f"  - Sample rate: {librosa_data['sample_rate']} Hz")
    print(f"  - Samples: {librosa_data['n_samples']}")
    print(f"  - Frames: {librosa_data['n_frames']}")
    print(f"  - Frequency bins: {librosa_data['n_frequency_bins']}")
    print(f"  - STFT shape: {librosa_data['stft_mag'].shape}")
    print(f"  - Mel shape: {librosa_data['mel'].shape}")
    print(f"  - MFCC shape: {librosa_data['mfcc'].shape}")
    print(f"  - RMS shape: {librosa_data['rms'].shape}")
    print(f"  - Onset envelope length: {len(librosa_data['onset'])}")
    print(f"  - Tempo: {librosa_data['tempo_bpm']:.2f} BPM")
    print()

    print("rsona extracted:")
    print(f"  - Sample rate: {rsona_data['sample_rate']} Hz")
    print(f"  - Samples: {rsona_data['n_samples']}")
    print(f"  - Frames: {rsona_data['n_frames']}")
    print(f"  - Frequency bins: {rsona_data['n_frequency_bins']}")
    print(f"  - STFT shape: {rsona_data['stft_mag'].shape}")
    print(f"  - Mel shape: {rsona_data['mel'].shape}")
    print(f"  - MFCC shape: {rsona_data['mfcc'].shape}")
    print(f"  - RMS shape: {rsona_data['rms'].shape}")
    print(f"  - Onset envelope length: {len(rsona_data['onset'])}")
    print(f"  - Tempo: {rsona_data['tempo_bpm']:.2f} BPM")
    print()

    # Validate each feature
    print("Validating features...")
    print()

    validator.validate_feature(
        "stft_magnitude",
        rsona_data["stft_mag"],
        librosa_data["stft_mag"],
        "Short-Time Fourier Transform magnitude spectrum",
    )

    validator.validate_feature(
        "mel_spectrogram",
        rsona_data["mel"],
        librosa_data["mel"],
        "Mel-frequency spectrogram",
    )

    validator.validate_feature(
        "mfcc",
        rsona_data["mfcc"],
        librosa_data["mfcc"],
        "Mel-Frequency Cepstral Coefficients",
    )

    validator.validate_feature(
        "rms", rsona_data["rms"], librosa_data["rms"], "Root Mean Square energy"
    )

    validator.validate_feature(
        "onset_envelope",
        rsona_data["onset"],
        librosa_data["onset"],
        "Onset strength envelope",
    )

    # Validate tempo
    tempo_diff = abs(rsona_data["tempo_bpm"] - librosa_data["tempo_bpm"])
    tempo_diff_pct = (tempo_diff / librosa_data["tempo_bpm"]) * 100
    tempo_pass = tempo_diff_pct <= (validator.tolerance * 100)

    validator.results["tempo"] = {
        "pass": tempo_pass,
        "rsona_value": rsona_data["tempo_bpm"],
        "librosa_value": librosa_data["tempo_bpm"],
        "difference_pct": tempo_diff_pct,
        "description": "Tempo estimation (BPM)",
    }

    print(
        f"Tempo: rsona={rsona_data['tempo_bpm']:.2f} BPM, "
        f"librosa={librosa_data['tempo_bpm']:.2f} BPM, "
        f"diff={tempo_diff_pct:.2f}% {'✓' if tempo_pass else '✗'}"
    )
    print()

    # Generate report
    report = generate_report(validator, args.audio_file, args.output)
    print(report)

    # Save to file if specified
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
