#!/usr/bin/env python3
"""
Quick test to verify chroma filter normalization fix.

This script compares chroma magnitudes between librosa and rsona
to verify that the peak normalization produces comparable amplitudes.
"""

import json
import subprocess
import sys

import librosa
import matplotlib.pyplot as plt
import numpy as np


def compute_librosa_chroma(audio_path, sr=22050, n_fft=2048, hop_length=512):
    """Compute chroma using librosa."""
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    return chroma


def compute_rsona_chroma(audio_path):
    """Compute chroma using rsona benchmark runner."""
    try:
        result = subprocess.run(
            [
                "cargo",
                "run",
                "--release",
                "--bin",
                "rsona_runner",
                "--",
                "--audio",
                audio_path,
                "--feature",
                "chroma_stft",
            ],
            capture_output=True,
            text=True,
            cwd=".",
            timeout=30,
        )

        if result.returncode != 0:
            print(f"Error running rsona: {result.stderr}")
            return None

        # Parse JSON output
        output = json.loads(result.stdout)
        chroma_data = output.get("output")

        if chroma_data is None:
            print("No chroma data in output")
            return None

        # Convert to numpy array - rsona outputs (n_chroma, n_frames)
        chroma = np.array(chroma_data)
        return chroma

    except Exception as e:
        print(f"Error computing rsona chroma: {e}")
        return None


def analyze_chroma_magnitudes(chroma, name="Chroma"):
    """Analyze chroma magnitude statistics."""
    print(f"\n{name}:")
    print(f"  Shape: {chroma.shape}")
    print(f"  Mean:  {np.mean(chroma):.6f}")
    print(f"  Std:   {np.std(chroma):.6f}")
    print(f"  Min:   {np.min(chroma):.6f}")
    print(f"  Max:   {np.max(chroma):.6f}")

    # Check how many values are > 0.5 (typical for normalized chroma)
    high_values = np.sum(chroma > 0.5)
    total_values = chroma.size
    percentage = (high_values / total_values) * 100
    print(f"  Values > 0.5: {high_values}/{total_values} ({percentage:.1f}%)")

    return {
        "mean": np.mean(chroma),
        "std": np.std(chroma),
        "min": np.min(chroma),
        "max": np.max(chroma),
        "high_percentage": percentage,
    }


def plot_chroma_comparison(librosa_chroma, rsona_chroma, frame_idx=100):
    """Plot a single frame comparison."""
    plt.figure(figsize=(12, 8))

    # Plot single frame
    plt.subplot(2, 1, 1)
    x = np.arange(12)
    width = 0.35
    plt.bar(
        x - width / 2, librosa_chroma[:, frame_idx], width, label="librosa", alpha=0.8
    )
    plt.bar(x + width / 2, rsona_chroma[:, frame_idx], width, label="rsona", alpha=0.8)
    plt.xlabel("Chroma Bin")
    plt.ylabel("Magnitude")
    plt.title(f"Chroma Comparison - Frame {frame_idx}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(x, ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])

    # Plot time series for one chroma bin
    plt.subplot(2, 1, 2)
    chroma_bin = 7  # G
    plt.plot(
        librosa_chroma[chroma_bin, :500],
        "b-",
        linewidth=1.5,
        label="librosa",
        alpha=0.7,
    )
    plt.plot(
        rsona_chroma[chroma_bin, :500], "r--", linewidth=1.5, label="rsona", alpha=0.7
    )
    plt.xlabel("Frame")
    plt.ylabel("Magnitude")
    plt.title(f"Chroma Bin {chroma_bin} Over Time (First 500 frames)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("chroma_fix_comparison.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison plot to: chroma_fix_comparison.png")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test_chroma_fix.py <audio_file>")
        print(
            "Example: python3 test_chroma_fix.py 'bench/audio/Alex-Productions - Future Bass Technology _ Shades.wav'"
        )
        sys.exit(1)

    audio_path = sys.argv[1]

    print("=" * 70)
    print("CHROMA FILTER FIX VERIFICATION")
    print("=" * 70)
    print(f"\nAudio file: {audio_path}")

    # Compute librosa chroma
    print("\n[1/2] Computing librosa chroma...")
    librosa_chroma = compute_librosa_chroma(audio_path)

    # Compute rsona chroma
    print("[2/2] Computing rsona chroma...")
    rsona_chroma = compute_rsona_chroma(audio_path)

    if rsona_chroma is None:
        print("\nFailed to compute rsona chroma")
        sys.exit(1)

    # Analyze magnitudes
    print("\n" + "=" * 70)
    print("MAGNITUDE ANALYSIS")
    print("=" * 70)

    librosa_stats = analyze_chroma_magnitudes(librosa_chroma, "Librosa Chroma")
    rsona_stats = analyze_chroma_magnitudes(rsona_chroma, "Rsona Chroma")

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    mean_ratio = rsona_stats["mean"] / librosa_stats["mean"]
    max_ratio = rsona_stats["max"] / librosa_stats["max"]

    print(f"\nMean magnitude ratio (rsona/librosa): {mean_ratio:.3f}")
    print(f"Max magnitude ratio (rsona/librosa):  {max_ratio:.3f}")

    # Assessment
    print("\n" + "=" * 70)
    print("ASSESSMENT")
    print("=" * 70)

    if mean_ratio > 0.8 and mean_ratio < 1.2:
        print("✓ PASS: Mean magnitudes are comparable (ratio close to 1.0)")
    else:
        print(
            f"✗ FAIL: Mean magnitudes differ significantly (ratio = {mean_ratio:.3f})"
        )

    if max_ratio > 0.8 and max_ratio < 1.2:
        print("✓ PASS: Max magnitudes are comparable (ratio close to 1.0)")
    else:
        print(f"✗ FAIL: Max magnitudes differ significantly (ratio = {max_ratio:.3f})")

    # Before fix, rsona had ~30% of values > 0.5
    # After fix, should be closer to librosa's percentage
    high_val_diff = abs(
        rsona_stats["high_percentage"] - librosa_stats["high_percentage"]
    )
    if high_val_diff < 20:
        print(
            f"✓ PASS: Similar distribution of high values (diff = {high_val_diff:.1f}%)"
        )
    else:
        print(
            f"✗ FAIL: Different distribution of high values (diff = {high_val_diff:.1f}%)"
        )

    # Plot comparison
    if librosa_chroma.shape == rsona_chroma.shape:
        plot_chroma_comparison(librosa_chroma, rsona_chroma)
    else:
        print(f"\nWarning: Shape mismatch, skipping plot")
        print(f"  Librosa: {librosa_chroma.shape}")
        print(f"  Rsona:   {rsona_chroma.shape}")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
