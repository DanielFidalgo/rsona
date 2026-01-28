#!/usr/bin/env python3
"""
Detailed comparison of rsona vs librosa chroma_stft to find root cause of differences.
"""

import json
import subprocess
import sys
import tempfile

import librosa
import numpy as np


def compute_librosa_chroma(audio_path):
    """Compute chroma with librosa and return detailed information."""
    print("=" * 70)
    print("LIBROSA CHROMA COMPUTATION")
    print("=" * 70)

    # Load audio
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    print(f"Audio: {len(y)} samples at {sr} Hz ({len(y) / sr:.2f}s)")

    # Compute STFT
    n_fft = 2048
    hop_length = 512
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=True)
    print(f"STFT shape: {S.shape}")
    print(f"STFT magnitude mean: {np.mean(np.abs(S)):.6f}")
    print(f"STFT magnitude std: {np.std(np.abs(S)):.6f}")
    print(f"STFT magnitude max: {np.max(np.abs(S)):.6f}")

    # Compute chroma without normalization
    chroma_no_norm = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, norm=None
    )
    print(f"\nChroma (no norm) shape: {chroma_no_norm.shape}")
    print(f"Chroma (no norm) mean: {np.mean(chroma_no_norm):.6f}")
    print(f"Chroma (no norm) std: {np.std(chroma_no_norm):.6f}")
    print(f"Chroma (no norm) max: {np.max(chroma_no_norm):.6f}")

    # Compute chroma with default normalization
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, norm=np.inf
    )
    print(f"\nChroma (norm=inf) shape: {chroma.shape}")
    print(f"Chroma (norm=inf) mean: {np.mean(chroma):.6f}")
    print(f"Chroma (norm=inf) std: {np.std(chroma):.6f}")
    print(f"Chroma (norm=inf) max: {np.max(chroma):.6f}")

    # Check frame-wise max normalization
    frame_maxes = np.max(chroma, axis=0)
    print(f"\nFrame max values:")
    print(f"  All frames have max=1.0: {np.allclose(frame_maxes, 1.0)}")
    print(f"  Min of frame maxes: {np.min(frame_maxes):.6f}")
    print(f"  Max of frame maxes: {np.max(frame_maxes):.6f}")

    # Show sample frames
    print(f"\nSample frame 100 (unnormalized):")
    print(f"  Values: {chroma_no_norm[:, 100]}")
    print(f"  Max: {np.max(chroma_no_norm[:, 100]):.6f}")

    print(f"\nSample frame 100 (normalized):")
    print(f"  Values: {chroma[:, 100]}")
    print(f"  Max: {np.max(chroma[:, 100]):.6f}")
    print(f"  Sum: {np.sum(chroma[:, 100]):.6f}")

    return {
        "chroma": chroma,
        "chroma_no_norm": chroma_no_norm,
        "stft": S,
        "y": y,
        "sr": sr,
    }


def analyze_rsona_chroma(rsona_results_path, librosa_data):
    """Compare rsona output with librosa."""
    print("\n" + "=" * 70)
    print("RSONA vs LIBROSA COMPARISON")
    print("=" * 70)

    # Load rsona results
    with open(rsona_results_path) as f:
        rsona_data = json.load(f)

    rsona_chroma = np.array(rsona_data["all_runs"][0]["output"])
    print(f"\nRsona chroma shape: {rsona_chroma.shape}")
    print(f"Rsona chroma mean: {np.mean(rsona_chroma):.6f}")
    print(f"Rsona chroma std: {np.std(rsona_chroma):.6f}")
    print(f"Rsona chroma max: {np.max(rsona_chroma):.6f}")

    librosa_chroma = librosa_data["chroma"]
    print(f"\nLibrosa chroma shape: {librosa_chroma.shape}")
    print(f"Librosa chroma mean: {np.mean(librosa_chroma):.6f}")
    print(f"Librosa chroma std: {np.std(librosa_chroma):.6f}")
    print(f"Librosa chroma max: {np.max(librosa_chroma):.6f}")

    # Compare
    diff = rsona_chroma - librosa_chroma
    print(f"\nDifference statistics:")
    print(f"  MAE: {np.mean(np.abs(diff)):.6f}")
    print(f"  RMSE: {np.sqrt(np.mean(diff**2)):.6f}")
    print(f"  Mean diff: {np.mean(diff):.6f}")
    print(f"  Std diff: {np.std(diff):.6f}")
    print(f"  Max abs diff: {np.max(np.abs(diff)):.6f}")

    correlation = np.corrcoef(rsona_chroma.flatten(), librosa_chroma.flatten())[0, 1]
    print(f"  Correlation: {correlation:.6f}")

    # Check if rsona values are systematically higher
    percent_higher = np.mean(rsona_chroma > librosa_chroma) * 100
    print(f"\n  Rsona values higher in {percent_higher:.1f}% of cases")

    # Compare sample frames
    print(f"\nSample frame 100 comparison:")
    print(f"  Librosa: {librosa_chroma[:, 100]}")
    print(f"  Rsona:   {rsona_chroma[:, 100]}")
    print(f"  Diff:    {rsona_chroma[:, 100] - librosa_chroma[:, 100]}")
    print(
        f"  Rsona/Librosa ratio: {np.mean(rsona_chroma[:, 100] / (librosa_chroma[:, 100] + 1e-10)):.3f}"
    )

    # Check if both have proper frame normalization
    rsona_frame_maxes = np.max(rsona_chroma, axis=0)
    librosa_frame_maxes = np.max(librosa_chroma, axis=0)

    print(f"\nFrame-wise max normalization check:")
    print(f"  Rsona: all frames max=1.0: {np.allclose(rsona_frame_maxes, 1.0)}")
    print(f"  Librosa: all frames max=1.0: {np.allclose(librosa_frame_maxes, 1.0)}")
    print(
        f"  Rsona frame maxes: min={np.min(rsona_frame_maxes):.6f}, max={np.max(rsona_frame_maxes):.6f}"
    )

    # Analyze per-bin statistics
    print(f"\nPer-chroma-bin statistics:")
    print(f"  Bin | Librosa mean | Rsona mean | Ratio | MAE")
    print(f"  ----|--------------|------------|-------|------")
    for i in range(12):
        lib_mean = np.mean(librosa_chroma[i, :])
        rso_mean = np.mean(rsona_chroma[i, :])
        ratio = rso_mean / lib_mean if lib_mean > 1e-10 else 0
        mae = np.mean(np.abs(rsona_chroma[i, :] - librosa_chroma[i, :]))
        print(
            f"  {i:2d}  | {lib_mean:12.6f} | {rso_mean:10.6f} | {ratio:5.3f} | {mae:.4f}"
        )

    # Compare histograms
    print(f"\nValue distribution:")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    lib_hist, _ = np.histogram(librosa_chroma.flatten(), bins=bins)
    rso_hist, _ = np.histogram(rsona_chroma.flatten(), bins=bins)
    print(f"  Range    | Librosa % | Rsona %")
    print(f"  ---------|-----------|--------")
    for i in range(len(bins) - 1):
        lib_pct = lib_hist[i] / librosa_chroma.size * 100
        rso_pct = rso_hist[i] / rsona_chroma.size * 100
        print(f"  {bins[i]:.1f}-{bins[i + 1]:.1f} | {lib_pct:9.2f} | {rso_pct:7.2f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_chroma_detailed.py <audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]

    # Compute librosa chroma
    librosa_data = compute_librosa_chroma(audio_path)

    # Run rsona benchmark to get output
    print("\n" + "=" * 70)
    print("RUNNING RSONA BENCHMARK")
    print("=" * 70)

    rsona_output = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    rsona_output_path = rsona_output.name
    rsona_output.close()

    cmd = [
        "cargo",
        "run",
        "--release",
        "--bin",
        "rsona_runner",
        "--",
        "--audio",
        audio_path,
        "--runs",
        "1",
        "--warmup",
        "0",
        "--output",
        rsona_output_path,
        "--feature",
        "chroma_stft",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, cwd="bench/librosa_comparison", capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"Error running rsona: {result.stderr}")
        sys.exit(1)

    print(result.stdout)

    # Analyze comparison
    analyze_rsona_chroma(rsona_output_path, librosa_data)

    # Cleanup
    import os

    os.unlink(rsona_output_path)

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("If rsona values are systematically higher with same frame normalization,")
    print("the issue is likely in:")
    print("  1. Chroma filter bank construction (weights)")
    print("  2. Magnitude calculation from complex STFT")
    print("  3. Filter application (summation)")
    print(
        "\nBoth implementations should have max=1.0 per frame if normalized correctly."
    )


if __name__ == "__main__":
    main()
