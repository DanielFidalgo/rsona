#!/usr/bin/env python3
"""
Test to examine librosa's chroma filter normalization behavior.

This script investigates whether librosa normalizes the chroma filterbank
itself (not the output chroma features).
"""

import librosa
import librosa.filters
import numpy as np


def test_filter_normalization():
    """Test if librosa normalizes its chroma filters."""

    print("=" * 70)
    print("LIBROSA CHROMA FILTER BANK NORMALIZATION TEST")
    print("=" * 70)

    # Parameters matching our benchmark
    sr = 44100
    n_fft = 2048
    n_chroma = 12

    # Build chroma filter bank using librosa's function
    print(f"\nBuilding chroma filterbank with:")
    print(f"  sr={sr}, n_fft={n_fft}, n_chroma={n_chroma}")

    # librosa.filters.chroma is the function that builds the filter bank
    # It has a norm parameter that controls filter normalization

    # Test with default parameters
    print("\n1. Default chroma filters (norm=None):")
    filters_no_norm = librosa.filters.chroma(
        sr=sr, n_fft=n_fft, n_chroma=n_chroma, norm=None
    )
    print(f"   Shape: {filters_no_norm.shape} (n_chroma, n_bins)")
    print(f"   Sum of all weights: {np.sum(filters_no_norm):.2f}")
    print(f"   Max weight: {np.max(filters_no_norm):.6f}")
    print(f"   Mean weight: {np.mean(filters_no_norm):.6f}")

    # Check per-chroma normalization
    print("\n   Per-chroma statistics (no norm):")
    for i in range(n_chroma):
        row_sum = np.sum(filters_no_norm[i, :])
        row_max = np.max(filters_no_norm[i, :])
        row_l2 = np.linalg.norm(filters_no_norm[i, :])
        print(
            f"     Chroma {i:2d}: sum={row_sum:8.2f}, max={row_max:.6f}, L2={row_l2:8.2f}"
        )

    # Test with norm=1 (column normalization)
    print("\n2. Chroma filters with norm=1 (column-wise L1):")
    filters_norm_1 = librosa.filters.chroma(
        sr=sr, n_fft=n_fft, n_chroma=n_chroma, norm=1
    )
    print(f"   Shape: {filters_norm_1.shape}")
    print(f"   Sum of all weights: {np.sum(filters_norm_1):.2f}")
    print(f"   Max weight: {np.max(filters_norm_1):.6f}")

    # Check column normalization
    col_sums = np.sum(filters_norm_1, axis=0)
    print(f"   Column sums (should be 1.0 if column-normalized):")
    print(f"     Min: {np.min(col_sums):.6f}")
    print(f"     Max: {np.max(col_sums):.6f}")
    print(f"     All close to 1.0: {np.allclose(col_sums[col_sums > 0], 1.0)}")

    # Test with norm=2 (column-wise L2)
    print("\n3. Chroma filters with norm=2 (column-wise L2):")
    filters_norm_2 = librosa.filters.chroma(
        sr=sr, n_fft=n_fft, n_chroma=n_chroma, norm=2
    )
    col_l2_norms = np.linalg.norm(filters_norm_2, axis=0)
    print(f"   Column L2 norms (should be 1.0 if column-normalized):")
    print(f"     Min: {np.min(col_l2_norms):.6f}")
    print(f"     Max: {np.max(col_l2_norms):.6f}")
    print(f"     All close to 1.0: {np.allclose(col_l2_norms[col_l2_norms > 0], 1.0)}")

    # Test with norm=np.inf (max normalization)
    print("\n4. Chroma filters with norm=np.inf (column-wise max):")
    filters_norm_inf = librosa.filters.chroma(
        sr=sr, n_fft=n_fft, n_chroma=n_chroma, norm=np.inf
    )
    col_maxes = np.max(filters_norm_inf, axis=0)
    print(f"   Column maxes (should be 1.0 if column-normalized):")
    print(f"     Min: {np.min(col_maxes):.6f}")
    print(f"     Max: {np.max(col_maxes):.6f}")
    print(f"     All close to 1.0: {np.allclose(col_maxes[col_maxes > 0], 1.0)}")

    # Now test what chroma_stft actually uses
    print("\n" + "=" * 70)
    print("WHAT DOES chroma_stft USE?")
    print("=" * 70)

    # Create test audio
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440.0 * t)

    # Compute chroma with different filter norms
    print("\nComputing chroma_stft with different filter norms:")

    # Default (no filter norm specified in chroma_stft)
    chroma_default = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=n_fft, hop_length=512, norm=None
    )
    print(f"\n1. chroma_stft with norm=None (no output norm):")
    print(f"   Mean: {np.mean(chroma_default):.2f}")
    print(f"   Max: {np.max(chroma_default):.2f}")

    # With output normalization
    chroma_out_norm = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=n_fft, hop_length=512, norm=np.inf
    )
    print(f"\n2. chroma_stft with norm=np.inf (output normalized):")
    print(f"   Mean: {np.mean(chroma_out_norm):.6f}")
    print(f"   Max: {np.max(chroma_out_norm):.6f}")

    # Check what the ratio is
    ratio = np.mean(chroma_default) / np.mean(chroma_out_norm)
    print(f"\n   Ratio (no norm / norm): {ratio:.2f}x")

    # Compare using different filter normalizations
    print("\n" + "=" * 70)
    print("MANUAL CHROMA COMPUTATION WITH DIFFERENT FILTER NORMS")
    print("=" * 70)

    # Compute STFT
    S = librosa.stft(y, n_fft=n_fft, hop_length=512)
    S_mag = np.abs(S)

    # Apply different filter banks
    print("\nApplying different filter banks to STFT:")

    for norm_type, filters in [
        ("None", filters_no_norm),
        ("1 (L1)", filters_norm_1),
        ("2 (L2)", filters_norm_2),
        ("np.inf (max)", filters_norm_inf),
    ]:
        chroma_manual = filters @ S_mag
        print(f"\n  Filter norm={norm_type}:")
        print(f"    Chroma mean: {np.mean(chroma_manual):.6f}")
        print(f"    Chroma max: {np.max(chroma_manual):.6f}")

        # Normalize output per-frame (like librosa does with norm=np.inf)
        chroma_frame_norm = chroma_manual.copy()
        for t in range(chroma_frame_norm.shape[1]):
            col_max = np.max(chroma_frame_norm[:, t])
            if col_max > 1e-10:
                chroma_frame_norm[:, t] /= col_max

        print(f"    After frame norm - mean: {np.mean(chroma_frame_norm):.6f}")

        # Compare with librosa's output
        mae = np.mean(np.abs(chroma_frame_norm - chroma_out_norm))
        print(f"    MAE vs librosa default: {mae:.6f}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("\nlibrosa.feature.chroma_stft:")
    print("  - Uses filters WITHOUT normalization by default")
    print(
        "  - The 'norm' parameter controls OUTPUT normalization, not filter normalization"
    )
    print("  - Filter normalization is separate (librosa.filters.chroma)")
    print("\nTo match librosa exactly, rsona should:")
    print("  - Use unnormalized filters (or test which norm librosa uses)")
    print("  - Apply per-frame max normalization to the output")


if __name__ == "__main__":
    test_filter_normalization()
