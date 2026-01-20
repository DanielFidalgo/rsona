#!/usr/bin/env python3
"""
Test to understand librosa's norm parameter in chroma_stft.

The documentation says:
- norm : None or np.inf
  - None: No normalization
  - np.inf: Column-wise max normalization (each chroma bin normalized to max=1)

But let's verify what actually happens.
"""

import librosa
import numpy as np


def main():
    # Create a simple sine wave at A440
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440.0 * t)

    print("=" * 70)
    print("Testing librosa.feature.chroma_stft normalization")
    print("=" * 70)

    # Test with default norm (which is np.inf)
    print("\n1. Default (norm=np.inf - should normalize columns)")
    chroma_default = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=512)
    print(f"   Shape: {chroma_default.shape} (n_chroma, n_frames)")
    print(f"   Mean: {np.mean(chroma_default):.6f}")
    print(f"   Std: {np.std(chroma_default):.6f}")
    print(f"   Global max: {np.max(chroma_default):.6f}")

    # Check max per chroma bin (row)
    print("   Max per chroma bin (should be 1.0 if column-normalized):")
    for i in range(12):
        row_max = np.max(chroma_default[i, :])
        print(f"      Chroma {i:2d}: {row_max:.6f}")

    # Check max per time frame (column)
    print("   Max per time frame (should be 1.0 if row-normalized):")
    frame_maxes = np.max(chroma_default, axis=0)
    print(f"      Min of frame maxes: {np.min(frame_maxes):.6f}")
    print(f"      Max of frame maxes: {np.max(frame_maxes):.6f}")
    print(f"      Mean of frame maxes: {np.mean(frame_maxes):.6f}")
    num_ones = np.sum(np.isclose(frame_maxes, 1.0))
    print(f"      Frames with max=1.0: {num_ones}/{len(frame_maxes)}")

    # Test with norm=None
    print("\n2. No normalization (norm=None)")
    chroma_none = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=2048, hop_length=512, norm=None
    )
    print(f"   Mean: {np.mean(chroma_none):.6f}")
    print(f"   Std: {np.std(chroma_none):.6f}")
    print(f"   Max: {np.max(chroma_none):.6f}")

    # Test with explicit norm=np.inf
    print("\n3. Explicit norm=np.inf (should be same as default)")
    chroma_inf = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=2048, hop_length=512, norm=np.inf
    )
    print(f"   Same as default: {np.allclose(chroma_inf, chroma_default)}")

    # Test with norm=1 (L1 norm)
    print("\n4. L1 norm (norm=1)")
    chroma_l1 = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=2048, hop_length=512, norm=1
    )
    print(f"   Mean: {np.mean(chroma_l1):.6f}")
    print(f"   Std: {np.std(chroma_l1):.6f}")
    # Check if columns sum to 1
    col_sums = np.sum(chroma_l1, axis=0)
    print(f"   Column sums (should be 1.0 if column-normalized):")
    print(f"      Min: {np.min(col_sums):.6f}")
    print(f"      Max: {np.max(col_sums):.6f}")
    # Check if rows sum to 1
    row_sums = np.sum(chroma_l1, axis=1)
    print(f"   Row sums (should be 1.0 if row-normalized):")
    print(f"      Min: {np.min(row_sums):.6f}")
    print(f"      Max: {np.max(row_sums):.6f}")

    # Test with norm=2 (L2 norm)
    print("\n5. L2 norm (norm=2)")
    chroma_l2 = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=2048, hop_length=512, norm=2
    )
    print(f"   Mean: {np.mean(chroma_l2):.6f}")
    print(f"   Std: {np.std(chroma_l2):.6f}")

    # Manual simulation of what we THINK column-wise max norm should do
    print("\n6. Manual column-wise (per-chroma-bin) max normalization")
    chroma_manual_col = chroma_none.copy()
    for i in range(12):  # For each chroma bin (row)
        row_max = np.max(chroma_manual_col[i, :])
        if row_max > 1e-10:
            chroma_manual_col[i, :] /= row_max
    print(f"   Matches default: {np.allclose(chroma_manual_col, chroma_default)}")
    if not np.allclose(chroma_manual_col, chroma_default):
        print(f"   Max diff: {np.max(np.abs(chroma_manual_col - chroma_default)):.6f}")

    # Manual simulation of row-wise (per-frame) max normalization
    print("\n7. Manual row-wise (per-frame) max normalization")
    chroma_manual_row = chroma_none.copy()
    for t in range(chroma_manual_row.shape[1]):  # For each time frame (column)
        col = chroma_manual_row[:, t]
        col_max = np.max(col)
        if col_max > 1e-10:
            chroma_manual_row[:, t] /= col_max
    print(f"   Matches default: {np.allclose(chroma_manual_row, chroma_default)}")
    if np.allclose(chroma_manual_row, chroma_default):
        print("   ✓ LIBROSA IS DOING PER-FRAME (ROW-WISE) NORMALIZATION!")

    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("=" * 70)
    if np.allclose(chroma_manual_row, chroma_default):
        print("librosa's norm=np.inf does PER-FRAME max normalization")
        print("  - Each time frame (column in the matrix) is normalized")
        print("  - The max value in each time frame becomes 1.0")
        print("  - This is NOT per-chroma-bin normalization")
    elif np.allclose(chroma_manual_col, chroma_default):
        print("librosa's norm=np.inf does PER-CHROMA-BIN max normalization")
        print("  - Each chroma bin (row in the matrix) is normalized")
        print("  - The max value in each chroma bin becomes 1.0")
    else:
        print("librosa's normalization is neither simple per-frame nor per-bin")
        print("Need to investigate further...")


if __name__ == "__main__":
    main()
