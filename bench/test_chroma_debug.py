#!/usr/bin/env python3
"""
Debug script to understand chroma normalization behavior.
"""

import sys

import librosa
import numpy as np


def test_chroma_normalization():
    """Test different normalization approaches for chroma."""

    # Load audio
    audio_path = "bench/audio/Alex-Productions - Future Bass Technology _ Shades.wav"
    print(f"Loading audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    print(f"Sample rate: {sr}, Duration: {len(y) / sr:.2f}s")

    # Compute chroma with default settings (norm=np.inf means column-wise max normalization)
    print("\n=== librosa default (norm=np.inf, which is column-wise max norm) ===")
    chroma_default = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=512)
    print(f"Shape: {chroma_default.shape}")
    print(f"Mean: {np.mean(chroma_default):.6f}")
    print(f"Std: {np.std(chroma_default):.6f}")
    print(f"Min: {np.min(chroma_default):.6f}")
    print(f"Max: {np.max(chroma_default):.6f}")

    # Check if each column (chroma bin) has max of 1.0
    print("\nColumn (chroma bin) max values:")
    for i in range(12):
        col_max = np.max(chroma_default[i, :])
        print(f"  Chroma bin {i}: max = {col_max:.6f}")

    # Compute chroma without normalization
    print("\n=== librosa with norm=None (no normalization) ===")
    chroma_no_norm = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=2048, hop_length=512, norm=None
    )
    print(f"Shape: {chroma_no_norm.shape}")
    print(f"Mean: {np.mean(chroma_no_norm):.6f}")
    print(f"Std: {np.std(chroma_no_norm):.6f}")
    print(f"Min: {np.min(chroma_no_norm):.6f}")
    print(f"Max: {np.max(chroma_no_norm):.6f}")

    # Manually apply column-wise max normalization to verify
    print("\n=== Manual column-wise max normalization ===")
    chroma_manual = chroma_no_norm.copy()
    for i in range(chroma_manual.shape[0]):  # For each chroma bin (row)
        col_max = np.max(chroma_manual[i, :])
        if col_max > 1e-10:
            chroma_manual[i, :] /= col_max

    print(
        f"Manual normalization matches default: {np.allclose(chroma_manual, chroma_default)}"
    )
    print(f"Max diff: {np.max(np.abs(chroma_manual - chroma_default)):.10f}")

    # Show what per-frame (row-wise) max normalization would look like
    print("\n=== Per-frame (row-wise) max normalization (WRONG for librosa) ===")
    chroma_frame_norm = chroma_no_norm.copy()
    for t in range(
        chroma_frame_norm.shape[1]
    ):  # For each time frame (column in the matrix)
        frame = chroma_frame_norm[:, t]
        frame_max = np.max(frame)
        if frame_max > 1e-10:
            chroma_frame_norm[:, t] /= frame_max

    print(f"Mean: {np.mean(chroma_frame_norm):.6f}")
    print(f"Std: {np.std(chroma_frame_norm):.6f}")
    print(f"Min: {np.min(chroma_frame_norm):.6f}")
    print(f"Max: {np.max(chroma_frame_norm):.6f}")
    print(
        f"Correlation with default: {np.corrcoef(chroma_frame_norm.flatten(), chroma_default.flatten())[0, 1]:.6f}"
    )

    print("\n=== Summary ===")
    print("librosa's default norm=np.inf means:")
    print("  - Normalize each CHROMA BIN (row) across all time frames")
    print("  - Each chroma bin's maximum value becomes 1.0")
    print("  - This is COLUMN-WISE normalization in the data structure")
    print("  - Shape is (n_chroma=12, n_frames)")
    print("\nrsona should:")
    print("  - Store data as (n_frames, n_chroma) internally")
    print("  - Apply normalization to each chroma bin across all frames")
    print("  - Transpose to (n_chroma, n_frames) when outputting to match librosa")


if __name__ == "__main__":
    test_chroma_normalization()
