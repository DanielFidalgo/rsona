#!/usr/bin/env python3
"""
Simple test to verify chroma normalization behavior.
Tests with a synthetic signal to understand the difference.
"""

import json
import subprocess
import sys
import tempfile
import wave

import librosa
import numpy as np


def create_test_audio(filename, duration=2.0, sr=22050):
    """Create a simple test audio file with a pure tone."""
    # A440 Hz sine wave
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    frequency = 440.0
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)

    # Save as WAV
    audio_int = (audio * 32767).astype(np.int16)
    with wave.open(filename, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sr)
        wav_file.writeframes(audio_int.tobytes())

    return audio, sr


def test_chroma_normalization():
    """Test chroma normalization with synthetic data."""

    # Create temporary audio file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_file = tmp.name

    print("Creating test audio file...")
    y, sr = create_test_audio(audio_file, duration=2.0, sr=22050)
    print(f"Audio: {len(y)} samples at {sr} Hz")

    # Test with librosa
    print("\n=== librosa chroma_stft (default norm=np.inf) ===")
    chroma_lib = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=512)
    print(f"Shape: {chroma_lib.shape}")
    print(f"Mean: {np.mean(chroma_lib):.6f}")
    print(f"Std: {np.std(chroma_lib):.6f}")
    print(f"Min: {np.min(chroma_lib):.6f}")
    print(f"Max: {np.max(chroma_lib):.6f}")

    # Check normalization
    print("\nPer-chroma-bin max values (should all be 1.0 with norm=np.inf):")
    for i in range(12):
        col_max = np.max(chroma_lib[i, :])
        print(f"  Chroma {i}: {col_max:.6f}")

    # Test with librosa without normalization
    print("\n=== librosa chroma_stft (norm=None) ===")
    chroma_lib_no_norm = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=2048, hop_length=512, norm=None
    )
    print(f"Mean: {np.mean(chroma_lib_no_norm):.6f}")
    print(f"Std: {np.std(chroma_lib_no_norm):.6f}")
    print(f"Max: {np.max(chroma_lib_no_norm):.6f}")

    # Manual column-wise max normalization
    print("\n=== Manual column-wise max normalization ===")
    chroma_manual = chroma_lib_no_norm.copy()
    for i in range(12):
        col_max = np.max(chroma_manual[i, :])
        if col_max > 1e-10:
            chroma_manual[i, :] /= col_max

    print(f"Mean: {np.mean(chroma_manual):.6f}")
    print(f"Std: {np.std(chroma_manual):.6f}")
    print(
        f"Matches librosa default: {np.allclose(chroma_manual, chroma_lib, rtol=1e-5)}"
    )
    print(f"Max absolute difference: {np.max(np.abs(chroma_manual - chroma_lib)):.10f}")

    # Test what happens if we normalize the WRONG way (per-frame instead of per-bin)
    print("\n=== WRONG: Per-frame max normalization ===")
    chroma_wrong = chroma_lib_no_norm.copy()
    for t in range(chroma_wrong.shape[1]):
        frame_max = np.max(chroma_wrong[:, t])
        if frame_max > 1e-10:
            chroma_wrong[:, t] /= frame_max

    print(f"Mean: {np.mean(chroma_wrong):.6f}")
    print(f"Std: {np.std(chroma_wrong):.6f}")
    correlation = np.corrcoef(chroma_wrong.flatten(), chroma_lib.flatten())[0, 1]
    print(f"Correlation with correct norm: {correlation:.6f}")
    mae = np.mean(np.abs(chroma_wrong - chroma_lib))
    print(f"MAE vs correct norm: {mae:.6f}")

    print("\n=== Summary ===")
    print("librosa's norm=np.inf (default):")
    print("  - Normalizes each of 12 chroma bins independently")
    print("  - Each chroma bin is normalized across ALL time frames")
    print("  - Max value in each chroma bin becomes 1.0")
    print(f"  - Result: mean={np.mean(chroma_lib):.4f}, max=1.0")
    print("\nIf you normalize per-frame instead (WRONG):")
    print("  - Each time frame is normalized independently")
    print("  - Max value in each frame becomes 1.0")
    print(
        f"  - Result: mean={np.mean(chroma_wrong):.4f}, correlation={correlation:.4f}"
    )

    # Clean up
    import os

    os.unlink(audio_file)


if __name__ == "__main__":
    test_chroma_normalization()
