#!/usr/bin/env python3
"""
Simple comparison script for chroma outputs between rsona and librosa.
"""

import json
import sys

import librosa
import numpy as np


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_chroma.py <audio_file> <rsona_output.json>")
        sys.exit(1)

    audio_path = sys.argv[1]
    rsona_json_path = sys.argv[2]

    print("=" * 70)
    print("CHROMA COMPARISON: rsona vs librosa")
    print("=" * 70)

    # Load audio and compute librosa chroma
    print(f"\nLoading audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    print(f"  Duration: {len(y) / sr:.2f}s, Sample rate: {sr} Hz")

    # Compute librosa chroma
    print("\nComputing librosa chroma...")
    chroma_lib = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=2048, hop_length=512, norm=np.inf
    )
    print(f"  Shape: {chroma_lib.shape}")
    print(f"  Mean: {np.mean(chroma_lib):.6f}")
    print(f"  Std: {np.std(chroma_lib):.6f}")
    print(f"  Min: {np.min(chroma_lib):.6f}")
    print(f"  Max: {np.max(chroma_lib):.6f}")

    # Load rsona output
    print(f"\nLoading rsona output: {rsona_json_path}")
    with open(rsona_json_path) as f:
        rsona_data = json.load(f)

    chroma_rsona = np.array(rsona_data["all_runs"][0]["output"])
    print(f"  Shape: {chroma_rsona.shape}")
    print(f"  Mean: {np.mean(chroma_rsona):.6f}")
    print(f"  Std: {np.std(chroma_rsona):.6f}")
    print(f"  Min: {np.min(chroma_rsona):.6f}")
    print(f"  Max: {np.max(chroma_rsona):.6f}")

    # Check shapes match
    if chroma_lib.shape != chroma_rsona.shape:
        print(f"\n❌ ERROR: Shape mismatch!")
        print(f"  librosa: {chroma_lib.shape}")
        print(f"  rsona: {chroma_rsona.shape}")
        sys.exit(1)

    # Compute differences
    print("\n" + "=" * 70)
    print("DIFFERENCE ANALYSIS")
    print("=" * 70)

    diff = chroma_rsona - chroma_lib
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    correlation = np.corrcoef(chroma_rsona.flatten(), chroma_lib.flatten())[0, 1]

    print(f"\nOverall metrics:")
    print(f"  MAE: {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  Correlation: {correlation:.6f}")
    print(f"  Mean difference: {np.mean(diff):.6f}")
    print(f"  Std difference: {np.std(diff):.6f}")

    # Check normalization
    print(f"\n" + "=" * 70)
    print("NORMALIZATION CHECK")
    print("=" * 70)

    lib_frame_maxes = np.max(chroma_lib, axis=0)
    rsona_frame_maxes = np.max(chroma_rsona, axis=0)

    print(f"\nPer-frame max values (should be 1.0 for norm=inf):")
    print(
        f"  librosa: min={np.min(lib_frame_maxes):.6f}, max={np.max(lib_frame_maxes):.6f}"
    )
    print(
        f"  rsona:   min={np.min(rsona_frame_maxes):.6f}, max={np.max(rsona_frame_maxes):.6f}"
    )

    lib_frames_at_1 = np.sum(np.isclose(lib_frame_maxes, 1.0, atol=1e-5))
    rsona_frames_at_1 = np.sum(np.isclose(rsona_frame_maxes, 1.0, atol=1e-5))

    print(f"\nFrames with max ≈ 1.0:")
    print(
        f"  librosa: {lib_frames_at_1}/{len(lib_frame_maxes)} ({lib_frames_at_1 / len(lib_frame_maxes) * 100:.1f}%)"
    )
    print(
        f"  rsona:   {rsona_frames_at_1}/{len(rsona_frame_maxes)} ({rsona_frames_at_1 / len(rsona_frame_maxes) * 100:.1f}%)"
    )

    # Sample frame comparison
    print(f"\n" + "=" * 70)
    print("SAMPLE FRAME COMPARISON")
    print("=" * 70)

    frame_idx = min(100, chroma_lib.shape[1] - 1)
    print(f"\nFrame {frame_idx}:")
    print(f"  Chroma | Librosa  | Rsona    | Diff     | Ratio")
    print(f"  -------|----------|----------|----------|-------")
    for i in range(12):
        lib_val = chroma_lib[i, frame_idx]
        rso_val = chroma_rsona[i, frame_idx]
        diff_val = rso_val - lib_val
        ratio = rso_val / lib_val if lib_val > 1e-10 else 0
        print(
            f"  {i:2d}     | {lib_val:8.6f} | {rso_val:8.6f} | {diff_val:8.6f} | {ratio:6.3f}"
        )

    print(f"\n  Frame max:")
    print(f"    librosa: {np.max(chroma_lib[:, frame_idx]):.6f}")
    print(f"    rsona:   {np.max(chroma_rsona[:, frame_idx]):.6f}")

    # Per-bin statistics
    print(f"\n" + "=" * 70)
    print("PER-CHROMA-BIN STATISTICS")
    print("=" * 70)

    print(f"\n  Bin | Lib Mean | Rso Mean | Ratio | MAE")
    print(f"  ----|----------|----------|-------|-------")
    for i in range(12):
        lib_mean = np.mean(chroma_lib[i, :])
        rso_mean = np.mean(chroma_rsona[i, :])
        ratio = rso_mean / lib_mean if lib_mean > 1e-10 else 0
        bin_mae = np.mean(np.abs(chroma_rsona[i, :] - chroma_lib[i, :]))
        print(
            f"  {i:2d}  | {lib_mean:8.6f} | {rso_mean:8.6f} | {ratio:5.3f} | {bin_mae:7.5f}"
        )

    # Summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    percent_higher = np.mean(chroma_rsona > chroma_lib) * 100
    print(f"\nRsona values are higher in {percent_higher:.1f}% of cases")

    if mae < 0.1:
        print(f"✅ PASS: MAE ({mae:.6f}) is below threshold (0.1)")
    else:
        print(f"❌ FAIL: MAE ({mae:.6f}) exceeds threshold (0.1)")

    if correlation >= 0.85:
        print(f"✅ PASS: Correlation ({correlation:.6f}) meets threshold (0.85)")
    else:
        print(f"❌ FAIL: Correlation ({correlation:.6f}) below threshold (0.85)")

    print(f"\nPossible causes for differences:")
    print(f"  - STFT implementation differences")
    print(f"  - Chroma filter bank weights")
    print(f"  - Floating point precision")
    print(f"  - Normalization edge cases (silent/near-silent frames)")


if __name__ == "__main__":
    main()
