#!/usr/bin/env python3
"""
Compare Actual Rsona Chroma Filters with Librosa Reference

This script loads the actual rsona filter weights (exported from Rust)
and compares them with librosa's reference implementation.

Usage:
    # First, export rsona filters:
    cargo run --example export_chroma_filters > rsona_filters.json

    # Then run comparison:
    python3 bench/compare_actual_filters.py rsona_filters.json
"""

import json
import sys

import librosa
import librosa.filters
import matplotlib.pyplot as plt
import numpy as np


def load_rsona_filters(json_path):
    """Load rsona filters from JSON export."""
    with open(json_path, "r") as f:
        data = json.load(f)

    filters = np.array(data["filters"])
    params = data["parameters"]
    stats = data["statistics"]

    print("Loaded rsona filters:")
    print(f"  Shape: {filters.shape}")
    print(f"  Parameters: {params}")
    print(f"  Peak statistics: {stats}")

    return filters, params


def get_librosa_filters(sr, n_fft, n_chroma):
    """Get librosa chroma filters with matching parameters."""
    filters = librosa.filters.chroma(sr=sr, n_fft=n_fft, n_chroma=n_chroma)
    return filters


def analyze_filter_differences(rsona_filters, librosa_filters):
    """Analyze differences between rsona and librosa filters."""
    print("\n" + "=" * 70)
    print("FILTER COMPARISON ANALYSIS")
    print("=" * 70)

    # Shape check
    print(f"\nShape:")
    print(f"  Rsona:   {rsona_filters.shape}")
    print(f"  Librosa: {librosa_filters.shape}")

    if rsona_filters.shape != librosa_filters.shape:
        print("  ❌ ERROR: Shape mismatch!")
        return

    # Peak analysis
    print("\nPer-filter peak values:")
    print("  Chroma | Rsona Peak | Librosa Peak | Ratio  | Diff")
    print("  -------|------------|--------------|--------|-------")

    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    rsona_peaks = []
    librosa_peaks = []
    ratios = []

    for i in range(rsona_filters.shape[0]):
        rsona_peak = np.max(rsona_filters[i, :])
        librosa_peak = np.max(librosa_filters[i, :])
        ratio = rsona_peak / librosa_peak if librosa_peak > 1e-10 else 0
        diff = rsona_peak - librosa_peak

        rsona_peaks.append(rsona_peak)
        librosa_peaks.append(librosa_peak)
        ratios.append(ratio)

        print(
            f"  {i:2d} ({note_names[i]:2s}) | {rsona_peak:10.6f} | {librosa_peak:12.6f} | {ratio:6.3f} | {diff:7.5f}"
        )

    rsona_peaks = np.array(rsona_peaks)
    librosa_peaks = np.array(librosa_peaks)
    ratios = np.array(ratios)

    # Overall statistics
    print("\nOverall peak statistics:")
    print(
        f"  Rsona:   mean={np.mean(rsona_peaks):.6f}, std={np.std(rsona_peaks):.6f}, range=[{np.min(rsona_peaks):.6f}, {np.max(rsona_peaks):.6f}]"
    )
    print(
        f"  Librosa: mean={np.mean(librosa_peaks):.6f}, std={np.std(librosa_peaks):.6f}, range=[{np.min(librosa_peaks):.6f}, {np.max(librosa_peaks):.6f}]"
    )
    print(
        f"  Ratio:   mean={np.mean(ratios):.3f}, std={np.std(ratios):.3f}, range=[{np.min(ratios):.3f}, {np.max(ratios):.3f}]"
    )

    # Element-wise comparison
    diff = rsona_filters - librosa_filters
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    max_diff = np.max(np.abs(diff))

    print("\nElement-wise comparison:")
    print(f"  MAE:      {mae:.6f}")
    print(f"  RMSE:     {rmse:.6f}")
    print(f"  Max diff: {max_diff:.6f}")

    # Correlation
    correlation = np.corrcoef(rsona_filters.flatten(), librosa_filters.flatten())[0, 1]
    print(f"  Correlation: {correlation:.6f}")

    # Assessment
    print("\n" + "=" * 70)
    print("ASSESSMENT")
    print("=" * 70)

    if np.allclose(rsona_peaks, librosa_peaks, atol=0.05):
        print("✅ PASS: Peak values match (within 0.05)")
    else:
        mean_ratio = np.mean(ratios)
        if mean_ratio < 0.5:
            print(f"❌ FAIL: Rsona peaks are {1 / mean_ratio:.1f}x too LOW")
        elif mean_ratio > 1.5:
            print(f"❌ FAIL: Rsona peaks are {mean_ratio:.1f}x too HIGH")
        else:
            print(f"⚠️  WARNING: Rsona peaks differ by {abs(1 - mean_ratio) * 100:.1f}%")

    if mae < 0.01:
        print(f"✅ PASS: MAE is low ({mae:.6f})")
    else:
        print(f"⚠️  WARNING: MAE is high ({mae:.6f})")

    if correlation > 0.98:
        print(f"✅ PASS: Correlation is excellent ({correlation:.6f})")
    elif correlation > 0.90:
        print(f"⚠️  WARNING: Correlation is acceptable ({correlation:.6f})")
    else:
        print(f"❌ FAIL: Correlation is poor ({correlation:.6f})")

    return rsona_peaks, librosa_peaks, ratios


def create_comparison_plots(
    rsona_filters, librosa_filters, output_dir="bench/artifacts"
):
    """Create comprehensive comparison plots."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Plot 1: Selected filters comparison (C, E, G, A)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    key_notes = [(0, "C"), (4, "E"), (7, "G"), (9, "A")]

    for col, (note_idx, note_name) in enumerate(key_notes):
        # Linear scale
        ax_lin = axes[0, col]
        x = np.arange(rsona_filters.shape[1])

        ax_lin.plot(
            x,
            librosa_filters[note_idx, :],
            "b-",
            linewidth=2,
            label="Librosa (Reference)",
            alpha=0.7,
        )
        ax_lin.plot(
            x, rsona_filters[note_idx, :], "r--", linewidth=2, label="Rsona", alpha=0.7
        )

        ax_lin.set_title(f"{note_name} - Linear Scale", fontsize=12, fontweight="bold")
        ax_lin.set_ylabel("Filter Weight", fontsize=10)
        ax_lin.grid(True, alpha=0.3)
        ax_lin.legend(fontsize=9)
        ax_lin.set_ylim([0, 1.05])

        # Add peak annotations
        rsona_peak = np.max(rsona_filters[note_idx, :])
        librosa_peak = np.max(librosa_filters[note_idx, :])
        ratio = rsona_peak / librosa_peak

        ax_lin.text(
            0.02,
            0.95,
            f"Before peak: {rsona_peak:.3f}\nAfter peak: {librosa_peak:.3f}\nRatio: {ratio:.2f}x",
            transform=ax_lin.transAxes,
            verticalalignment="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
        )

        # Log scale
        ax_log = axes[1, col]

        librosa_log = librosa_filters[note_idx, :].copy()
        librosa_log[librosa_log < 1e-10] = 1e-10
        rsona_log = rsona_filters[note_idx, :].copy()
        rsona_log[rsona_log < 1e-10] = 1e-10

        ax_log.semilogy(
            x, librosa_log, "b-", linewidth=2, label="Librosa (Reference)", alpha=0.7
        )
        ax_log.semilogy(x, rsona_log, "r--", linewidth=2, label="Rsona", alpha=0.7)

        ax_log.set_title(f"{note_name} - Log Scale", fontsize=12, fontweight="bold")
        ax_log.set_xlabel("FFT Bin", fontsize=10)
        ax_log.set_ylabel("Filter Weight (log)", fontsize=10)
        ax_log.grid(True, alpha=0.3)
        ax_log.legend(fontsize=9)
        ax_log.set_ylim([1e-10, 2])

    fig.suptitle(
        "Chroma Filter for G (Linear Scale) - Fix Comparison",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    output_path = f"{output_dir}/actual_filter_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    plt.close()

    # Plot 2: Peak comparison bar chart
    fig, ax = plt.subplots(figsize=(14, 8))

    rsona_peaks = [np.max(rsona_filters[i, :]) for i in range(12)]
    librosa_peaks = [np.max(librosa_filters[i, :]) for i in range(12)]

    x = np.arange(12)
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        rsona_peaks,
        width,
        label="Rsona",
        color="red",
        alpha=0.7,
        edgecolor="darkred",
        linewidth=2,
    )
    bars2 = ax.bar(
        x + width / 2,
        librosa_peaks,
        width,
        label="Librosa (Reference)",
        color="blue",
        alpha=0.7,
        edgecolor="darkblue",
        linewidth=2,
    )

    ax.axhline(
        y=1.0,
        color="green",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label="Target (1.0)",
    )

    ax.set_xlabel("Chroma Bin (Note)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Filter Peak Value", fontsize=12, fontweight="bold")
    ax.set_title(
        "Chroma Filter Peak Values: Rsona vs Librosa", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(note_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 1.15])

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="darkred",
        )

    for bar in bars2:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="darkblue",
        )

    plt.tight_layout()
    output_path = f"{output_dir}/actual_peak_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 compare_actual_filters.py <rsona_filters.json>")
        print("\nFirst export filters with:")
        print("  cargo run --example export_chroma_filters > rsona_filters.json")
        sys.exit(1)

    json_path = sys.argv[1]

    print("=" * 70)
    print("RSONA vs LIBROSA CHROMA FILTER COMPARISON")
    print("=" * 70)

    # Load rsona filters
    print(f"\nLoading rsona filters from: {json_path}")
    rsona_filters, params = load_rsona_filters(json_path)

    # Get librosa filters with matching parameters
    print(f"\nGenerating librosa filters with matching parameters...")
    librosa_filters = get_librosa_filters(
        params["sample_rate"], params["n_fft"], params["n_chroma"]
    )
    print(f"  Shape: {librosa_filters.shape}")

    # Analyze differences
    analyze_filter_differences(rsona_filters, librosa_filters)

    # Create plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    create_comparison_plots(rsona_filters, librosa_filters)

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
