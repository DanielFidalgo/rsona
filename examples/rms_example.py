#!/usr/bin/env python3
"""
Example demonstrating librosa's RMS feature extraction.

This script shows how to compute RMS (Root Mean Square) energy
using librosa, both from time-domain audio and from spectrograms.
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def generate_test_signal(sr=22050, duration=2.0, freq=440.0):
    """Generate a simple sine wave for testing."""
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)
    return signal, sr


def rms_from_audio_example():
    """Compute RMS directly from audio samples."""
    print("=" * 60)
    print("RMS from Audio Samples")
    print("=" * 60)

    # Generate test signal
    y, sr = generate_test_signal()

    # Compute RMS with default parameters
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)

    print(f"Signal shape: {y.shape}")
    print(f"Sample rate: {sr} Hz")
    print(f"RMS shape: {rms.shape}")
    print(f"Number of frames: {rms.shape[1]}")
    print(f"RMS range: [{rms.min():.6f}, {rms.max():.6f}]")
    print(f"RMS mean: {rms.mean():.6f}")
    print(f"First 5 RMS values: {rms[0, :5]}")
    print()

    return y, sr, rms


def rms_from_spectrogram_example():
    """Compute RMS from STFT spectrogram."""
    print("=" * 60)
    print("RMS from Spectrogram")
    print("=" * 60)

    # Generate test signal
    y, sr = generate_test_signal()

    # Compute STFT
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))

    # Compute RMS from spectrogram
    rms = librosa.feature.rms(S=S, frame_length=2048)

    print(f"Spectrogram shape: {S.shape}")
    print(f"RMS shape: {rms.shape}")
    print(f"Number of frames: {rms.shape[1]}")
    print(f"RMS range: [{rms.min():.6f}, {rms.max():.6f}]")
    print(f"RMS mean: {rms.mean():.6f}")
    print(f"First 5 RMS values: {rms[0, :5]}")
    print()

    return S, rms


def compare_audio_vs_spectrogram():
    """Compare RMS computed from audio vs spectrogram."""
    print("=" * 60)
    print("Comparison: Audio vs Spectrogram")
    print("=" * 60)

    # Generate test signal
    y, sr = generate_test_signal()

    # RMS from audio (with rectangular window for better comparison)
    rms_audio = librosa.feature.rms(
        y=y, frame_length=2048, hop_length=512, center=False
    )

    # RMS from spectrogram (rectangular window)
    S = np.abs(
        librosa.stft(y, n_fft=2048, hop_length=512, window=np.ones(2048), center=False)
    )
    rms_spec = librosa.feature.rms(S=S, frame_length=2048)

    print(f"RMS from audio shape: {rms_audio.shape}")
    print(f"RMS from spectrogram shape: {rms_spec.shape}")
    print(f"Audio RMS mean: {rms_audio.mean():.6f}")
    print(f"Spectrogram RMS mean: {rms_spec.mean():.6f}")
    print(f"Difference: {np.abs(rms_audio - rms_spec).mean():.6f}")
    print(f"Max difference: {np.abs(rms_audio - rms_spec).max():.6f}")
    print()

    # They should match very closely with rectangular window
    if rms_audio.shape == rms_spec.shape:
        correlation = np.corrcoef(rms_audio[0], rms_spec[0])[0, 1]
        print(f"Correlation: {correlation:.6f}")

    return rms_audio, rms_spec


def constant_signal_test():
    """Test RMS on a constant signal."""
    print("=" * 60)
    print("RMS of Constant Signal")
    print("=" * 60)

    # Constant signal of amplitude 0.5
    y = np.ones(4096) * 0.5
    sr = 16000

    # Compute RMS with rectangular window and no centering
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=256, center=False)

    print(f"Signal amplitude: 0.5")
    print(f"Expected RMS: 0.5")
    print(f"Computed RMS mean: {rms.mean():.6f}")
    print(f"RMS range: [{rms.min():.6f}, {rms.max():.6f}]")
    print(f"All complete frames have RMS ≈ 0.5: {np.allclose(rms[0, :-1], 0.5)}")
    print()


def visualize_rms():
    """Create visualization of RMS energy over time."""
    print("=" * 60)
    print("Visualizing RMS Energy")
    print("=" * 60)

    # Load example audio or generate test signal
    y, sr = generate_test_signal(duration=3.0, freq=440.0)

    # Add amplitude modulation to make it more interesting
    t = np.linspace(0, 3.0, len(y))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)  # 2 Hz modulation
    y = y * envelope

    # Compute RMS
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)

    # Compute spectrogram for visualization
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    # Create figure
    fig, ax = plt.subplots(nrows=3, figsize=(12, 8), sharex=True)

    # Plot waveform
    times = librosa.times_like(y, sr=sr)
    ax[0].plot(times, y, alpha=0.7)
    ax[0].set_ylabel("Amplitude")
    ax[0].set_title("Audio Waveform")
    ax[0].grid(True, alpha=0.3)

    # Plot RMS energy
    times_rms = librosa.times_like(rms, sr=sr, hop_length=512)
    ax[1].semilogy(times_rms, rms[0], linewidth=2, label="RMS Energy")
    ax[1].set_ylabel("RMS (log scale)")
    ax[1].set_title("RMS Energy Over Time")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    # Plot spectrogram
    img = librosa.display.specshow(
        S_db, sr=sr, hop_length=512, x_axis="time", y_axis="log", ax=ax[2]
    )
    ax[2].set_title("Log-frequency Spectrogram")
    fig.colorbar(img, ax=ax[2], format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig("rms_visualization.png", dpi=150, bbox_inches="tight")
    print("Saved visualization to: rms_visualization.png")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("=" * 60)
    print("librosa RMS Feature Extraction Examples")
    print("=" * 60)
    print()

    # Run examples
    rms_from_audio_example()
    rms_from_spectrogram_example()
    compare_audio_vs_spectrogram()
    constant_signal_test()

    try:
        visualize_rms()
    except Exception as e:
        print(f"Could not create visualization: {e}")

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print()
    print("Key Points:")
    print("- RMS measures the energy/loudness of audio over time")
    print("- Can be computed from time-domain samples or frequency-domain spectrogram")
    print(
        "- With rectangular window and no centering, both methods give identical results"
    )
    print("- For a constant signal, RMS equals the absolute value of the signal")
    print("- For a sine wave, RMS ≈ amplitude / sqrt(2) ≈ 0.707 * amplitude")
    print()


if __name__ == "__main__":
    main()
