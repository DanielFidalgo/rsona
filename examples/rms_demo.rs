//! RMS (Root Mean Square) energy extraction example.
//!
//! This example demonstrates how to compute RMS energy from audio,
//! both directly from time-domain frames and from spectrograms.
//!
//! Run with:
//! ```bash
//! cargo run --example rms_demo
//! ```

use rsona::{
    audio::Buffer,
    feature::{rms, rms_from_spectrogram},
    signal::{FrameConfig, Window, frame},
    spectrum::{StftConfig, stft},
};
use std::f32::consts::PI;

fn generate_sine_wave(sample_rate: u32, duration: f32, frequency: f32) -> Buffer {
    let n_samples = (sample_rate as f32 * duration) as usize;
    let mut samples = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let t = i as f32 / sample_rate as f32;
        let sample = (2.0 * PI * frequency * t).sin();
        samples.push(sample);
    }

    Buffer::new(sample_rate, 1, samples)
}

fn example_rms_from_audio() {
    println!("{}", "=".repeat(60));
    println!("RMS from Time-Domain Audio");
    println!("{}", "=".repeat(60));

    // Generate a 440 Hz sine wave (A4 note)
    let audio = generate_sine_wave(22050, 2.0, 440.0);

    println!(
        "Audio: {} samples, {} Hz",
        audio.samples.len(),
        audio.sample_rate
    );

    // Configure framing
    let config = FrameConfig {
        frame_size: 2048,
        hop_size: 512,
        window: Window::Hann,
        center: true,
        ..Default::default()
    };

    // Frame the audio
    let frames = frame(&audio, config).expect("Failed to frame audio");
    println!(
        "Frames: {} frames, size {}",
        frames.n_frames(),
        frames.frame_size()
    );

    // Compute RMS
    let rms_result = rms(&frames);

    println!("RMS: {} values", rms_result.n_frames());
    println!(
        "RMS range: [{:.6}, {:.6}]",
        rms_result
            .values()
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min),
        rms_result
            .values()
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
    );
    let mean_rms: f32 = rms_result.values().iter().sum::<f32>() / rms_result.n_frames() as f32;
    println!("RMS mean: {:.6}", mean_rms);
    println!(
        "First 5 RMS values: {:?}",
        &rms_result.values()[..5.min(rms_result.n_frames())]
    );

    // For a sine wave with amplitude 1.0, RMS should be approximately 1/sqrt(2) ≈ 0.707
    println!("Expected RMS for sine wave: ~0.707 (1/sqrt(2))");
    println!();
}

fn example_rms_from_spectrogram() {
    println!("{}", "=".repeat(60));
    println!("RMS from Spectrogram");
    println!("{}", "=".repeat(60));

    // Generate a 440 Hz sine wave
    let audio = generate_sine_wave(22050, 2.0, 440.0);

    // Configure framing
    let frame_config = FrameConfig {
        frame_size: 2048,
        hop_size: 512,
        window: Window::Hann,
        center: true,
        ..Default::default()
    };

    // Frame the audio
    let frames = frame(&audio, frame_config).expect("Failed to frame audio");

    // Compute STFT
    let stft_config = StftConfig { n_fft: 2048 };
    let spectrogram = stft(&frames, stft_config).expect("Failed to compute STFT");

    println!(
        "Spectrogram: {} frames × {} bins",
        spectrogram.n_frames(),
        spectrogram.n_bins()
    );

    // Compute RMS from spectrogram
    let rms_result = rms_from_spectrogram(&spectrogram);

    println!("RMS: {} values", rms_result.n_frames());
    println!(
        "RMS range: [{:.6}, {:.6}]",
        rms_result
            .values()
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min),
        rms_result
            .values()
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
    );
    let mean_rms: f32 = rms_result.values().iter().sum::<f32>() / rms_result.n_frames() as f32;
    println!("RMS mean: {:.6}", mean_rms);
    println!(
        "First 5 RMS values: {:?}",
        &rms_result.values()[..5.min(rms_result.n_frames())]
    );
    println!();
}

fn example_compare_methods() {
    println!("{}", "=".repeat(60));
    println!("Comparison: Time-Domain vs Spectrogram RMS");
    println!("{}", "=".repeat(60));

    // Generate a 440 Hz sine wave
    let audio = generate_sine_wave(16000, 1.0, 440.0);

    // Use rectangular window and no centering for exact comparison
    let frame_config = FrameConfig {
        frame_size: 512,
        hop_size: 256,
        window: Window::Rectangular,
        center: false,
        ..Default::default()
    };

    // Frame the audio
    let frames = frame(&audio, frame_config).expect("Failed to frame audio");

    // Compute RMS from time domain
    let rms_time = rms(&frames);

    // Compute STFT and RMS from spectrogram
    let stft_config = StftConfig { n_fft: 512 };
    let spectrogram = stft(&frames, stft_config).expect("Failed to compute STFT");
    let rms_spec = rms_from_spectrogram(&spectrogram);

    println!("Time-domain RMS: {} values", rms_time.n_frames());
    println!("Spectrogram RMS: {} values", rms_spec.n_frames());

    // Calculate statistics
    let mean_time: f32 = rms_time.values().iter().sum::<f32>() / rms_time.n_frames() as f32;
    let mean_spec: f32 = rms_spec.values().iter().sum::<f32>() / rms_spec.n_frames() as f32;

    println!("Time-domain RMS mean: {:.6}", mean_time);
    println!("Spectrogram RMS mean: {:.6}", mean_spec);
    println!("Difference: {:.6}", (mean_time - mean_spec).abs());

    // Calculate correlation
    let n = rms_time.n_frames().min(rms_spec.n_frames());
    let mut diff_sum = 0.0f32;
    let mut max_diff = 0.0f32;

    for i in 0..n {
        let diff = (rms_time.values()[i] - rms_spec.values()[i]).abs();
        diff_sum += diff;
        max_diff = max_diff.max(diff);
    }

    println!("Average absolute difference: {:.6}", diff_sum / n as f32);
    println!("Maximum absolute difference: {:.6}", max_diff);
    println!(
        "Relative difference: {:.2}%",
        100.0 * diff_sum / n as f32 / mean_time
    );
    println!();
    println!("Note: With rectangular window and no centering, both methods");
    println!("      should produce nearly identical results (Parseval's theorem).");
    println!();
}

fn example_constant_signal() {
    println!("{}", "=".repeat(60));
    println!("RMS of Constant Signal");
    println!("{}", "=".repeat(60));

    // Create a constant signal with amplitude 0.5
    let samples = vec![0.5f32; 8192];
    let audio = Buffer::new(16000, 1, samples);

    let config = FrameConfig {
        frame_size: 512,
        hop_size: 256,
        window: Window::Rectangular,
        center: false,
        ..Default::default()
    };

    let frames = frame(&audio, config).expect("Failed to frame audio");
    let rms_result = rms(&frames);

    println!("Signal amplitude: 0.5");
    println!("Expected RMS: 0.5");

    let mean_rms: f32 = rms_result.values().iter().sum::<f32>() / rms_result.n_frames() as f32;
    println!("Computed RMS mean: {:.6}", mean_rms);
    println!(
        "RMS range: [{:.6}, {:.6}]",
        rms_result
            .values()
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min),
        rms_result
            .values()
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
    );

    // Check that all complete frames have RMS ≈ 0.5
    let n_complete = ((8192 - 512) / 256) + 1;
    let complete_frames_correct = rms_result.values()[..n_complete]
        .iter()
        .all(|&v| (v - 0.5).abs() < 1e-6);

    println!(
        "All complete frames have RMS ≈ 0.5: {}",
        complete_frames_correct
    );
    println!();
}

fn example_energy_analysis() {
    println!("{}", "=".repeat(60));
    println!("Energy Analysis with RMS");
    println!("{}", "=".repeat(60));

    // Create a signal with amplitude modulation
    let sample_rate = 16000;
    let duration = 2.0;
    let n_samples = (sample_rate as f32 * duration) as usize;
    let mut samples = Vec::with_capacity(n_samples);

    // Generate sine wave with 2 Hz amplitude modulation
    for i in 0..n_samples {
        let t = i as f32 / sample_rate as f32;
        let carrier = (2.0 * PI * 440.0 * t).sin();
        let envelope = 0.5 + 0.5 * (2.0 * PI * 2.0 * t).sin();
        samples.push(carrier * envelope);
    }

    let audio = Buffer::new(sample_rate, 1, samples);

    let config = FrameConfig {
        frame_size: 2048,
        hop_size: 512,
        ..Default::default()
    };

    let frames = frame(&audio, config).expect("Failed to frame audio");
    let rms_result = rms(&frames);

    println!("Signal: {} samples, modulated at 2 Hz", audio.samples.len());
    println!("RMS: {} values", rms_result.n_frames());

    // Find peaks and valleys in RMS
    let values = rms_result.values();
    let max_rms = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min_rms = values.iter().copied().fold(f32::INFINITY, f32::min);
    let mean_rms: f32 = values.iter().sum::<f32>() / values.len() as f32;

    println!("RMS statistics:");
    println!("  Min: {:.6}", min_rms);
    println!("  Max: {:.6}", max_rms);
    println!("  Mean: {:.6}", mean_rms);
    println!(
        "  Dynamic range: {:.2} dB",
        20.0 * (max_rms / min_rms).log10()
    );
    println!();
    println!("RMS can be used for:");
    println!("  - Energy/loudness tracking");
    println!("  - Voice activity detection");
    println!("  - Dynamic range compression");
    println!("  - Audio segmentation");
    println!();
}

fn main() {
    println!();
    println!("{}", "=".repeat(60));
    println!("rsona RMS Feature Extraction Examples");
    println!("{}", "=".repeat(60));
    println!();

    example_rms_from_audio();
    example_rms_from_spectrogram();
    example_compare_methods();
    example_constant_signal();
    example_energy_analysis();

    println!("{}", "=".repeat(60));
    println!("All examples completed successfully!");
    println!("{}", "=".repeat(60));
    println!();
    println!("Key Points:");
    println!("- RMS measures the energy/loudness of audio over time");
    println!("- Can be computed from time-domain frames or frequency-domain spectrogram");
    println!("- With rectangular window and no centering, both methods match (Parseval's theorem)");
    println!("- For a constant signal, RMS equals the absolute value of the signal");
    println!("- For a sine wave, RMS ≈ amplitude / sqrt(2) ≈ 0.707 × amplitude");
    println!();
}
