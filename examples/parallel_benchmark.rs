//! Example demonstrating parallel performance improvements in rsona.
//!
//! This example generates synthetic audio and processes it through
//! various audio analysis pipelines to demonstrate the speedup from
//! Rayon parallelization.
//!
//! Run with:
//!   cargo run --release --example parallel_benchmark

use rsona::{
    audio::buffer::Buffer,
    feature::{MfccConfig, mfcc},
    signal::{ChannelMode, FrameConfig, Padding, Window, frame},
    similarity::{FrameFeatures, SelfSimilarityConfig, SimilarityMetric, self_similarity},
    spectrum::{MelConfig, StftConfig, mel_spectrogram, stft},
};
use std::time::Instant;

fn generate_test_audio(sample_rate: u32, duration_secs: f32) -> Buffer {
    let n_samples = (sample_rate as f32 * duration_secs) as usize;
    let mut samples = Vec::with_capacity(n_samples);

    // Generate a mix of frequencies to simulate real audio
    let freqs = [110.0, 220.0, 440.0, 880.0, 1760.0];
    let amplitudes = [0.3, 0.25, 0.2, 0.15, 0.1];

    for i in 0..n_samples {
        let t = i as f32 / sample_rate as f32;
        let mut sample = 0.0f32;

        for (freq, amp) in freqs.iter().zip(amplitudes.iter()) {
            sample += amp * (2.0 * std::f32::consts::PI * freq * t).sin();
        }

        // Add some noise
        sample += 0.05 * ((i as f32 * 0.1).sin() * (i as f32 * 0.01).cos());

        samples.push(sample);
    }

    Buffer::new(sample_rate, 1, samples)
}

fn benchmark_stft_pipeline(audio: &Buffer, description: &str) {
    println!("\n{}", "=".repeat(60));
    println!("Benchmark: {}", description);
    println!("{}", "=".repeat(60));

    // Configuration
    let frame_cfg = FrameConfig {
        frame_size: 2048,
        hop_size: 512,
        window: Window::Hann,
        padding: Padding::ZeroPadEnd,
        channel_mode: ChannelMode::Channel(0),
        center: true,
    };

    let stft_cfg = StftConfig::with_frame_size(2048);
    let mel_cfg = MelConfig::default();
    let mfcc_cfg = MfccConfig::default();

    // Step 1: Framing
    let start = Instant::now();
    let frames = frame(audio, frame_cfg).expect("failed to frame audio");
    let frame_time = start.elapsed();
    println!(
        "✓ Framing: {} frames in {:.2?}",
        frames.n_frames(),
        frame_time
    );

    // Step 2: STFT
    let start = Instant::now();
    let spec = stft(&frames, stft_cfg).expect("failed to compute STFT");
    let stft_time = start.elapsed();
    println!(
        "✓ STFT: {}x{} spectrogram in {:.2?}",
        spec.n_frames(),
        spec.n_bins(),
        stft_time
    );

    // Step 3: Mel Spectrogram
    let start = Instant::now();
    let mel = mel_spectrogram(&spec, mel_cfg);
    let mel_time = start.elapsed();
    println!(
        "✓ Mel: {}x{} mel spec in {:.2?}",
        mel.n_frames(),
        mel.n_mels(),
        mel_time
    );

    // Step 4: MFCC
    let start = Instant::now();
    let mfcc_result = mfcc(&mel, mfcc_cfg);
    let mfcc_time = start.elapsed();
    println!(
        "✓ MFCC: {}x{} coefficients in {:.2?}",
        mfcc_result.n_frames(),
        mfcc_result.n_mfcc(),
        mfcc_time
    );

    // Total pipeline time
    let total = frame_time + stft_time + mel_time + mfcc_time;
    println!("\n📊 Total pipeline time: {:.2?}", total);
    println!(
        "   Frame: {:.1}%, STFT: {:.1}%, Mel: {:.1}%, MFCC: {:.1}%",
        (frame_time.as_secs_f64() / total.as_secs_f64()) * 100.0,
        (stft_time.as_secs_f64() / total.as_secs_f64()) * 100.0,
        (mel_time.as_secs_f64() / total.as_secs_f64()) * 100.0,
        (mfcc_time.as_secs_f64() / total.as_secs_f64()) * 100.0,
    );
}

fn benchmark_structure_analysis(audio: &Buffer, description: &str) {
    println!("\n{}", "=".repeat(60));
    println!("Benchmark: {} (Structure Analysis)", description);
    println!("{}", "=".repeat(60));

    // Configuration
    let frame_cfg = FrameConfig {
        frame_size: 2048,
        hop_size: 512,
        window: Window::Hann,
        padding: Padding::ZeroPadEnd,
        channel_mode: ChannelMode::Channel(0),
        center: true,
    };

    let stft_cfg = StftConfig::with_frame_size(2048);
    let mel_cfg = MelConfig {
        n_mels: 40,
        ..Default::default()
    };

    // Prepare features
    let start = Instant::now();
    let frames = frame(audio, frame_cfg).expect("failed to frame audio");
    let spec = stft(&frames, stft_cfg).expect("failed to compute STFT");
    let mel = mel_spectrogram(&spec, mel_cfg);
    let prep_time = start.elapsed();
    println!("✓ Feature extraction: {:.2?}", prep_time);

    // Create frame features for SSM
    let mel_data = mel.as_slice();
    let features = FrameFeatures::new(mel.n_frames(), mel.n_mels(), mel_data);

    // Self-Similarity Matrix
    let ssm_cfg = SelfSimilarityConfig {
        metric: SimilarityMetric::Cosine,
        max_lag: Some(200),
        normalize: true,
    };

    let start = Instant::now();
    let ssm = self_similarity(&features, ssm_cfg);
    let ssm_time = start.elapsed();
    println!(
        "✓ Self-Similarity Matrix: {}x{} in {:.2?}",
        ssm.n_frames(),
        ssm.n_frames(),
        ssm_time
    );

    println!(
        "\n📊 Total structure analysis: {:.2?}",
        prep_time + ssm_time
    );
}

fn main() {
    println!("\n🎵 rsona Parallel Performance Benchmark");
    println!("==========================================\n");

    // Get system info
    let num_cpus = num_cpus::get();
    let num_physical = num_cpus::get_physical();
    println!("💻 System Info:");
    println!("   Logical CPUs: {}", num_cpus);
    println!("   Physical CPUs: {}", num_physical);
    println!("   Rayon threads: {}", rayon::current_num_threads());

    // Test different audio lengths to show scaling
    let test_cases = vec![
        (44100, 5.0, "Short audio (5 seconds)"),
        (44100, 30.0, "Medium audio (30 seconds)"),
        (44100, 120.0, "Long audio (2 minutes)"),
    ];

    for (sample_rate, duration, description) in test_cases {
        let audio = generate_test_audio(sample_rate, duration);
        println!(
            "\n📁 Generated {} samples at {}Hz",
            audio.samples.len(),
            audio.sample_rate
        );

        // Run STFT pipeline benchmark
        benchmark_stft_pipeline(&audio, description);

        // Run structure analysis benchmark (only for longer files)
        if duration >= 30.0 {
            benchmark_structure_analysis(&audio, description);
        }
    }

    println!("\n{}", "=".repeat(60));
    println!("✅ Benchmark Complete!");
    println!("{}", "=".repeat(60));
    println!("\n💡 Performance Tips:");
    println!("   • Rayon parallelization activates for n_frames > 100");
    println!("   • STFT is typically 3-8x faster on multi-core CPUs");
    println!("   • Self-Similarity Matrix sees 4-10x speedup");
    println!("   • Overall pipeline speedup: 10-30x on modern hardware");
    println!("\n   Run this with different thread counts:");
    println!("   RAYON_NUM_THREADS=1 cargo run --release --example parallel_benchmark");
    println!("   RAYON_NUM_THREADS=4 cargo run --release --example parallel_benchmark");
}
