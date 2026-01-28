//! MFCC Benchmark - Measure FFT-based DCT performance improvement
//!
//! Usage: cargo run --release --bin mfcc_benchmark <audio_file> [num_iterations]

use rsona::feature::{MfccConfig, mfcc};
use rsona::pipeline;
use rsona::spectrum::{MelConfig, mel_spectrogram};
use std::env;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <audio_file> [num_iterations]", args[0]);
        std::process::exit(1);
    }

    let audio_path = &args[1];
    let num_iterations = if args.len() >= 3 {
        args[2].parse::<usize>().unwrap_or(100)
    } else {
        100
    };

    println!("MFCC FFT-based DCT Benchmark");
    println!("============================");
    println!("Audio file: {}", audio_path);
    println!("Iterations: {}", num_iterations);
    println!();

    // Load audio and compute pipeline with standard settings (2048 frame size, 512 hop)
    println!("Loading audio and computing STFT...");
    let pipeline = pipeline::standard(audio_path)?;
    println!("Audio loaded successfully");
    println!();

    // Setup feature configurations
    let mel_cfg = MelConfig {
        n_mels: 128,
        ..Default::default()
    };

    let mfcc_cfg = MfccConfig {
        n_mfcc: 20,
        ..Default::default()
    };

    // Pre-compute mel spectrogram
    println!("Computing Mel spectrogram...");
    let mel = mel_spectrogram(&pipeline.spec, mel_cfg);
    println!(
        "Mel spectrogram shape: {} frames x {} mels",
        mel.n_frames(),
        mel.n_mels()
    );
    println!();

    // Warmup
    println!("Warming up (10 iterations)...");
    for _ in 0..10 {
        let _ = mfcc(&mel, mfcc_cfg.clone());
    }
    println!();

    // Benchmark MFCC only
    println!(
        "Benchmarking MFCC computation ({} iterations)...",
        num_iterations
    );
    let mut timings = Vec::with_capacity(num_iterations);

    for i in 0..num_iterations {
        let start = Instant::now();
        let result = mfcc(&mel, mfcc_cfg.clone());
        let elapsed = start.elapsed();

        timings.push(elapsed.as_secs_f64() * 1000.0); // Convert to milliseconds

        // Verify result shape on first iteration
        if i == 0 {
            println!(
                "MFCC shape: {} frames x {} coefficients",
                result.n_frames(),
                result.n_mfcc()
            );
        }
    }

    // Calculate statistics
    timings.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean = timings.iter().sum::<f64>() / timings.len() as f64;
    let median = timings[timings.len() / 2];
    let min = timings[0];
    let max = timings[timings.len() - 1];

    let variance = timings.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / timings.len() as f64;
    let std_dev = variance.sqrt();

    let p95 = timings[(timings.len() as f64 * 0.95) as usize];
    let p99 = timings[(timings.len() as f64 * 0.99) as usize];

    // Print results
    println!();
    println!("Results:");
    println!("--------");
    println!("Mean:     {:.4} ms", mean);
    println!("Median:   {:.4} ms", median);
    println!("Std Dev:  {:.4} ms", std_dev);
    println!("Min:      {:.4} ms", min);
    println!("Max:      {:.4} ms", max);
    println!("P95:      {:.4} ms", p95);
    println!("P99:      {:.4} ms", p99);
    println!();

    // Calculate throughput
    let frames_per_second = mel.n_frames() as f64 / (mean / 1000.0);

    println!("Throughput:");
    println!("-----------");
    println!("Frames/sec:        {:.0}", frames_per_second);
    println!();

    // Compare with baseline if available
    // Reference baseline for this audio (6.36ms average)
    let reference_baseline = 6.36;
    let speedup = reference_baseline / mean;

    println!("Comparison to baseline:");
    println!("----------------------");
    println!("Reference baseline: {:.2} ms", reference_baseline);
    println!("rsona (current):    {:.4} ms", mean);
    println!("Speedup:            {:.2}x", speedup);
    println!();

    // Target after optimization
    let target_speedup = 2.5; // Expected 2.5-3x improvement
    let target_time = reference_baseline / target_speedup;
    let improvement_needed = (mean / target_time - 1.0) * 100.0;

    if mean <= target_time {
        println!("✅ Target achieved! Current performance meets or exceeds 2.5x speedup goal.");
    } else {
        println!(
            "🎯 Target: {:.2} ms ({:.1}x speedup)",
            target_time, target_speedup
        );
        println!(
            "   Need {:.1}% improvement to reach target",
            improvement_needed
        );
    }

    Ok(())
}
