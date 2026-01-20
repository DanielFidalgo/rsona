use std::env;

use rsona::feature;
use rsona::pipeline;
use rsona::spectrum;

fn main() {
    let path = env::args()
        .nth(1)
        .expect("usage: cargo run --example mfcc -- <audio file>");

    // Load audio with speech analysis preset (512 frame size, 256 hop)
    // This is optimized for MFCC computation with lower latency
    let pipeline = pipeline::speech(&path).expect("failed to load audio");

    println!("Audio loaded:");
    println!("  sample rate: {} Hz", pipeline.buffer.sample_rate);
    println!("  frames: {}", pipeline.frames.n_frames());
    println!("  frequency bins: {}", pipeline.spec.n_bins());
    println!();

    // Compute mel spectrogram
    let mel = spectrum::mel_spectrogram(
        &pipeline.spec,
        spectrum::MelConfig {
            n_mels: 40,
            ..Default::default()
        },
    );

    // Compute MFCC
    let mfcc = feature::mfcc(
        &mel,
        feature::MfccConfig {
            n_mfcc: 13,
            ..Default::default()
        },
    );

    println!("MFCC computed:");
    println!("  frames: {}", mfcc.n_frames());
    println!("  coeffs: {}", mfcc.n_mfcc());
    println!();

    // Show first frame's coefficients
    if mfcc.n_frames() > 0 {
        println!("First frame coefficients:");
        let n_mfcc = mfcc.n_mfcc();
        let first_frame = &mfcc.as_slice()[..n_mfcc];
        for (i, &coeff) in first_frame.iter().enumerate() {
            println!("  c{}: {:.4}", i, coeff);
        }
    }
}
