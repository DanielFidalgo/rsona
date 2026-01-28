use std::env;
use tracing::info;

use rsona::feature;
use rsona::pipeline;
use rsona::spectrum;
use rsona::utils::logging;

fn main() {
    // Initialize logging
    logging::init_verbose();

    let path = env::args()
        .nth(1)
        .expect("usage: cargo run --example mfcc -- <audio file>");

    // Load audio with speech analysis preset (512 frame size, 256 hop)
    // This is optimized for MFCC computation with lower latency
    let pipeline = pipeline::speech(&path).expect("failed to load audio");

    info!("Audio loaded:");
    info!("  sample rate: {} Hz", pipeline.buffer.sample_rate);
    info!("  frames: {}", pipeline.frames.n_frames());
    info!("  frequency bins: {}", pipeline.spec.n_bins());
    info!("");

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

    info!("MFCC computed:");
    info!("  frames: {}", mfcc.n_frames());
    info!("  coeffs: {}", mfcc.n_mfcc());
    info!("");

    // Show first frame's coefficients
    if mfcc.n_frames() > 0 {
        info!("First frame coefficients:");
        let n_mfcc = mfcc.n_mfcc();
        let first_frame = &mfcc.as_slice()[..n_mfcc];
        for (i, &coeff) in first_frame.iter().enumerate() {
            info!("  c{}: {:.4}", i, coeff);
        }
    }
}
