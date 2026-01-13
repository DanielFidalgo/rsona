//! Example: intro / loop / outro segmentation using rsona.
//!
//! Usage:
//!   cargo run --example segment_loop path/to/audio.wav

use std::env;
use std::error::Error;

use rsona::{
    audio,
    feature::{MfccConfig, mfcc, onset_strength},
    signal::{FrameConfig, frame},
    similarity::{FrameFeatures, SelfSimilarityConfig, self_similarity},
    spectrum::{MelConfig, StftConfig, mel_spectrogram, stft},
    structure::{SegmentationConfig, segment_intro_loop_outro_with_onsets},
};

fn main() -> Result<(), Box<dyn Error>> {
    let path = env::args().nth(1).expect("usage: segment_loop <audio.wav>");

    // --- load audio ---
    let buffer = audio::load(&path)?;

    println!(
        "Loaded audio: {} samples @ {} Hz ({} channels)",
        buffer.samples.len(),
        buffer.sample_rate,
        buffer.channels
    );

    // --- frame the signal ---
    let frame_cfg = FrameConfig {
        frame_size: 2048,
        hop_size: 512,
        ..Default::default()
    };

    let frames = frame(&buffer, frame_cfg)?;

    // --- STFT ---
    let spec = stft(&frames, StftConfig::default())?;

    // --- Mel spectrogram ---
    let mel = mel_spectrogram(&spec, MelConfig::default());

    // --- MFCC ---
    let mfcc = mfcc(&mel, MfccConfig::default());

    // --- Self-similarity over MFCCs ---
    let feats = FrameFeatures::new(mfcc.n_frames(), mfcc.n_mfcc(), mfcc.as_slice());

    let ssm = self_similarity(
        &feats,
        SelfSimilarityConfig {
            max_lag: Some(1500),
            ..Default::default()
        },
    );

    // --- Onset strength ---
    let onset = onset_strength(&spec, Default::default());

    // --- Segment intro / loop / outro ---
    let seg = segment_intro_loop_outro_with_onsets(
        &frames,
        &ssm,
        onset.values(),
        SegmentationConfig::default(),
    )
    .expect("segmentation failed");

    println!();
    println!("=== Segmentation ===");
    println!(
        "Intro: {:.2}s → {:.2}s",
        seg.intro.start_seconds, seg.intro.end_seconds
    );
    println!(
        "Loop : {:.2}s → {:.2}s (len {:.2}s)",
        seg.loop_seg.start_seconds,
        seg.loop_seg.end_seconds,
        seg.loop_seg.end_seconds - seg.loop_seg.start_seconds
    );
    println!(
        "Outro: {:.2}s → {:.2}s",
        seg.outro.start_seconds, seg.outro.end_seconds
    );

    println!();
    println!("Loop confidence: {:.2}", seg.loop_confidence);
    if let Some(bpm) = seg.tempo_bpm {
        println!("Estimated tempo: {:.1} BPM", bpm);
    }

    Ok(())
}
