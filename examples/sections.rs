use rsona::similarity::*;
use rsona::structure;
use rsona::{audio, feature, signal, spectrum};
use std::env;

fn main() {
    let path = env::args()
        .nth(1)
        .expect("usage: cargo run --example sections -- <audio>");

    let audio = audio::load(&path).expect("load failed");
    let frames = signal::frame(&audio, Default::default()).expect("frame failed");
    let spec = spectrum::stft(
        &frames,
        spectrum::StftConfig::with_frame_size(frames.frame_size()),
    )
    .expect("stft failed");

    let mel = spectrum::mel_spectrogram(&spec, Default::default());
    let mfcc = feature::mfcc(&mel, Default::default());

    let feats = FrameFeatures::new(mfcc.n_frames(), mfcc.n_mfcc(), &mfcc.as_slice());
    let ssm = self_similarity(
        &feats,
        SelfSimilarityConfig {
            max_lag: Some(800),
            ..Default::default()
        },
    );

    let result = structure::section_boundaries_from_novelty(&ssm, Default::default());

    println!("boundaries (frames): {:?}", result.boundary_frames);

    // Convert to seconds
    let sr = frames.sample_rate() as f64;
    let hop = frames.hop_size() as f64;
    let times: Vec<f64> = result
        .boundary_frames
        .iter()
        .map(|&f| (f as f64 * hop) / sr)
        .collect();
    println!("boundaries (sec): {:?}", times);
}
