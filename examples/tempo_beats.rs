use rsona::{audio, feature, signal, spectrum, temporal};
use std::env;

fn main() {
    let path = env::args()
        .nth(1)
        .expect("usage: cargo run --example tempo_beats -- <audio file>");

    let audio = audio::load(&path).expect("load failed");
    let frames = signal::frame(&audio, Default::default()).expect("frame failed");
    let spec = spectrum::stft(
        &frames,
        spectrum::StftConfig::with_frame_size(frames.frame_size()),
    )
    .expect("stft failed");

    let onset = feature::onset_strength(&spec, Default::default());

    let tempo = temporal::estimate_tempo(
        onset.values(),
        frames.sample_rate(),
        frames.hop_size(),
        Default::default(),
    );

    let beats = temporal::track_beats(
        onset.values(),
        frames.sample_rate(),
        frames.hop_size(),
        tempo.period_frames,
        Default::default(),
    );

    println!("tempo ≈ {:.1} BPM", tempo.bpm);
    println!("first 10 beat times (s):");
    for t in beats.beat_times.iter().take(10) {
        println!("{:.3}", t);
    }
}
