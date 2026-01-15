use std::env;

use rsona::audio;
use rsona::feature;
use rsona::signal;
use rsona::spectrum;

fn main() {
    let path = env::args()
        .nth(1)
        .expect("usage: cargo run --example mfcc -- <audio file>");

    // 1) Load audio
    let audio = audio::load(&path).expect("failed to load audio");

    // 2) Frame signal
    let frames = signal::frame(
        &audio,
        signal::FrameConfig {
            frame_size: 512,
            hop_size: 256,
            window: signal::Window::Hann,
            padding: signal::Padding::ZeroPadEnd,
            channel_mode: signal::ChannelMode::Channel(0),
            center: true,
        },
    )
    .expect("framing failed");

    // 3) STFT
    let spec = spectrum::stft(&frames, spectrum::StftConfig { n_fft: 512 }).expect("stft failed");

    // 4) Mel spectrogram
    let mel = spectrum::mel_spectrogram(
        &spec,
        spectrum::MelConfig {
            n_mels: 40,
            ..Default::default()
        },
    );

    // 5) MFCC
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
}
