use std::env;

use rsona::audio;
use rsona::signal;
use rsona::spectrum;

fn main() {
    let path = env::args()
        .nth(1)
        .expect("usage: cargo run --example mel_db -- <audio file>");

    let audio = audio::load(&path).expect("failed to load audio");

    let frames = signal::frame(
        &audio,
        signal::FrameConfig {
            frame_size: 1024,
            hop_size: 512,
            window: signal::Window::Hann,
            padding: signal::Padding::ZeroPadEnd,
            channel_mode: signal::ChannelMode::Channel(0),
        },
    )
    .expect("framing failed");

    let spec = spectrum::stft(&frames, spectrum::StftConfig { n_fft: 1024 }).expect("stft failed");

    let mel = spectrum::mel_spectrogram(&spec, Default::default());

    let mel_db = spectrum::power_to_db(mel.as_slice(), spectrum::DbConfig::default());

    println!("Mel spectrogram (dB):");
    println!("  frames: {}", mel.n_frames());
    println!("  mels:   {}", mel.n_mels());
    println!("  values: {}", mel_db.len());
}
