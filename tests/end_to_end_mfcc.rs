use std::path::PathBuf;

use rsona::audio;
use rsona::feature;
use rsona::signal;
use rsona::spectrum;

use hound;

/// Write a temporary mono WAV file containing a sine wave.
/// Returns the path to the file.
fn write_test_wav() -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push("rsona_test_sine.wav");

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(&path, spec).expect("failed to create wav writer");

    let freq = 440.0;
    let duration_sec = 0.5;
    let n_samples = (spec.sample_rate as f32 * duration_sec) as usize;

    for i in 0..n_samples {
        let t = i as f32 / spec.sample_rate as f32;
        let sample = (2.0 * std::f32::consts::PI * freq * t).sin();
        let scaled = (sample * i16::MAX as f32) as i16;
        writer.write_sample(scaled).unwrap();
    }

    writer.finalize().unwrap();
    path
}

#[test]
fn wav_to_mfcc_pipeline_produces_expected_shape() {
    let wav_path = write_test_wav();

    // 1) Load audio
    let audio = audio::load(&wav_path).expect("audio load failed");
    assert_eq!(audio.sample_rate, 16_000);
    assert_eq!(audio.channels, 1);
    assert!(!audio.samples.is_empty());

    // 2) Frame signal
    let frame_cfg = signal::FrameConfig {
        frame_size: 512,
        hop_size: 256,
        window: signal::Window::Hann,
        padding: signal::Padding::ZeroPadEnd,
        channel_mode: signal::ChannelMode::Channel(0),
        center: true,
    };

    let frames = signal::frame(&audio, frame_cfg).expect("framing failed");
    assert!(frames.n_frames() > 0);
    assert_eq!(frames.frame_size(), 512);

    // 3) STFT
    let stft_cfg = spectrum::StftConfig { n_fft: 512 };

    let spec = spectrum::stft(&frames, stft_cfg).expect("stft failed");
    assert_eq!(spec.n_frames(), frames.n_frames());
    assert_eq!(spec.n_bins(), 512 / 2 + 1);

    // 4) Mel spectrogram
    let mel = spectrum::mel_spectrogram(
        &spec,
        spectrum::MelConfig {
            n_mels: 40,
            ..Default::default()
        },
    );

    assert_eq!(mel.n_frames(), spec.n_frames());
    assert_eq!(mel.n_mels(), 40);

    // 5) MFCC
    let mfcc = feature::mfcc(
        &mel,
        feature::MfccConfig {
            n_mfcc: 13,
            ..Default::default()
        },
    );

    assert_eq!(mfcc.n_frames(), mel.n_frames());
    assert_eq!(mfcc.n_mfcc(), 13);

    // Clean up temp file
    let _ = std::fs::remove_file(&wav_path);
}
