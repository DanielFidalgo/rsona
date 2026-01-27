use serde::Serialize;
use std::time::Instant;

use rsona::{
    audio,
    feature::{MfccConfig, mfcc, onset_strength_from_mel, rms},
    signal::{FrameConfig, frame},
    spectrum::{DbConfig, MelConfig, StftConfig, mel_spectrogram, stft},
    temporal::estimate_tempo,
};

#[derive(Serialize)]
struct BenchmarkResult {
    audio_info: AudioInfo,
    parameters: Parameters,
    timing: Timing,
    results: Results,
}

#[derive(Serialize)]
struct AudioInfo {
    audio_file: String,
    duration_seconds: f64,
    sample_rate: u32,
    n_samples: usize,
}

#[derive(Serialize)]
struct Parameters {
    frame_size: usize,
    hop_size: usize,
    n_fft: usize,
    n_mels: usize,
    n_mfcc: usize,
}

#[derive(Serialize)]
struct Timing {
    load_time_ms: f64,
    frame_time_ms: f64,
    stft_time_ms: f64,
    mel_time_ms: f64,
    mfcc_time_ms: f64,
    rms_time_ms: f64,
    onset_time_ms: f64,
    tempo_time_ms: f64,
    total_time_ms: f64,
}

#[derive(Serialize)]
struct Results {
    tempo_bpm: f32,
    n_frames: usize,
    n_frequency_bins: usize,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <audio_file>", args[0]);
        std::process::exit(1);
    }

    let path = &args[1];

    // Start total timing
    let t_total = Instant::now();

    // Load audio
    let t0 = Instant::now();
    let buffer = audio::load(path).unwrap_or_else(|e| {
        eprintln!("Error loading audio: {}", e);
        std::process::exit(1);
    });
    let load_time = t0.elapsed();

    let duration_seconds = buffer.samples.len() as f64 / buffer.sample_rate as f64;
    let n_samples = buffer.samples.len();
    let sample_rate = buffer.sample_rate;

    // Frame audio
    let frame_cfg = FrameConfig::default();
    let frame_size = frame_cfg.frame_size;
    let t1 = Instant::now();
    let frames = frame(&buffer, frame_cfg).unwrap_or_else(|e| {
        eprintln!("Error framing audio: {}", e);
        std::process::exit(1);
    });
    let frame_time = t1.elapsed();

    let hop_size = frames.hop_size();

    // STFT
    let stft_cfg = StftConfig::default();
    let n_fft = stft_cfg.n_fft;
    let t2 = Instant::now();
    let spec = stft(&frames, stft_cfg).unwrap_or_else(|e| {
        eprintln!("Error computing STFT: {}", e);
        std::process::exit(1);
    });
    let stft_time = t2.elapsed();

    let n_frequency_bins = spec.n_bins();
    let n_frames = spec.n_frames();

    // Mel spectrogram
    let mel_cfg = MelConfig::default();
    let t3 = Instant::now();
    let mel = mel_spectrogram(&spec, mel_cfg);
    let mel_time = t3.elapsed();

    let n_mels = mel.n_mels();

    // MFCC
    let mfcc_cfg = MfccConfig::default();
    let t4 = Instant::now();
    let mfcc_result = mfcc(&mel, mfcc_cfg);
    let mfcc_time = t4.elapsed();

    let n_mfcc = mfcc_result.n_mfcc();

    // RMS
    let t5 = Instant::now();
    let _rms_result = rms(&frames);
    let rms_time = t5.elapsed();

    // Onset strength
    let t6 = Instant::now();
    let onset = onset_strength_from_mel(&mel, DbConfig::default());
    let onset_time = t6.elapsed();

    // Tempo estimation
    let t7 = Instant::now();
    let tempo = estimate_tempo(
        onset.values(),
        frames.sample_rate(),
        hop_size,
        Default::default(),
    );
    let tempo_time = t7.elapsed();

    // Total time
    let total_time = t_total.elapsed();

    // Build result structure
    let result = BenchmarkResult {
        audio_info: AudioInfo {
            audio_file: path.clone(),
            duration_seconds,
            sample_rate,
            n_samples,
        },
        parameters: Parameters {
            frame_size,
            hop_size,
            n_fft,
            n_mels,
            n_mfcc,
        },
        timing: Timing {
            load_time_ms: load_time.as_secs_f64() * 1000.0,
            frame_time_ms: frame_time.as_secs_f64() * 1000.0,
            stft_time_ms: stft_time.as_secs_f64() * 1000.0,
            mel_time_ms: mel_time.as_secs_f64() * 1000.0,
            mfcc_time_ms: mfcc_time.as_secs_f64() * 1000.0,
            rms_time_ms: rms_time.as_secs_f64() * 1000.0,
            onset_time_ms: onset_time.as_secs_f64() * 1000.0,
            tempo_time_ms: tempo_time.as_secs_f64() * 1000.0,
            total_time_ms: total_time.as_secs_f64() * 1000.0,
        },
        results: Results {
            tempo_bpm: tempo.bpm,
            n_frames,
            n_frequency_bins,
        },
    };

    // Output JSON
    let json = serde_json::to_string_pretty(&result).unwrap_or_else(|e| {
        eprintln!("Error serializing result: {}", e);
        std::process::exit(1);
    });

    println!("{}", json);
}
