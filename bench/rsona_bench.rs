use serde::Serialize;
use std::time::Instant;

use rsona::{
    audio,
    feature::{MfccConfig, mfcc, onset_strength_from_mel},
    signal::{FrameConfig, frame},
    spectrum::{DbConfig, MelConfig, StftConfig, mel_spectrogram, stft},
    temporal::estimate_tempo,
};

#[derive(Serialize)]
struct BenchResult {
    audio: String,
    time_sec: f64,
    tempo_bpm: f32,
    n_frames: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    timings: Option<TimingBreakdown>,
}

#[derive(Serialize)]
struct TimingBreakdown {
    load_ms: f64,
    frame_ms: f64,
    stft_ms: f64,
    mel_ms: f64,
    mfcc_ms: f64,
    onset_ms: f64,
    tempo_ms: f64,
}

fn main() {
    let path = std::env::args().nth(1).expect("audio path");
    let detailed = std::env::args().any(|arg| arg == "--detailed");

    let t_total = Instant::now();

    let t0 = Instant::now();
    let buffer = audio::load(&path).unwrap();
    let load_time = t0.elapsed();

    let t1 = Instant::now();
    let frames = frame(&buffer, FrameConfig::default()).unwrap();
    let frame_time = t1.elapsed();

    let t2 = Instant::now();
    let spec = stft(&frames, StftConfig::default()).unwrap();
    let stft_time = t2.elapsed();

    let t3 = Instant::now();
    let mel = mel_spectrogram(&spec, MelConfig::default());
    let mel_time = t3.elapsed();

    let t4 = Instant::now();
    let mfcc = mfcc(&mel, MfccConfig::default());
    let mfcc_time = t4.elapsed();

    let t5 = Instant::now();
    let onset = onset_strength_from_mel(&mel, DbConfig::default());
    let onset_time = t5.elapsed();

    let t6 = Instant::now();
    let tempo = estimate_tempo(
        onset.values(),
        frames.sample_rate(),
        frames.hop_size(),
        Default::default(),
    );
    let tempo_time = t6.elapsed();

    let elapsed = t_total.elapsed().as_secs_f64();

    let timings = if detailed {
        Some(TimingBreakdown {
            load_ms: load_time.as_secs_f64() * 1000.0,
            frame_ms: frame_time.as_secs_f64() * 1000.0,
            stft_ms: stft_time.as_secs_f64() * 1000.0,
            mel_ms: mel_time.as_secs_f64() * 1000.0,
            mfcc_ms: mfcc_time.as_secs_f64() * 1000.0,
            onset_ms: onset_time.as_secs_f64() * 1000.0,
            tempo_ms: tempo_time.as_secs_f64() * 1000.0,
        })
    } else {
        None
    };

    let res = BenchResult {
        audio: path,
        time_sec: elapsed,
        tempo_bpm: tempo.bpm,
        n_frames: mfcc.n_frames(),
        timings,
    };

    println!("{}", serde_json::to_string(&res).unwrap());

    if detailed {
        eprintln!("\n=== Detailed Timing Breakdown ===");
        if let Some(t) = &res.timings {
            eprintln!(
                "Audio Load:  {:>8.2} ms ({:>5.1}%)",
                t.load_ms,
                t.load_ms / elapsed / 10.0
            );
            eprintln!(
                "Framing:     {:>8.2} ms ({:>5.1}%)",
                t.frame_ms,
                t.frame_ms / elapsed / 10.0
            );
            eprintln!(
                "STFT:        {:>8.2} ms ({:>5.1}%)",
                t.stft_ms,
                t.stft_ms / elapsed / 10.0
            );
            eprintln!(
                "Mel Spec:    {:>8.2} ms ({:>5.1}%)",
                t.mel_ms,
                t.mel_ms / elapsed / 10.0
            );
            eprintln!(
                "MFCC:        {:>8.2} ms ({:>5.1}%)",
                t.mfcc_ms,
                t.mfcc_ms / elapsed / 10.0
            );
            eprintln!(
                "Onset:       {:>8.2} ms ({:>5.1}%)",
                t.onset_ms,
                t.onset_ms / elapsed / 10.0
            );
            eprintln!(
                "Tempo:       {:>8.2} ms ({:>5.1}%)",
                t.tempo_ms,
                t.tempo_ms / elapsed / 10.0
            );
            eprintln!("---");
            eprintln!("Total:       {:>8.2} ms", elapsed * 1000.0);
        }
    }
}
