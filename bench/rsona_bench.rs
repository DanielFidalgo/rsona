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
    #[serde(skip_serializing_if = "Option::is_none")]
    correctness: Option<CorrectnessMetrics>,
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
    n_mfcc_coeffs: usize,
}

#[derive(Serialize)]
struct CorrectnessMetrics {
    tempo_within_tolerance: bool,
    frame_count_correct: bool,
    all_features_valid: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    validation_notes: Option<Vec<String>>,
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

    // Validate correctness
    let mut validation_notes = Vec::new();
    let mut all_valid = true;

    // Check tempo is reasonable (30-300 BPM range for most music)
    let tempo_valid = tempo.bpm >= 30.0 && tempo.bpm <= 300.0;
    if !tempo_valid {
        validation_notes.push(format!("Tempo {} BPM outside typical range", tempo.bpm));
        all_valid = false;
    }

    // Check frame count matches expected
    let expected_frames = (n_samples / hop_size).saturating_sub(1);
    let frame_count_valid = n_frames == expected_frames;
    if !frame_count_valid {
        validation_notes.push(format!(
            "Frame count {} doesn't match expected {}",
            n_frames, expected_frames
        ));
        all_valid = false;
    }

    // Check all features produced valid output
    let features_valid = n_mfcc > 0 && n_frequency_bins > 0 && n_frames > 0;
    if !features_valid {
        validation_notes.push("One or more features produced invalid output".to_string());
        all_valid = false;
    }

    let correctness = Some(CorrectnessMetrics {
        tempo_within_tolerance: tempo_valid,
        frame_count_correct: frame_count_valid,
        all_features_valid: features_valid && all_valid,
        validation_notes: if validation_notes.is_empty() {
            None
        } else {
            Some(validation_notes)
        },
    });

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
            n_mfcc_coeffs: n_mfcc,
        },
        correctness,
    };

    // Output JSON
    let json = serde_json::to_string_pretty(&result).unwrap_or_else(|e| {
        eprintln!("Error serializing result: {}", e);
        std::process::exit(1);
    });

    println!("{}", json);
}
