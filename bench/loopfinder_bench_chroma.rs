//! Beat-synchronized loopfinder using chroma features.
//!
//! Implementation approach:
//! - Extract beat-synchronized chroma features
//! - Search for best loop at beat level (not frame level)
//! - Use distance/similarity metrics on beat-level features
//!
//! Usage:
//!   cargo build --release --bin loopfinder_bench_chroma
//!   ./target/release/loopfinder_bench_chroma <audio_file>

use serde::Serialize;
use std::time::Instant;

use rsona::{
    audio,
    feature::{ChromaConfig, ChromaNorm, chroma_stft, onset_strength_from_mel},
    signal::{FrameConfig, frame},
    spectrum::{MelConfig, StftConfig, mel_spectrogram, stft},
    temporal::{
        AggregationMethod, BeatConfig, SyncConfig, TempoConfig, estimate_tempo, sync_to_beats,
        track_beats,
    },
};

#[derive(Serialize)]
struct LoopfinderChromaResult {
    audio_file: String,
    duration_seconds: f64,
    sample_rate: u32,
    n_samples: usize,
    parameters: Parameters,
    timing: Timing,
    tempo: TempoInfo,
    best_result: BestLoop,
    segmentation: Segmentation,
    approach: String,
}

#[derive(Serialize)]
struct Parameters {
    hop_size: usize,
    frame_size: usize,
    n_chroma: usize,
    min_loop_beats: usize,
    normalization: String,
}

#[derive(Serialize)]
struct Timing {
    load_time_ms: f64,
    frame_time_ms: f64,
    stft_time_ms: f64,
    chroma_time_ms: f64,
    mel_time_ms: f64,
    onset_time_ms: f64,
    tempo_time_ms: f64,
    beat_time_ms: f64,
    sync_time_ms: f64,
    search_time_ms: f64,
    total_time_ms: f64,
}

#[derive(Serialize)]
struct TempoInfo {
    bpm: f32,
    period_frames: usize,
    n_beats: usize,
}

#[derive(Serialize)]
struct BestLoop {
    loop_begin_sample: usize,
    loop_end_sample: usize,
    loop_begin_seconds: f64,
    loop_end_seconds: f64,
    loop_duration_seconds: f64,
    loop_begin_beat: usize,
    loop_end_beat: usize,
    loop_begin_frame: usize,
    loop_end_frame: usize,
    similarity_score: f32,
}

#[derive(Serialize)]
struct Segmentation {
    intro_seconds: f64,
    loop_seconds: f64,
    outro_seconds: f64,
    intro_beats: usize,
    loop_beats: usize,
    outro_beats: usize,
}

/// Compute Manhattan distance between two beat-chroma vectors.
fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Find best loop by searching all beat pairs.
/// Returns (start_beat, end_beat, distance).
fn find_best_loop_beats(
    beat_chroma: &ndarray::Array2<f32>,
    min_loop_beats: usize,
) -> (usize, usize, f32) {
    let n_beats = beat_chroma.nrows();

    let mut best_start = 0;
    let mut best_end = min_loop_beats;
    let mut best_distance = f32::MAX;

    // Search all valid beat pairs
    for start in 0..n_beats {
        for end in (start + min_loop_beats)..n_beats {
            // Compute distance between start and end beats
            let start_features = beat_chroma.row(start);
            let end_features = beat_chroma.row(end);

            let distance = manhattan_distance(
                start_features.as_slice().unwrap(),
                end_features.as_slice().unwrap(),
            );

            if distance < best_distance {
                best_distance = distance;
                best_start = start;
                best_end = end;
            }
        }
    }

    (best_start, best_end, best_distance)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <audio_path>", args[0]);
        std::process::exit(1);
    }

    let audio_path = &args[1];
    let detailed = args.contains(&"--detailed".to_string());

    let total_start = Instant::now();

    // --- Load Audio ---
    let t_load = Instant::now();
    let buffer = audio::load(audio_path).expect("Failed to load audio");
    let load_time = t_load.elapsed();

    let duration_seconds = buffer.samples.len() as f64 / buffer.sample_rate as f64;
    let n_samples = buffer.samples.len();
    let sr = buffer.sample_rate;

    if detailed {
        eprintln!("Loaded: {} samples @ {} Hz", n_samples, sr);
        eprintln!("Duration: {:.2}s", duration_seconds);
    }

    // --- Frame ---
    let frame_cfg = FrameConfig {
        frame_size: 2048,
        hop_size: 512,
        ..Default::default()
    };

    let hop_size = frame_cfg.hop_size;
    let frame_size = frame_cfg.frame_size;

    let t_frame = Instant::now();
    let frames = frame(&buffer, frame_cfg).expect("Failed to frame audio");
    let frame_time = t_frame.elapsed();

    let n_frames = frames.n_frames();

    if detailed {
        eprintln!("Frames: {}", n_frames);
    }

    // --- STFT ---
    let t_stft = Instant::now();
    let spec = stft(&frames, StftConfig::default()).expect("Failed to compute STFT");
    let stft_time = t_stft.elapsed();

    if detailed {
        eprintln!("STFT: {} frames × {} bins", spec.n_frames(), spec.n_bins());
    }

    // --- Chroma ---
    let t_chroma = Instant::now();
    let chroma = chroma_stft(
        &spec,
        ChromaConfig {
            n_chroma: 12,
            norm: ChromaNorm::L2,
            fmin: 32.7,
            fmax: None,
            ..Default::default()
        },
    );
    let chroma_time = t_chroma.elapsed();

    if detailed {
        eprintln!(
            "Chroma: {} frames × {} bins",
            chroma.n_frames(),
            chroma.n_chroma()
        );
    }

    // --- Mel Spectrogram for onset detection ---
    let t_mel = Instant::now();
    let mel = mel_spectrogram(&spec, MelConfig::default());
    let mel_time = t_mel.elapsed();

    // --- Onset Strength ---
    let t_onset = Instant::now();
    let onset_env = onset_strength_from_mel(&mel, Default::default());
    let onset_time = t_onset.elapsed();

    if detailed {
        eprintln!("Onset envelope: {} frames", onset_env.n_frames());
    }

    // --- Tempo Estimation ---
    let t_tempo = Instant::now();
    let tempo = estimate_tempo(onset_env.values(), sr, hop_size, TempoConfig::default());
    let tempo_time = t_tempo.elapsed();

    if detailed {
        eprintln!("Tempo: {:.2} BPM", tempo.bpm);
        eprintln!("Period: {} frames per beat", tempo.period_frames);
    }

    // --- Beat Tracking ---
    let t_beat = Instant::now();
    let beats = track_beats(
        onset_env.values(),
        sr,
        hop_size,
        tempo.period_frames,
        BeatConfig::default(),
    );
    let beat_time = t_beat.elapsed();

    let n_beats = beats.beat_frames.len();

    if detailed {
        eprintln!("Detected {} beats", n_beats);
    }

    // --- Synchronize Chroma to Beats ---
    let t_sync = Instant::now();
    let beat_chroma = sync_to_beats(
        chroma.as_ndarray(),
        &beats.beat_frames,
        SyncConfig {
            method: AggregationMethod::Mean,
            pad_end: true,
        },
    );
    let sync_time = t_sync.elapsed();

    if detailed {
        eprintln!(
            "Beat-synchronized chroma: {} beats × {} bins",
            beat_chroma.nrows(),
            beat_chroma.ncols()
        );
    }

    // --- Find Best Loop ---
    let min_loop_beats = (10.0 * tempo.bpm / 60.0).round() as usize; // ~10 seconds worth of beats

    let t_search = Instant::now();
    let (best_start_beat, best_end_beat, similarity_score) =
        find_best_loop_beats(&beat_chroma, min_loop_beats);
    let search_time = t_search.elapsed();

    // Convert beat indices to frame/sample/time
    let loop_begin_frame = beats.beat_frames[best_start_beat];
    let loop_end_frame = beats.beat_frames[best_end_beat];

    let loop_begin_sample = loop_begin_frame * hop_size;
    let loop_end_sample = loop_end_frame * hop_size;

    let loop_begin_seconds = beats.beat_times[best_start_beat];
    let loop_end_seconds = beats.beat_times[best_end_beat];

    let loop_duration_seconds = loop_end_seconds - loop_begin_seconds;

    // Simple segmentation: intro = before loop, loop = loop region, outro = after loop
    let intro_beats = best_start_beat;
    let loop_beats = best_end_beat - best_start_beat;
    let outro_beats = n_beats - best_end_beat;

    let intro_seconds = loop_begin_seconds;
    let loop_seconds = loop_duration_seconds;
    let outro_seconds = duration_seconds - loop_end_seconds;

    let total_time = total_start.elapsed();

    // --- Build Result ---
    let result = LoopfinderChromaResult {
        audio_file: audio_path.clone(),
        duration_seconds,
        sample_rate: sr,
        n_samples,
        parameters: Parameters {
            hop_size,
            frame_size,
            n_chroma: 12,
            min_loop_beats,
            normalization: "L2".to_string(),
        },
        timing: Timing {
            load_time_ms: load_time.as_secs_f64() * 1000.0,
            frame_time_ms: frame_time.as_secs_f64() * 1000.0,
            stft_time_ms: stft_time.as_secs_f64() * 1000.0,
            chroma_time_ms: chroma_time.as_secs_f64() * 1000.0,
            mel_time_ms: mel_time.as_secs_f64() * 1000.0,
            onset_time_ms: onset_time.as_secs_f64() * 1000.0,
            tempo_time_ms: tempo_time.as_secs_f64() * 1000.0,
            beat_time_ms: beat_time.as_secs_f64() * 1000.0,
            sync_time_ms: sync_time.as_secs_f64() * 1000.0,
            search_time_ms: search_time.as_secs_f64() * 1000.0,
            total_time_ms: total_time.as_secs_f64() * 1000.0,
        },
        tempo: TempoInfo {
            bpm: tempo.bpm,
            period_frames: tempo.period_frames,
            n_beats,
        },
        best_result: BestLoop {
            loop_begin_sample,
            loop_end_sample,
            loop_begin_seconds,
            loop_end_seconds,
            loop_duration_seconds,
            loop_begin_beat: best_start_beat,
            loop_end_beat: best_end_beat,
            loop_begin_frame,
            loop_end_frame,
            similarity_score,
        },
        segmentation: Segmentation {
            intro_seconds,
            loop_seconds,
            outro_seconds,
            intro_beats,
            loop_beats,
            outro_beats,
        },
        approach: "beat_synchronized_chroma".to_string(),
    };

    // Output JSON
    println!("{}", serde_json::to_string_pretty(&result).unwrap());

    if detailed {
        eprintln!("\n=== Results ===");
        eprintln!("Tempo: {:.2} BPM ({} beats)", tempo.bpm, n_beats);
        eprintln!(
            "Best loop: beat {} → beat {} ({} beats)",
            best_start_beat, best_end_beat, loop_beats
        );
        eprintln!(
            "  Time: {:.2}s → {:.2}s ({:.2}s duration)",
            loop_begin_seconds, loop_end_seconds, loop_duration_seconds
        );
        eprintln!(
            "  Frames: {} → {} (sample {} → {})",
            loop_begin_frame, loop_end_frame, loop_begin_sample, loop_end_sample
        );
        eprintln!("  Similarity score: {:.6}", similarity_score);
        eprintln!("\nSegmentation:");
        eprintln!("  Intro: {:.2}s ({} beats)", intro_seconds, intro_beats);
        eprintln!("  Loop:  {:.2}s ({} beats)", loop_seconds, loop_beats);
        eprintln!("  Outro: {:.2}s ({} beats)", outro_seconds, outro_beats);
        eprintln!("\nTotal time: {:.2}ms", total_time.as_secs_f64() * 1000.0);
    }
}
