use serde::Serialize;
use std::time::Instant;

use rsona::{
    audio,
    feature::{
        ChromaConfig, ChromaNorm, MfccConfig, OnsetConfig, chroma_stft, mfcc, onset_strength,
    },
    signal::{FrameConfig, frame},
    spectrum::{MelConfig, StftConfig, mel_spectrogram, stft},
    structure::{BeatLoopConfig, DistanceMetric, LoopPreference, find_loop_by_beats},
    temporal::{BeatConfig, TempoConfig, estimate_tempo, track_beats},
};

#[derive(Serialize)]
struct LoopfinderBeatsResult {
    audio_file: String,
    duration_seconds: f64,
    sample_rate: u32,
    n_samples: usize,
    parameters: Parameters,
    timing: Timing,
    best_result: BestLoop,
    segmentation: Segmentation,
    approach: String,
}

#[derive(Serialize)]
struct Parameters {
    hop_size: usize,
    frame_size: usize,
    n_chroma: usize,
    n_mfcc: usize,
    min_loop_seconds: f64,
    feature_window_frames: usize,
    distance_metric: String,
    length_preference: f32,
    strategy: String,
    tempo_bpm: Option<f32>,
}

#[derive(Serialize)]
struct Timing {
    load_time_ms: f64,
    frame_time_ms: f64,
    stft_time_ms: f64,
    mel_time_ms: f64,
    chroma_time_ms: f64,
    mfcc_time_ms: f64,
    onset_time_ms: f64,
    tempo_time_ms: f64,
    beat_time_ms: f64,
    loop_search_time_ms: f64,
    total_time_ms: f64,
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
    score: f32,
    normalized_score: f32,
}

#[derive(Serialize)]
struct Segmentation {
    intro_seconds: f64,
    loop_seconds: f64,
    outro_seconds: f64,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: {} <audio_path> [--detailed] [--length-preference=<value>] [--strategy=<mode>]",
            args[0]
        );
        eprintln!("  --length-preference: 0.0 (pure similarity) to 0.1+ (prefer longer loops)");
        eprintln!("                       Default: 0.0, Recommended: 0.02-0.05");
        eprintln!("  --strategy:          similarity | balanced | musical");
        eprintln!("                       similarity: Tightest match (game loops)");
        eprintln!("                       musical: Phrase-aligned (DJ/composition)");
        eprintln!("                       balanced: Hybrid approach");
        std::process::exit(1);
    }

    let audio_path = &args[1];
    let detailed = args.iter().any(|arg| arg == "--detailed");

    // Parse length preference
    let length_preference = args
        .iter()
        .find(|arg| arg.starts_with("--length-preference="))
        .and_then(|arg| arg.strip_prefix("--length-preference="))
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.0);

    // Parse strategy
    let strategy_str = args
        .iter()
        .find(|arg| arg.starts_with("--strategy="))
        .and_then(|arg| arg.strip_prefix("--strategy="))
        .map(|s| s.to_lowercase())
        .unwrap_or_else(|| "similarity".to_string());

    let preference = match strategy_str.as_str() {
        "musical" | "musical_structure" => LoopPreference::MusicalStructure,
        "balanced" => LoopPreference::Balanced,
        _ => LoopPreference::Similarity,
    };

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

    // --- Mel Spectrogram ---
    let t_mel = Instant::now();
    let mel = mel_spectrogram(&spec, MelConfig::default());
    let mel_time = t_mel.elapsed();

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

    // --- MFCC ---
    let t_mfcc = Instant::now();
    let mfcc_result = mfcc(&mel, MfccConfig::default());
    let mfcc_time = t_mfcc.elapsed();

    if detailed {
        eprintln!(
            "MFCC: {} frames × {} coefficients",
            mfcc_result.n_frames(),
            mfcc_result.n_mfcc()
        );
    }

    // --- Onset Strength ---
    let t_onset = Instant::now();
    let onset_env = onset_strength(&spec, OnsetConfig::default());
    let onset_time = t_onset.elapsed();

    if detailed {
        eprintln!("Onset envelope: {} frames", onset_env.values().len());
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

    // --- Find Loop Using Beat-Synchronized Features ---
    let min_loop_seconds = 10.0;
    let min_loop_samples = (min_loop_seconds * sr as f64) as usize;

    let loop_config = BeatLoopConfig {
        min_length_samples: min_loop_samples,
        max_length_samples: None,
        feature_window_frames: 50, // ~0.6 seconds worth of frames
        max_candidates: 100,
        metric: DistanceMetric::Manhattan,
        length_preference,
        preference,
        tempo_bpm: Some(tempo.bpm),
    };

    let t_loop = Instant::now();
    let loop_result = find_loop_by_beats(
        &frames,
        &beats,
        &chroma,
        &mfcc_result,
        &onset_env,
        loop_config.clone(),
    )
    .expect("Failed to find loop");
    let loop_time = t_loop.elapsed();

    let total_time = total_start.elapsed();

    if detailed {
        eprintln!(
            "\nBest loop: beat {} → beat {} ({} beats)",
            loop_result.start_beat,
            loop_result.end_beat,
            loop_result.end_beat - loop_result.start_beat
        );
        eprintln!(
            "  Time: {:.2}s → {:.2}s ({:.2}s duration)",
            loop_result.start_seconds, loop_result.end_seconds, loop_result.duration_seconds
        );
        eprintln!(
            "  Score: {:.6} (length_preference: {:.3})",
            loop_result.score, length_preference
        );

        // Show top 10 candidates
        eprintln!("\nTop 10 candidates:");
        for (i, (start_beat, end_beat, score)) in loop_result.candidates.iter().take(10).enumerate()
        {
            let start_sec = beats.beat_times[*start_beat];
            let end_sec = beats.beat_times[*end_beat];
            let duration = end_sec - start_sec;
            let num_beats = end_beat - start_beat;
            let num_bars = num_beats as f32 / 4.0;
            eprintln!(
                "  {}. beats {}-{} ({} beats, {:.1} bars): {:.2}s-{:.2}s ({:.2}s) score={:.4}",
                i + 1,
                start_beat,
                end_beat,
                num_beats,
                num_bars,
                start_sec,
                end_sec,
                duration,
                score
            );
        }
    }

    // --- Build Result ---
    let intro_seconds = loop_result.start_seconds;
    let loop_seconds = loop_result.duration_seconds;
    let outro_seconds = duration_seconds - loop_result.end_seconds;

    // Normalize score to [0, 1] range (lower is better)
    // For Manhattan distance on normalized features, typical range is 0-50
    let normalized_score = (loop_result.score / 50.0).min(1.0);

    let result = LoopfinderBeatsResult {
        audio_file: audio_path.clone(),
        duration_seconds,
        sample_rate: sr,
        n_samples,
        parameters: Parameters {
            hop_size,
            frame_size,
            n_chroma: 12,
            n_mfcc: mfcc_result.n_mfcc(),
            min_loop_seconds,
            feature_window_frames: 50,
            distance_metric: "manhattan".to_string(),
            length_preference,
            strategy: format!("{:?}", preference),
            tempo_bpm: Some(tempo.bpm),
        },
        timing: Timing {
            load_time_ms: load_time.as_secs_f64() * 1000.0,
            frame_time_ms: frame_time.as_secs_f64() * 1000.0,
            stft_time_ms: stft_time.as_secs_f64() * 1000.0,
            mel_time_ms: mel_time.as_secs_f64() * 1000.0,
            chroma_time_ms: chroma_time.as_secs_f64() * 1000.0,
            mfcc_time_ms: mfcc_time.as_secs_f64() * 1000.0,
            onset_time_ms: onset_time.as_secs_f64() * 1000.0,
            tempo_time_ms: tempo_time.as_secs_f64() * 1000.0,
            beat_time_ms: beat_time.as_secs_f64() * 1000.0,
            loop_search_time_ms: loop_time.as_secs_f64() * 1000.0,
            total_time_ms: total_time.as_secs_f64() * 1000.0,
        },
        best_result: BestLoop {
            loop_begin_sample: loop_result.start_sample,
            loop_end_sample: loop_result.end_sample,
            loop_begin_seconds: loop_result.start_seconds,
            loop_end_seconds: loop_result.end_seconds,
            loop_duration_seconds: loop_result.duration_seconds,
            loop_begin_beat: loop_result.start_beat,
            loop_end_beat: loop_result.end_beat,
            loop_begin_frame: beats.beat_frames[loop_result.start_beat],
            loop_end_frame: beats.beat_frames[loop_result.end_beat],
            score: loop_result.score,
            normalized_score,
        },
        segmentation: Segmentation {
            intro_seconds,
            loop_seconds,
            outro_seconds,
        },
        approach: "beat_synchronized_pairwise".to_string(),
    };

    // Output JSON
    println!("{}", serde_json::to_string_pretty(&result).unwrap());

    if detailed {
        eprintln!("\n=== Results ===");
        eprintln!("Tempo: {:.2} BPM ({} beats)", tempo.bpm, n_beats);
        eprintln!(
            "Best loop: beat {} → beat {} ({} beats)",
            result.best_result.loop_begin_beat,
            result.best_result.loop_end_beat,
            result.best_result.loop_end_beat - result.best_result.loop_begin_beat
        );
        eprintln!(
            "  Time: {:.2}s → {:.2}s ({:.2}s duration)",
            result.best_result.loop_begin_seconds,
            result.best_result.loop_end_seconds,
            result.best_result.loop_duration_seconds
        );
        eprintln!(
            "  Frames: {} → {} (samples {} → {})",
            result.best_result.loop_begin_frame,
            result.best_result.loop_end_frame,
            result.best_result.loop_begin_sample,
            result.best_result.loop_end_sample
        );
        eprintln!(
            "  Score: {:.6} (length_preference: {:.3})",
            loop_result.score, length_preference
        );
        eprintln!("\nSegmentation:");
        eprintln!("  Intro: {:.2}s", result.segmentation.intro_seconds);
        eprintln!("  Loop:  {:.2}s", result.segmentation.loop_seconds);
        eprintln!("  Outro: {:.2}s", result.segmentation.outro_seconds);
        eprintln!("\nTotal time: {:.2}ms", result.timing.total_time_ms);
    }
}
