use serde::Serialize;
use std::time::Instant;

use rsona::{
    audio,
    feature::{MfccConfig, OnsetConfig, mfcc, onset_strength},
    signal::{FrameConfig, frame},
    similarity::{
        FrameFeatures, LagEnergyConfig, RepeatPhaseConfig, SelfSimilarity, SelfSimilarityConfig,
        best_repeat_phase, estimate_repeat_lag, self_similarity,
    },
    spectrum::{MelConfig, StftConfig, mel_spectrogram, stft},
    structure::{SegmentationConfig, segment_intro_loop_outro_with_onsets},
    temporal::{BeatConfig, TempoConfig, estimate_tempo, track_beats},
};

#[derive(Serialize)]
struct ImprovedLoopResult {
    audio_file: String,
    duration_seconds: f64,
    sample_rate: u32,
    n_samples: usize,
    n_frames: usize,

    // Timing
    timing: Timing,

    // Original rsona result
    rsona_original: LoopSegment,

    // Improved result with similarity refinement
    rsona_improved: LoopSegment,

    // Analysis
    analysis: Analysis,
}

#[derive(Serialize)]
struct Timing {
    total_ms: f64,
    features_ms: f64,
    similarity_ms: f64,
    lag_estimation_ms: f64,
    phase_search_ms: f64,
    similarity_refinement_ms: f64,
}

#[derive(Serialize)]
struct LoopSegment {
    loop_begin_sample: usize,
    loop_end_sample: usize,
    loop_begin_seconds: f64,
    loop_end_seconds: f64,
    loop_duration_seconds: f64,
    loop_begin_frame: usize,
    loop_end_frame: usize,
    lag_frames: usize,
    confidence: f32,
    score: f32,
}

#[derive(Serialize)]
struct Analysis {
    detected_lag_frames: usize,
    detected_lag_seconds: f64,
    n_phase_candidates_tested: usize,
    best_phase_similarity: f32,
    tempo_bpm: Option<f32>,
    beat_frames: Option<usize>,
    top_phase_candidates: Vec<PhaseCandidate>,
}

#[derive(Serialize, Clone, Debug)]
struct PhaseCandidate {
    frame: usize,
    seconds: f64,
    score: f32,
}

    if detailed {
        eprintln!("\nPhase comparison:");
        eprintln!(
            "  Original: frame {} ({:.2}s)",
            original_start,
            original_start as f64 * hop_size as f64 / sample_rate as f64
        );
        eprintln!(
            "  Improved: frame {} ({:.2}s), score {:.4}",
            improved_start,
            improved_start as f64 * hop_size as f64 / sample_rate as f64,
            improved_score
        );
    }

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <audio_path> [--detailed]", args[0]);
        std::process::exit(1);
    }

    let audio_path = &args[1];
    let detailed = args.iter().any(|arg| arg == "--detailed");

    let total_start = Instant::now();

    // --- Load and prepare audio ---
    let buffer = audio::load(audio_path).expect("Failed to load audio");
    let duration_seconds = buffer.samples.len() as f64 / buffer.sample_rate as f64;

    let frame_cfg = FrameConfig {
        frame_size: 2048,
        hop_size: 512,
        ..Default::default()
    };

    let hop_size = frame_cfg.hop_size;
    let sample_rate = buffer.sample_rate;

    if detailed {
        eprintln!("Audio: {:.2}s @ {}Hz", duration_seconds, sample_rate);
    }

    // --- Feature extraction ---
    let features_start = Instant::now();

    let frames = frame(&buffer, frame_cfg).expect("Failed to frame audio");
    let n_frames = frames.n_frames();
    let spec = stft(&frames, StftConfig::default()).expect("Failed to compute STFT");
    let mel = mel_spectrogram(&spec, MelConfig::default());
    let mfcc_result = mfcc(&mel, MfccConfig::default());
    let onset = onset_strength(&spec, OnsetConfig::default());

    let features_time = features_start.elapsed();

    if detailed {
        eprintln!(
            "Features: {} frames, {} MFCCs",
            n_frames,
            mfcc_result.n_mfcc()
        );
    }

    // --- Self-similarity matrix ---
    let similarity_start = Instant::now();

    let max_ssm_lag = (n_frames * 3 / 4).min(6000).max(800);
    let feats = FrameFeatures::new(
        mfcc_result.n_frames(),
        mfcc_result.n_mfcc(),
        mfcc_result.as_slice(),
    );

    let ssm = self_similarity(
        &feats,
        SelfSimilarityConfig {
            max_lag: Some(max_ssm_lag),
            ..Default::default()
        },
    );

    let similarity_time = similarity_start.elapsed();

    if detailed {
        eprintln!("SSM: {} x {} frames", ssm.n_frames(), ssm.n_frames());
    }

    // --- Estimate repeat lag (period) ---
    let lag_start = Instant::now();

    let max_loop_lag = (n_frames * 2 / 3).min(5000).max(800);
    let lag_config = LagEnergyConfig {
        min_lag: 20,
        max_lag: max_loop_lag,
        ..Default::default()
    };

    let lag_est = estimate_repeat_lag(&ssm, lag_config).expect("Failed to estimate lag");
    let detected_lag = lag_est.best_lag;

    let lag_time = lag_start.elapsed();

    if detailed {
        eprintln!(
            "Detected lag: {} frames ({:.2}s)",
            detected_lag,
            detected_lag as f64 * hop_size as f64 / sample_rate as f64
        );
    }

    // --- Original rsona phase search ---
    let phase_start = Instant::now();

    let phase_config = RepeatPhaseConfig {
        window: 50,
        smooth: Some(15),
    };

    let phase_est =
        best_repeat_phase(&ssm, detected_lag, phase_config.clone()).expect("Failed to find phase");

    let phase_time = phase_start.elapsed();

    let original_start = phase_est.start_frame;
    let original_end = phase_est.end_frame;

    // --- Improved: Similarity-based phase refinement ---
    let refinement_start = Instant::now();

    // Search multiple phase candidates with broader similarity scoring
    let (improved_start, improved_score, top_candidates) =
        find_best_phase_by_similarity(&ssm, detected_lag, hop_size, sample_rate, detailed);

    let improved_end = improved_start + detected_lag;

    let refinement_time = refinement_start.elapsed();

    if detailed {
        eprintln!("\nPhase comparison:");
        eprintln!(
            "  Original: frame {} ({:.2}s)",
            original_start,
            original_start as f64 * hop_size as f64 / sample_rate as f64
        );
        eprintln!(
            "  Improved: frame {} ({:.2}s), score {:.4}",
            improved_start,
            improved_start as f64 * hop_size as f64 / sample_rate as f64,
            improved_score
        );
    }

    // --- Tempo/beat detection ---
    let tempo = estimate_tempo(
        onset.values(),
        frames.sample_rate(),
        frames.hop_size(),
        TempoConfig::default(),
    );

    let beats = track_beats(
        onset.values(),
        frames.sample_rate(),
        frames.hop_size(),
        tempo.period_frames,
        BeatConfig::default(),
    );

    // --- Build results ---
    let total_time = total_start.elapsed();

    let to_seconds = |frame: usize| -> f64 { frame as f64 * hop_size as f64 / sample_rate as f64 };

    let to_sample = |frame: usize| -> usize { frame * hop_size };

    let rsona_original = LoopSegment {
        loop_begin_sample: to_sample(original_start),
        loop_end_sample: to_sample(original_end),
        loop_begin_seconds: to_seconds(original_start),
        loop_end_seconds: to_seconds(original_end),
        loop_duration_seconds: to_seconds(original_end) - to_seconds(original_start),
        loop_begin_frame: original_start,
        loop_end_frame: original_end,
        lag_frames: detected_lag,
        confidence: phase_est.confidence,
        score: phase_est.score,
    };

    let rsona_improved = LoopSegment {
        loop_begin_sample: to_sample(improved_start),
        loop_end_sample: to_sample(improved_end),
        loop_begin_seconds: to_seconds(improved_start),
        loop_end_seconds: to_seconds(improved_end),
        loop_duration_seconds: to_seconds(improved_end) - to_seconds(improved_start),
        loop_begin_frame: improved_start,
        loop_end_frame: improved_end,
        lag_frames: detected_lag,
        confidence: phase_est.confidence, // Reuse for now
        score: improved_score,
    };

    let result = ImprovedLoopResult {
        audio_file: audio_path.to_string(),
        duration_seconds,
        sample_rate,
        n_samples: buffer.samples.len(),
        n_frames,
        timing: Timing {
            total_ms: total_time.as_secs_f64() * 1000.0,
            features_ms: features_time.as_secs_f64() * 1000.0,
            similarity_ms: similarity_time.as_secs_f64() * 1000.0,
            lag_estimation_ms: lag_time.as_secs_f64() * 1000.0,
            phase_search_ms: phase_time.as_secs_f64() * 1000.0,
            similarity_refinement_ms: refinement_time.as_secs_f64() * 1000.0,
        },
        rsona_original,
        rsona_improved,
        analysis: Analysis {
            detected_lag_frames: detected_lag,
            detected_lag_seconds: to_seconds(detected_lag),
            n_phase_candidates_tested: top_candidates.len(),
            best_phase_similarity: improved_score,
            tempo_bpm: Some(tempo.bpm),
            beat_frames: Some(beats.beat_frames.len()),
            top_phase_candidates: top_candidates.iter().take(20).cloned().collect(),
        },
    };

    // Output
    println!("{}", serde_json::to_string_pretty(&result).unwrap());

    if detailed {
        eprintln!("\n=== Summary ===");
        eprintln!(
            "Original: {:.2}s → {:.2}s ({:.2}s loop)",
            result.rsona_original.loop_begin_seconds,
            result.rsona_original.loop_end_seconds,
            result.rsona_original.loop_duration_seconds,
        );
        eprintln!(
            "Improved: {:.2}s → {:.2}s ({:.2}s loop)",
            result.rsona_improved.loop_begin_seconds,
            result.rsona_improved.loop_end_seconds,
            result.rsona_improved.loop_duration_seconds,
        );
        eprintln!(
            "\nPhase shift: {:.2}s",
            (result.rsona_improved.loop_begin_seconds - result.rsona_original.loop_begin_seconds)
                .abs()
        );
    }
}

/// Find best phase by computing cumulative similarity scores over entire loop region
fn find_best_phase_by_similarity(
    ssm: &SelfSimilarity,
    lag: usize,
    hop_size: usize,
    sample_rate: u32,
    detailed: bool,
) -> (usize, f32, Vec<PhaseCandidate>) {
    let n = ssm.n_frames();
    let max_start = n.saturating_sub(lag + 10);

    if max_start < 10 {
        return (0, 0.0, Vec::new());
    }

    let to_seconds = |frame: usize| -> f64 { frame as f64 * hop_size as f64 / sample_rate as f64 };

    // Search more densely to find the phase librosa found
    let search_start = max_start / 20; // Start earlier
    let search_end = max_start * 95 / 100; // Search almost to the end
    let step = ((search_end - search_start) / 500).max(1); // More candidates

    let mut candidates: Vec<PhaseCandidate> = Vec::new();

    if detailed {
        eprintln!(
            "\nSearching phase candidates: frame {} ({:.1}s) to {} ({:.1}s), step {}",
            search_start,
            to_seconds(search_start),
            search_end,
            to_seconds(search_end),
            step
        );
    }

    // Coarse search
    for start in (search_start..=search_end).step_by(step) {
        let score = compute_loop_similarity_score(ssm, start, lag);
        candidates.push(PhaseCandidate {
            frame: start,
            seconds: to_seconds(start),
            score,
        });
    }

    // Sort by score descending
    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    let best_coarse = candidates[0].frame;
    let best_score_coarse = candidates[0].score;

    // Fine search around top 5 coarse candidates
    let mut fine_candidates: Vec<PhaseCandidate> = Vec::new();
    for candidate in candidates.iter().take(5) {
        let refine_start = candidate.frame.saturating_sub(step);
        let refine_end = (candidate.frame + step).min(max_start);

        for start in refine_start..=refine_end {
            let score = compute_loop_similarity_score(ssm, start, lag);
            fine_candidates.push(PhaseCandidate {
                frame: start,
                seconds: to_seconds(start),
                score,
            });
        }
    }

    // Combine and sort all candidates
    candidates.extend(fine_candidates);
    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    candidates.dedup_by_key(|c| c.frame); // Remove duplicates

    let best_start = candidates[0].frame;
    let best_score = candidates[0].score;

    if detailed {
        eprintln!(
            "Tested {} unique candidates, best at frame {} ({:.2}s) with score {:.6}",
            candidates.len(),
            best_start,
            to_seconds(best_start),
            best_score
        );
    }

    (best_start, best_score, candidates)
}

/// Compute similarity score for a loop region
/// Scores how well frames [start..start+lag] match [start+lag..start+2*lag]
fn compute_loop_similarity_score(ssm: &SelfSimilarity, start: usize, lag: usize) -> f32 {
    let n = ssm.n_frames();
    let loop_end = start + lag;

    if loop_end + lag > n {
        return 0.0;
    }

    // Compute average similarity between loop region and its repeat
    let mut sum = 0.0f64;
    let mut count = 0u64;

    // Sample points throughout the loop (every Nth frame to be efficient)
    let sample_step = (lag / 50).max(1);

    for offset in (0..lag).step_by(sample_step) {
        let frame_a = start + offset;
        let frame_b = loop_end + offset;

        if frame_b < n {
            sum += ssm.value(frame_a, frame_b) as f64;
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        (sum / count as f64) as f32
    }
}
