//! Intro / loop / outro segmentation built on top of:
//! - self-similarity
//! - lag energy (repeat length)
//! - best repeat phase (loop start)
//! - novelty boundaries (section boundaries)
//! - optional beat grid snapping (from onset envelope)

use crate::signal::Frames;
use crate::similarity::{
    LagEnergyConfig, RepeatPhaseConfig, SelfSimilarity, best_repeat_phase, estimate_repeat_lag,
};
use crate::structure::{SectionBoundaryConfig, section_boundaries_from_novelty};
use crate::temporal::{BeatConfig, TempoConfig, estimate_tempo, track_beats};

/// Segmentation kind
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentKind {
    /// Intro segment
    Intro,
    /// Loop segment
    Loop,
    /// Outro segment
    Outro,
}

/// Segment
#[derive(Debug, Clone)]
pub struct Segment {
    /// Segment kind
    pub kind: SegmentKind,
    /// Start frame
    pub start_frame: usize,
    ///End frame
    pub end_frame: usize,
    /// Start sample
    pub start_sample: usize,
    /// End sample
    pub end_sample: usize,
    /// Start seconds
    pub start_seconds: f64,
    /// End seconds
    pub end_seconds: f64,
}

/// Segmentation result
#[derive(Debug, Clone)]
pub struct SegmentationResult {
    /// Intro segment
    pub intro: Segment,
    /// Loop segment
    pub loop_seg: Segment,
    /// Outro segment
    pub outro: Segment,
    /// Best lag frames
    pub best_lag_frames: usize,
    /// Best phase start
    pub best_phase_start: usize,
    /// Number of times the loop pattern repeats
    pub num_loop_repeats: usize,
    /// Loop confidence
    pub loop_confidence: f32,
    /// Section boundaries
    pub section_boundaries: Vec<usize>,
    /// Beat frames
    pub beat_frames: Option<Vec<usize>>,
    /// Tempo BPM
    pub tempo_bpm: Option<f32>,
}

/// Segmentation configuration
#[derive(Debug, Clone)]
pub struct SegmentationConfig {
    /// Lag energy config (repeat length search).
    pub lag: LagEnergyConfig,

    /// Phase config (best loop start for given lag).
    pub phase: RepeatPhaseConfig,

    /// Section boundary config (novelty + peak pick).
    pub boundaries: SectionBoundaryConfig,

    /// If true, try to align loop start/end to nearest section boundary.
    pub snap_to_section_boundaries: bool,

    /// Maximum distance (frames) to snap to a section boundary.
    pub section_snap_radius_frames: usize,

    /// If true, estimate tempo and snap loop start/end to nearest beats.
    pub snap_to_beats: bool,

    /// Tempo config (if snap_to_beats).
    pub tempo: TempoConfig,

    /// Beat snapping config (if snap_to_beats).
    pub beats: BeatConfig,

    /// Maximum distance (frames) to snap to a beat.
    pub beat_snap_radius_frames: usize,

    /// Minimum allowed loop length in frames (guard).
    pub min_loop_frames: usize,

    /// Maximum allowed drift from estimated lag when snapping (fraction).
    /// Example: 0.25 allows ±25% drift before we revert to exact lag length.
    pub max_length_drift_frac: f32,

    /// Minimum similarity threshold for detecting continued repetition.
    /// Higher values = stricter (pattern must match closely to count as repeat).
    pub repetition_similarity_threshold: f32,

    /// If true, use exhaustive search to find best matching loop pair.
    /// This is slower but more thorough.
    pub use_exhaustive_search: bool,
}

impl Default for SegmentationConfig {
    fn default() -> Self {
        Self {
            lag: LagEnergyConfig::default(),
            phase: RepeatPhaseConfig::default(),
            boundaries: SectionBoundaryConfig::default(),
            snap_to_section_boundaries: true,
            section_snap_radius_frames: 24,
            snap_to_beats: true,
            tempo: TempoConfig::default(),
            beats: BeatConfig::default(),
            beat_snap_radius_frames: 4,
            min_loop_frames: 30,
            max_length_drift_frac: 0.25,
            repetition_similarity_threshold: 0.80,
            use_exhaustive_search: true,
        }
    }
}

/// Segment intro / loop / outro using an SSM (no onset envelope).
///
/// This uses:
/// - repeat lag estimate
/// - best repeat phase
/// - novelty boundaries for optional snapping
///
/// Beat snapping is not available without onsets; use
/// `segment_intro_loop_outro_with_onsets`.
pub fn segment_intro_loop_outro(
    frames: &Frames,
    ssm: &SelfSimilarity,
    cfg: SegmentationConfig,
) -> Option<SegmentationResult> {
    segment_intro_loop_outro_impl(frames, ssm, None, cfg)
}

/// Segment intro / loop / outro using SSM + onset envelope (recommended).
///
/// This adds tempo/beat estimation + beat snapping.
pub fn segment_intro_loop_outro_with_onsets(
    frames: &Frames,
    ssm: &SelfSimilarity,
    onset_env: &[f32],
    cfg: SegmentationConfig,
) -> Option<SegmentationResult> {
    // Sanity: onset env should match frame count
    if onset_env.len() != ssm.n_frames() {
        return None;
    }
    segment_intro_loop_outro_impl(frames, ssm, Some(onset_env), cfg)
}

// --- internal implementation ---

fn segment_intro_loop_outro_impl(
    frames: &Frames,
    ssm: &SelfSimilarity,
    onset_env: Option<&[f32]>,
    cfg: SegmentationConfig,
) -> Option<SegmentationResult> {
    let n = ssm.n_frames();
    if n < 2 {
        return None;
    }

    // 1) Estimate repeat lag (loop length)
    let lag_est = estimate_repeat_lag(ssm, cfg.lag.clone())?;
    let lag = lag_est.best_lag;

    if lag < cfg.min_loop_frames || lag >= n {
        return None;
    }

    // 2) Find loop boundaries
    let (mut loop_start, detected_loop_end, num_repeats, original_phase_start, phase_confidence) =
        if cfg.use_exhaustive_search {
            // Use exhaustive search to find best matching pair
            let (start, end, repeats) = find_best_matching_loop_pair(ssm, lag, cfg.min_loop_frames);
            // For exhaustive search, confidence is based on lag estimation only
            (start, end, repeats, start, lag_est.confidence)
        } else {
            // Use autocorrelation-based phase detection
            let phase_est = best_repeat_phase(ssm, lag, cfg.phase.clone())?;
            let loop_start = phase_est.start_frame;
            let (detected_end, repeats) =
                find_repetition_end(ssm, loop_start, lag, cfg.repetition_similarity_threshold);
            (
                loop_start,
                detected_end,
                repeats,
                phase_est.start_frame,
                phase_est.confidence,
            )
        };

    let mut loop_end = detected_loop_end;
    let original_loop_end = loop_start + lag;

    // 4) Section boundaries from novelty curve
    let boundaries = section_boundaries_from_novelty(ssm, cfg.boundaries.clone());
    let section_frames = boundaries.boundary_frames.clone();

    // 5) Optional snap to nearest section boundaries
    if cfg.snap_to_section_boundaries && !section_frames.is_empty() {
        if let Some(s) =
            snap_to_nearest(&section_frames, loop_start, cfg.section_snap_radius_frames)
        {
            loop_start = s;
        }
        if let Some(e) = snap_to_nearest(&section_frames, loop_end, cfg.section_snap_radius_frames)
        {
            loop_end = e;
        }

        if loop_end <= loop_start + 1 {
            loop_start = original_phase_start;
            loop_end = original_loop_end;
        }
    }

    // 6) Optional beat snapping (requires onset envelope)
    let mut beat_frames_out: Option<Vec<usize>> = None;
    let mut tempo_bpm_out: Option<f32> = None;

    if cfg.snap_to_beats {
        if let Some(env) = onset_env {
            let tempo = estimate_tempo(
                env,
                frames.sample_rate(),
                frames.hop_size(),
                cfg.tempo.clone(),
            );
            let beats = track_beats(
                env,
                frames.sample_rate(),
                frames.hop_size(),
                tempo.period_frames,
                cfg.beats.clone(),
            );

            tempo_bpm_out = Some(tempo.bpm);
            beat_frames_out = Some(beats.beat_frames.clone());

            if !beats.beat_frames.is_empty() {
                if let Some(s) =
                    snap_to_nearest(&beats.beat_frames, loop_start, cfg.beat_snap_radius_frames)
                {
                    loop_start = s;
                }
                if let Some(e) =
                    snap_to_nearest(&beats.beat_frames, loop_end, cfg.beat_snap_radius_frames)
                {
                    loop_end = e;
                }

                if loop_end <= loop_start + 1 {
                    loop_start = original_phase_start;
                    loop_end = original_loop_end;
                }
            }
        }
    }

    // Clamp
    loop_start = loop_start.min(n);
    loop_end = loop_end.min(n).max(loop_start);

    // If snapping shortened loop too much, fall back to detected end
    if loop_end.saturating_sub(loop_start) < cfg.min_loop_frames {
        loop_start = original_phase_start;
        loop_end = detected_loop_end;
    }

    // Validate: don't allow snapping to create loops much smaller than detected size
    let snapped_len = loop_end.saturating_sub(loop_start);
    let detected_len = detected_loop_end - original_phase_start;
    if snapped_len > 0 && detected_len > 0 {
        // If snapping reduced size by more than 25%, revert to detected end
        if (snapped_len as f32) < (detected_len as f32 * 0.75) {
            loop_end = detected_loop_end;
        }
    }

    // Final validation: ensure outro is reasonable
    let outro_len = n.saturating_sub(loop_end);
    let loop_len = loop_end.saturating_sub(loop_start);

    // Heuristic: Prefer single-loop structure with reasonable outro
    // If we have multiple repeats but outro is very short, reconsider
    if num_repeats > 1 && !cfg.use_exhaustive_search {
        let single_loop_end = loop_start + lag;
        let single_outro_len = n.saturating_sub(single_loop_end);
        let min_reasonable_outro = lag / 4; // At least 25% of loop period

        // If single-loop would give a more reasonable outro, prefer that
        if outro_len < min_reasonable_outro && single_outro_len >= min_reasonable_outro {
            loop_end = single_loop_end;
        }
    }

    // If outro > 1.5x loop length, something is wrong - try to extend loop_end
    if outro_len > loop_len + (loop_len / 2) && num_repeats == 1 && !cfg.use_exhaustive_search {
        // Maybe we detected only one repeat but pattern continues
        // Try to find if there's another repeat we missed
        let (extended_end, extended_repeats) = find_repetition_end(
            ssm,
            loop_start,
            lag,
            cfg.repetition_similarity_threshold * 0.75, // Lower threshold for retry
        );
        if extended_repeats > num_repeats {
            let extended_outro = n.saturating_sub(extended_end);
            let min_acceptable_outro = lag / 5; // At least 20% of period

            // Only accept extended loop if outro is still reasonable
            if extended_outro >= min_acceptable_outro {
                loop_end = extended_end;
            }
        }
    }

    let intro = make_segment(frames, SegmentKind::Intro, 0, loop_start);
    let loop_seg = make_segment(frames, SegmentKind::Loop, loop_start, loop_end);
    let outro = make_segment(frames, SegmentKind::Outro, loop_end, n);

    Some(SegmentationResult {
        intro,
        loop_seg,
        outro,
        best_lag_frames: lag_est.best_lag,
        best_phase_start: original_phase_start,
        num_loop_repeats: num_repeats,
        loop_confidence: lag_est.confidence.min(phase_confidence),
        section_boundaries: section_frames,
        beat_frames: beat_frames_out,
        tempo_bpm: tempo_bpm_out,
    })
}

/// Detect where the repeating pattern actually ends by checking continued similarity.
///
/// This extends the loop region beyond one period if the pattern continues to repeat.
fn find_repetition_end(
    ssm: &SelfSimilarity,
    loop_start: usize,
    lag: usize,
    min_similarity: f32,
) -> (usize, usize) {
    let n = ssm.n_frames();
    let mut current_end = loop_start + lag;
    let mut num_repeats = 1;

    // Window size for similarity checking (sample 10% of pattern)
    let window_size = (lag / 10).max(5);

    #[cfg(debug_assertions)]
    eprintln!(
        "[DEBUG] Repetition detection: start_frame={}, lag={}, threshold={:.3}",
        loop_start, lag, min_similarity
    );

    loop {
        // Can we fit another full repetition?
        let next_end = current_end + lag;
        if next_end > n {
            #[cfg(debug_assertions)]
            eprintln!(
                "[DEBUG] Cannot fit another repetition: next_end={} > n={}",
                next_end, n
            );
            break;
        }

        // Check similarity between the original pattern and the potential next repeat
        let mut similarity_sum = 0.0f64;
        let mut count = 0u64;

        // Sample points throughout the pattern
        let step = (lag / window_size).max(1);
        for offset in (0..lag).step_by(step) {
            let frame_original = loop_start + offset;
            let frame_candidate = current_end + offset;

            if frame_candidate < n && frame_original < n {
                similarity_sum += ssm.value(frame_original, frame_candidate) as f64;
                count += 1;
            }
        }

        let avg_similarity = if count > 0 {
            (similarity_sum / count as f64) as f32
        } else {
            0.0
        };

        // Check if accepting this repeat would leave a reasonable outro
        let remaining_after_next = n.saturating_sub(next_end);
        let min_outro = lag / 4; // Outro should be at least 25% of loop period
        let has_reasonable_outro = remaining_after_next >= min_outro;

        #[cfg(debug_assertions)]
        eprintln!(
            "[DEBUG] Checking repeat #{}: similarity={:.4}, threshold={:.3}, outro_after={}, min_outro={}, accept={}",
            num_repeats + 1,
            avg_similarity,
            min_similarity,
            remaining_after_next,
            min_outro,
            avg_similarity >= min_similarity && has_reasonable_outro
        );

        // If similarity is high enough AND outro would be reasonable, accept repeat
        if avg_similarity >= min_similarity && has_reasonable_outro {
            current_end = next_end;
            num_repeats += 1;
        } else {
            break;
        }
    }

    #[cfg(debug_assertions)]
    eprintln!(
        "[DEBUG] Final result: {} repeat(s), end_frame={}",
        num_repeats, current_end
    );

    (current_end, num_repeats)
}

/// Find best matching loop pair by exhaustive search.
///
/// This searches for the pair of segments [start..start+lag] and [start+lag..start+2*lag]
/// that have the highest similarity, ensuring a reasonable outro remains.
fn find_best_matching_loop_pair(
    ssm: &SelfSimilarity,
    lag: usize,
    min_loop_frames: usize,
) -> (usize, usize, usize) {
    let n = ssm.n_frames();

    // Minimum outro should be at least 10% of lag or 50 frames
    let min_outro = (lag / 10).max(50);

    // Search space: start can be from 0 to n - lag - min_outro
    let max_start = n.saturating_sub(lag + min_outro);

    if max_start < min_loop_frames {
        // Fallback: use basic detection
        return (0, lag.min(n), 1);
    }

    let mut best_start = 0;
    let mut best_score = 0.0f32;

    // Sample step: test every Nth frame to balance speed vs accuracy
    let step = (lag / 100).max(10).min(50);

    #[cfg(debug_assertions)]
    eprintln!(
        "[DEBUG] Exhaustive search: testing start positions 0..{} (step {})",
        max_start, step
    );

    // Exhaustive search over possible start positions
    for start in (0..=max_start).step_by(step) {
        let end = start + lag;

        if end + min_outro > n {
            break;
        }

        // Compute average similarity between [start..end] and [end..end+lag]
        let score = compute_segment_pair_similarity(ssm, start, end, lag);

        if score > best_score {
            best_score = score;
            best_start = start;
        }
    }

    // Refine search around best candidate
    let refine_start = best_start.saturating_sub(step);
    let refine_end = (best_start + step).min(max_start);

    for start in refine_start..=refine_end {
        let end = start + lag;

        if end + min_outro > n {
            break;
        }

        let score = compute_segment_pair_similarity(ssm, start, end, lag);

        if score > best_score {
            best_score = score;
            best_start = start;
        }
    }

    let loop_end = best_start + lag;

    #[cfg(debug_assertions)]
    eprintln!(
        "[DEBUG] Exhaustive search result: start_frame={}, end_frame={}, score={:.4}",
        best_start, loop_end, best_score
    );

    // Check if pattern repeats after first occurrence
    let remaining = n.saturating_sub(loop_end);
    let num_repeats = if remaining >= lag {
        // Could potentially repeat, but default to 1 for exhaustive search
        // (we found the best single match)
        1
    } else {
        1
    };

    (best_start, loop_end, num_repeats)
}

/// Compute similarity score between two segments.
fn compute_segment_pair_similarity(
    ssm: &SelfSimilarity,
    start: usize,
    end: usize,
    lag: usize,
) -> f32 {
    let n = ssm.n_frames();
    let mut sum = 0.0f64;
    let mut count = 0u64;

    // Sample points within the loop region (every Nth frame)
    let sample_step = (lag / 50).max(1);

    for offset in (0..lag).step_by(sample_step) {
        let frame_a = start + offset;
        let frame_b = end + offset;

        if frame_a < n && frame_b < n {
            sum += ssm.value(frame_a, frame_b) as f64;
            count += 1;
        }
    }

    if count > 0 {
        (sum / count as f64) as f32
    } else {
        0.0
    }
}

fn make_segment(
    frames: &Frames,
    kind: SegmentKind,
    start_frame: usize,
    end_frame: usize,
) -> Segment {
    let hop = frames.hop_size();
    let sr = frames.sample_rate() as f64;

    let start_sample = start_frame * hop;
    let end_sample = end_frame * hop;

    let start_seconds = start_sample as f64 / sr;
    let end_seconds = end_sample as f64 / sr;

    Segment {
        kind,
        start_frame,
        end_frame,
        start_sample,
        end_sample,
        start_seconds,
        end_seconds,
    }
}

fn snap_to_nearest(points: &[usize], x: usize, radius: usize) -> Option<usize> {
    let mut best: Option<(usize, usize)> = None; // (dist, val)
    for &p in points {
        let dist = if p > x { p - x } else { x - p };
        if dist <= radius {
            match best {
                None => best = Some((dist, p)),
                Some((bd, _)) if dist < bd => best = Some((dist, p)),
                _ => {}
            }
        }
    }
    best.map(|(_, p)| p)
}
