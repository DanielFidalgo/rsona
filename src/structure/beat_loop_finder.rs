//! Beat-synchronized loop finder using chroma features.
//!
//! This module implements a loop finding algorithm that:
//! 1. Tracks beats in the audio
//! 2. Extracts features synchronized to each beat
//! 3. Computes pairwise distances between beat features
//! 4. Finds the best matching beat pair as loop points
//!
//! This approach is more accurate than lag-based methods but more computationally expensive.

use crate::{
    feature::{Chromagram, MfccResult, OnsetEnvelope},
    signal::Frames,
    temporal::BeatTrack,
};
use rayon::prelude::*;

/// Result of beat-synchronized loop finding
#[derive(Debug, Clone)]
pub struct BeatLoopResult {
    /// Start beat index
    pub start_beat: usize,
    /// End beat index
    pub end_beat: usize,
    /// Start sample in audio
    pub start_sample: usize,
    /// End sample in audio
    pub end_sample: usize,
    /// Start time in seconds
    pub start_seconds: f64,
    /// End time in seconds
    pub end_seconds: f64,
    /// Loop duration in seconds
    pub duration_seconds: f64,
    /// Distance/similarity score (lower is better for distance metrics)
    pub score: f32,
    /// All candidate loop pairs (start_beat, end_beat, score)
    pub candidates: Vec<(usize, usize, f32)>,
}

/// Loop selection strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Loop detection preference determining how to balance similarity matching vs. musical structure.
///
/// This enum controls the scoring algorithm used to select the best loop candidate:
///
/// # Use Cases
///
/// - **Game developers**: Use `Similarity` for seamless background music loops
/// - **DJs/Producers**: Use `MusicalStructure` for musically meaningful phrase boundaries
/// - **General use**: Use `Balanced` for a compromise between both approaches
///
/// # How It Works
///
/// - `Similarity`: Pure distance-based matching. Finds the tightest similarity match
///   regardless of musical structure (best for seamless loops that may be shorter)
///
/// - `MusicalStructure`: Applies a penalty to loops that don't align with standard
///   musical phrase lengths (8, 16, 32, 64, 128 bars in 4/4 time). Requires tempo
///   to be provided in `BeatLoopConfig::tempo_bpm`. Prefers longer, musically
///   coherent sections even if similarity is slightly lower.
///
/// - `Balanced`: Applies a moderate penalty for non-standard phrase lengths,
///   allowing some flexibility while still preferring musical structure.
///
/// # Examples
///
/// ```rust,ignore
/// use rsona::structure::{BeatLoopConfig, LoopPreference, DistanceMetric};
///
/// // For game background music - tightest match
/// let game_config = BeatLoopConfig {
///     preference: LoopPreference::Similarity,
///     tempo_bpm: None, // Tempo not required for Similarity mode
///     ..Default::default()
/// };
///
/// // For DJ mixing - musical phrase boundaries
/// let dj_config = BeatLoopConfig {
///     preference: LoopPreference::MusicalStructure,
///     tempo_bpm: Some(128.0), // Tempo required for musical structure
///     ..Default::default()
/// };
///
/// // Balanced approach
/// let balanced_config = BeatLoopConfig {
///     preference: LoopPreference::Balanced,
///     tempo_bpm: Some(120.0),
///     ..Default::default()
/// };
/// ```
pub enum LoopPreference {
    /// Optimize for tightest similarity match (best for seamless game loops)
    ///
    /// This mode ignores musical structure and focuses purely on finding the
    /// beat pair with the lowest feature distance. Results in the most seamless
    /// loops but may not align with musical phrase boundaries.
    ///
    /// **Tempo not required** for this mode.
    Similarity,

    /// Optimize for musical phrase boundaries (best for DJ mixing, composition)
    ///
    /// This mode applies a quadratic penalty to loops that don't align with
    /// standard musical phrase lengths (8, 16, 32, 64, 128 bars). Strongly
    /// prefers musically coherent sections that make sense as standalone
    /// musical units.
    ///
    /// **Requires tempo** to be set in `BeatLoopConfig::tempo_bpm`.
    MusicalStructure,

    /// Balance between similarity and musical structure
    ///
    /// This mode applies a linear penalty for non-standard phrase lengths,
    /// providing a middle ground between tight similarity matching and
    /// musical coherence.
    ///
    /// **Tempo recommended** but not strictly required.
    Balanced,
}

impl Default for LoopPreference {
    fn default() -> Self {
        Self::Similarity
    }
}

/// Configuration for beat-synchronized loop finding
#[derive(Debug, Clone)]
pub struct BeatLoopConfig {
    /// Minimum loop duration in samples
    pub min_length_samples: usize,
    /// Maximum loop duration in samples (None = unlimited)
    pub max_length_samples: Option<usize>,
    /// Feature window length in frames after each beat
    pub feature_window_frames: usize,
    /// Maximum number of candidates to return
    pub max_candidates: usize,
    /// Distance metric to use
    pub metric: DistanceMetric,
    /// Length preference weight (0.0 = prioritize similarity only, 0.05 = prefer longer loops)
    ///
    /// This parameter balances between similarity (distance) and loop length.
    /// Score = distance - (length_preference * loop_duration_seconds)
    ///
    /// Recommended values:
    /// - 0.0: Pure similarity (shortest matching loop)
    /// - 0.02-0.05: Balanced (moderate preference for longer loops)
    /// - 0.1+: Strong length preference (may sacrifice similarity)
    pub length_preference: f32,
    /// Loop selection strategy (similarity vs musical structure)
    ///
    /// This determines whether to prefer tight similarity matches or
    /// musically meaningful phrase boundaries (power-of-2 bar counts).
    pub preference: LoopPreference,
    /// Tempo in BPM (required for musical structure mode)
    ///
    /// Used to calculate bar boundaries when preference is MusicalStructure.
    /// Can be obtained from tempo estimation.
    pub tempo_bpm: Option<f32>,
}

impl Default for BeatLoopConfig {
    fn default() -> Self {
        Self {
            min_length_samples: 44100 * 10, // 10 seconds at 44.1kHz
            max_length_samples: None,
            feature_window_frames: 50, // ~0.6 seconds at hop=512, sr=44100
            max_candidates: 100,
            metric: DistanceMetric::Manhattan,
            length_preference: 0.0, // Default: pure similarity, no length preference
            preference: LoopPreference::Similarity,
            tempo_bpm: None,
        }
    }
}

/// Distance metrics for beat feature comparison
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Manhattan (L1) distance
    Manhattan,
    /// Euclidean (L2) distance
    Euclidean,
    /// Cosine distance (1 - cosine similarity)
    Cosine,
}

/// Find loop points using beat-synchronized features
///
/// Implementation approach:
/// 1. Track beats
/// 2. Extract features around each beat
/// 3. Compute pairwise distances
/// 4. Find best matching pair within constraints
pub fn find_loop_by_beats(
    frames: &Frames,
    beats: &BeatTrack,
    chroma: &Chromagram,
    mfcc: &MfccResult,
    onset_env: &OnsetEnvelope,
    cfg: BeatLoopConfig,
) -> Option<BeatLoopResult> {
    let n_beats = beats.beat_frames.len();
    if n_beats < 2 {
        return None;
    }

    let n_frames = chroma.n_frames();

    // 1. Build beat-synchronized features
    let beat_features = extract_beat_features(
        chroma,
        mfcc,
        onset_env,
        &beats.beat_frames,
        cfg.feature_window_frames,
        n_frames,
    );

    if beat_features.is_empty() {
        return None;
    }

    // 2. Normalize features (simple z-score normalization per dimension)
    let normalized = normalize_features(&beat_features);

    // 3. Find all valid loop pairs with their distances
    let mut candidates = find_valid_beat_pairs(
        &normalized,
        &beats.beat_frames,
        frames.hop_size(),
        frames.sample_rate(),
        &cfg,
    );

    if candidates.is_empty() {
        return None;
    }

    // 4. Sort by score (ascending for distance metrics - lower is better)
    candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    // Keep only top candidates
    candidates.truncate(cfg.max_candidates);

    // 5. Best candidate
    let (start_beat, end_beat, score) = candidates[0];

    let start_sample = beats.beat_frames[start_beat] * frames.hop_size();
    let end_sample = beats.beat_frames[end_beat] * frames.hop_size();

    let start_seconds = beats.beat_times[start_beat];
    let end_seconds = beats.beat_times[end_beat];
    let duration_seconds = end_seconds - start_seconds;

    Some(BeatLoopResult {
        start_beat,
        end_beat,
        start_sample,
        end_sample,
        start_seconds,
        end_seconds,
        duration_seconds,
        score,
        candidates,
    })
}

/// Extract features for each beat, aggregating over a window
fn extract_beat_features(
    chroma: &Chromagram,
    mfcc: &MfccResult,
    onset_env: &OnsetEnvelope,
    beat_frames: &[usize],
    window_frames: usize,
    n_frames: usize,
) -> Vec<Vec<f32>> {
    let chroma_arr = chroma.as_ndarray();
    let mfcc_arr = mfcc.as_ndarray();

    let mut beat_features = Vec::new();

    for &beat_frame in beat_frames {
        let end_frame = (beat_frame + window_frames).min(n_frames);

        if end_frame <= beat_frame {
            continue;
        }

        let mut features = Vec::new();

        // Chroma features (average over window)
        for chroma_bin in 0..chroma.n_chroma() {
            let mut sum = 0.0;
            let mut count = 0;
            for frame in beat_frame..end_frame {
                if frame < chroma_arr.nrows() {
                    sum += chroma_arr[[frame, chroma_bin]];
                    count += 1;
                }
            }
            if count > 0 {
                features.push(sum / count as f32);
            }
        }

        // MFCC features (average over window)
        for mfcc_bin in 0..mfcc.n_mfcc() {
            let mut sum = 0.0;
            let mut count = 0;
            for frame in beat_frame..end_frame {
                if frame < mfcc_arr.nrows() {
                    sum += mfcc_arr[[frame, mfcc_bin]];
                    count += 1;
                }
            }
            if count > 0 {
                features.push(sum / count as f32);
            }
        }

        // Onset envelope features (average over window)
        let onset_values = onset_env.values();
        if beat_frame < onset_values.len() {
            let mut sum = 0.0;
            let mut count = 0;
            for frame in beat_frame..end_frame.min(onset_values.len()) {
                sum += onset_values[frame];
                count += 1;
            }
            if count > 0 {
                features.push(sum / count as f32);
            }
        }

        beat_features.push(features);
    }

    beat_features
}

/// Normalize features using z-score normalization per dimension
fn normalize_features(features: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if features.is_empty() {
        return Vec::new();
    }

    let n_beats = features.len();
    let n_dims = features[0].len();

    // Compute mean and std for each dimension
    let mut means = vec![0.0; n_dims];
    let mut stds = vec![0.0; n_dims];

    // Mean - parallelize for large feature sets
    if n_beats > 100 {
        means = (0..n_dims)
            .into_par_iter()
            .map(|dim_idx| {
                features
                    .iter()
                    .map(|feat| feat[dim_idx] as f64)
                    .sum::<f64>()
            })
            .collect();
    } else {
        for beat_feats in features {
            for (i, &val) in beat_feats.iter().enumerate() {
                means[i] += val as f64;
            }
        }
    }
    for mean in &mut means {
        *mean /= n_beats as f64;
    }

    // Std - parallelize for large feature sets
    if n_beats > 100 {
        stds = (0..n_dims)
            .into_par_iter()
            .map(|dim_idx| {
                features
                    .iter()
                    .map(|feat| {
                        let diff = feat[dim_idx] as f64 - means[dim_idx];
                        diff * diff
                    })
                    .sum::<f64>()
            })
            .collect();
    } else {
        for beat_feats in features {
            for (i, &val) in beat_feats.iter().enumerate() {
                let diff = val as f64 - means[i];
                stds[i] += diff * diff;
            }
        }
    }
    for std in &mut stds {
        *std = (*std / n_beats as f64).sqrt();
        if *std < 1e-8 {
            *std = 1.0; // Avoid division by zero
        }
    }

    // Normalize - parallelize for large feature sets
    if n_beats > 100 {
        features
            .par_iter()
            .map(|beat_feats| {
                beat_feats
                    .iter()
                    .enumerate()
                    .map(|(i, &val)| ((val as f64 - means[i]) / stds[i]) as f32)
                    .collect::<Vec<f32>>()
            })
            .collect()
    } else {
        let mut normalized = Vec::with_capacity(n_beats);
        for beat_feats in features {
            let mut norm_feats = Vec::with_capacity(n_dims);
            for (i, &val) in beat_feats.iter().enumerate() {
                let z = ((val as f64 - means[i]) / stds[i]) as f32;
                norm_feats.push(z);
            }
            normalized.push(norm_feats);
        }
        normalized
    }
}

/// Find all valid beat pairs and compute distances with optional length weighting
fn find_valid_beat_pairs(
    beat_features: &[Vec<f32>],
    beat_frames: &[usize],
    hop_size: usize,
    sample_rate: u32,
    cfg: &BeatLoopConfig,
) -> Vec<(usize, usize, f32)> {
    let n_beats = beat_features.len();

    // Parallelize distance computation for large beat counts
    if n_beats > 50 {
        (0..n_beats)
            .into_par_iter()
            .flat_map(|beat1_idx| {
                let mut local_candidates = Vec::new();
                for beat2_idx in (beat1_idx + 1)..n_beats {
                    // Check length constraints
                    let sample1 = beat_frames[beat1_idx] * hop_size;
                    let sample2 = beat_frames[beat2_idx] * hop_size;
                    let length = sample2 - sample1;

                    if length < cfg.min_length_samples {
                        continue;
                    }

                    if let Some(max_len) = cfg.max_length_samples
                        && length > max_len
                    {
                        continue;
                    }

                    // Compute distance
                    let distance = compute_distance(
                        &beat_features[beat1_idx],
                        &beat_features[beat2_idx],
                        cfg.metric,
                    );

                    // Apply length weighting and/or musical structure preference
                    let duration_seconds = length as f32 / (sample_rate as f32 * hop_size as f32);

                    let mut score = distance;

                    // Apply length preference if configured
                    if cfg.length_preference > 0.0 {
                        score -= cfg.length_preference * duration_seconds;
                    }

                    // Apply musical structure bonus if configured
                    if cfg.preference != LoopPreference::Similarity {
                        let structure_adjustment = calculate_musical_structure_score(
                            duration_seconds,
                            cfg.tempo_bpm,
                            cfg.preference,
                        );
                        score += structure_adjustment;
                    }

                    local_candidates.push((beat1_idx, beat2_idx, score));
                }
                local_candidates
            })
            .collect()
    } else {
        let mut candidates = Vec::new();

        for beat1_idx in 0..n_beats {
            for beat2_idx in (beat1_idx + 1)..n_beats {
                // Check length constraints
                let sample1 = beat_frames[beat1_idx] * hop_size;
                let sample2 = beat_frames[beat2_idx] * hop_size;
                let length = sample2 - sample1;

                if length < cfg.min_length_samples {
                    continue;
                }

                if let Some(max_len) = cfg.max_length_samples
                    && length > max_len
                {
                    continue;
                }

                // Compute distance
                let distance = compute_distance(
                    &beat_features[beat1_idx],
                    &beat_features[beat2_idx],
                    cfg.metric,
                );

                // Apply length weighting and/or musical structure preference
                let duration_seconds = length as f32 / (sample_rate as f32 * hop_size as f32);

                let mut score = distance;

                // Apply length preference if configured
                if cfg.length_preference > 0.0 {
                    score -= cfg.length_preference * duration_seconds;
                }

                // Apply musical structure bonus if configured
                if cfg.preference != LoopPreference::Similarity {
                    let structure_adjustment = calculate_musical_structure_score(
                        duration_seconds,
                        cfg.tempo_bpm,
                        cfg.preference,
                    );
                    score += structure_adjustment;
                }

                candidates.push((beat1_idx, beat2_idx, score));
            }
        }

        candidates
    }
}

/// Calculate musical structure score adjustment
///
/// Returns a penalty (positive = worse score) or bonus (negative = better score)
/// based on how well the loop duration aligns with musical phrase boundaries.
///
/// Prefers loops that are power-of-2 bar counts (16, 32, 64 bars) which are
/// common in electronic/dance music structure.
fn calculate_musical_structure_score(
    duration_seconds: f32,
    tempo_bpm: Option<f32>,
    preference: LoopPreference,
) -> f32 {
    // If no tempo provided, can't calculate musical structure
    let tempo = match tempo_bpm {
        Some(t) if t > 0.0 => t,
        _ => return 0.0,
    };

    // Calculate number of beats in this loop duration
    let beats_per_second = tempo / 60.0;
    let num_beats = duration_seconds * beats_per_second;

    // Assuming 4/4 time signature (4 beats per bar)
    let num_bars = num_beats / 4.0;

    // Find nearest power-of-2 bar count that makes musical sense
    // Common loop lengths: 8, 16, 32, 64 bars
    let ideal_bar_counts = [8.0, 16.0, 32.0, 64.0, 128.0];

    let min_distance = ideal_bar_counts
        .iter()
        .map(|&ideal| (num_bars - ideal).abs())
        .fold(f32::INFINITY, f32::min);

    // Calculate penalty based on deviation from ideal
    // The further from a power-of-2 bar count, the higher the penalty
    let penalty = match preference {
        LoopPreference::Similarity => 0.0, // No adjustment
        LoopPreference::MusicalStructure => {
            // Strong preference for musical structure
            // Penalty grows quadratically with distance from ideal
            // Increased from 0.5 to 2.0 to more strongly prefer power-of-2 bar counts
            min_distance * min_distance * 2.0
        }
        LoopPreference::Balanced => {
            // Moderate preference for musical structure
            // Linear penalty
            min_distance * 0.2
        }
    };

    penalty
}

/// Compute distance between two feature vectors
fn compute_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    match metric {
        DistanceMetric::Manhattan => a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum(),
        DistanceMetric::Euclidean => {
            let sum_sq: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
            sum_sq.sqrt()
        }
        DistanceMetric::Cosine => {
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

            if norm_a < 1e-8 || norm_b < 1e-8 {
                1.0 // Maximum distance
            } else {
                1.0 - (dot / (norm_a * norm_b))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manhattan_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dist = compute_distance(&a, &b, DistanceMetric::Manhattan);
        assert!((dist - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dist = compute_distance(&a, &b, DistanceMetric::Euclidean);
        // sqrt(9 + 9 + 9) = sqrt(27) ≈ 5.196
        assert!((dist - 5.196).abs() < 0.01);
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let dist = compute_distance(&a, &b, DistanceMetric::Cosine);
        assert!(dist.abs() < 1e-6); // Same direction = 0 distance

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let dist = compute_distance(&a, &b, DistanceMetric::Cosine);
        assert!((dist - 1.0).abs() < 1e-6); // Orthogonal = 1 distance
    }

    #[test]
    fn test_normalize_features() {
        let features = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let normalized = normalize_features(&features);

        // Check each dimension has mean ~0 and std ~1
        for dim in 0..3 {
            let mean: f32 = normalized.iter().map(|f| f[dim]).sum::<f32>() / 3.0;
            assert!(mean.abs() < 1e-5);
        }
    }

    #[test]
    fn test_musical_structure_score() {
        // Test with 32 bars at 120 BPM (ideal for MusicalStructure mode)
        let duration_32_bars = 64.0; // 32 bars * 4 beats * 0.5 seconds per beat
        let tempo = Some(120.0);

        // Similarity mode should return 0 (no adjustment)
        let score_similarity =
            calculate_musical_structure_score(duration_32_bars, tempo, LoopPreference::Similarity);
        assert_eq!(score_similarity, 0.0);

        // MusicalStructure mode should return 0 for ideal 32-bar loop
        let score_musical = calculate_musical_structure_score(
            duration_32_bars,
            tempo,
            LoopPreference::MusicalStructure,
        );
        assert!(score_musical < 0.1); // Should be very close to 0

        // Balanced mode should also be low for ideal length
        let score_balanced =
            calculate_musical_structure_score(duration_32_bars, tempo, LoopPreference::Balanced);
        assert!(score_balanced < 0.1);

        // Test with non-ideal duration (20 bars) - should have higher penalty
        let duration_20_bars = 40.0; // 20 bars * 4 beats * 0.5 seconds per beat

        let score_musical_bad = calculate_musical_structure_score(
            duration_20_bars,
            tempo,
            LoopPreference::MusicalStructure,
        );
        assert!(score_musical_bad > 1.0); // Should have significant penalty

        // Test without tempo (should return 0)
        let score_no_tempo = calculate_musical_structure_score(
            duration_32_bars,
            None,
            LoopPreference::MusicalStructure,
        );
        assert_eq!(score_no_tempo, 0.0);
    }

    #[test]
    fn test_loop_preference_default() {
        let pref = LoopPreference::default();
        assert!(matches!(pref, LoopPreference::Similarity));
    }
}
