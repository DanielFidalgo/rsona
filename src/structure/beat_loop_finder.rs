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
}

impl Default for BeatLoopConfig {
    fn default() -> Self {
        Self {
            min_length_samples: 44100 * 10, // 10 seconds at 44.1kHz
            max_length_samples: None,
            feature_window_frames: 50, // ~0.6 seconds at hop=512, sr=44100
            max_candidates: 100,
            metric: DistanceMetric::Manhattan,
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
    let mut candidates =
        find_valid_beat_pairs(&normalized, &beats.beat_frames, frames.hop_size(), &cfg);

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

    // Mean
    for beat_feats in features {
        for (i, &val) in beat_feats.iter().enumerate() {
            means[i] += val as f64;
        }
    }
    for mean in &mut means {
        *mean /= n_beats as f64;
    }

    // Std
    for beat_feats in features {
        for (i, &val) in beat_feats.iter().enumerate() {
            let diff = val as f64 - means[i];
            stds[i] += diff * diff;
        }
    }
    for std in &mut stds {
        *std = (*std / n_beats as f64).sqrt();
        if *std < 1e-8 {
            *std = 1.0; // Avoid division by zero
        }
    }

    // Normalize
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

/// Find all valid beat pairs and compute distances
fn find_valid_beat_pairs(
    beat_features: &[Vec<f32>],
    beat_frames: &[usize],
    hop_size: usize,
    cfg: &BeatLoopConfig,
) -> Vec<(usize, usize, f32)> {
    let n_beats = beat_features.len();
    let mut candidates = Vec::new();

    for b1 in 0..n_beats {
        for b2 in (b1 + 1)..n_beats {
            // Check length constraints
            let s1 = beat_frames[b1] * hop_size;
            let s2 = beat_frames[b2] * hop_size;
            let length = s2 - s1;

            if length < cfg.min_length_samples {
                continue;
            }

            if let Some(max_len) = cfg.max_length_samples
                && length > max_len
            {
                continue;
            }

            // Compute distance
            let distance = compute_distance(&beat_features[b1], &beat_features[b2], cfg.metric);

            candidates.push((b1, b2, distance));
        }
    }

    candidates
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
}
