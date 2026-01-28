//! Beat synchronization utilities.
//!
//! This module provides functions to aggregate frame-level features
//! at beat boundaries, enabling beat-synchronized analysis.

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

/// Aggregation method for beat synchronization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AggregationMethod {
    /// Mean of all frames in the beat interval.
    #[default]
    Mean,
    /// Median of all frames in the beat interval.
    Median,
    /// Maximum value across frames in the beat interval.
    Max,
    /// First frame in the beat interval.
    First,
    /// Last frame in the beat interval.
    Last,
}

/// Configuration for beat synchronization.
#[derive(Debug, Clone)]
pub struct SyncConfig {
    /// Aggregation method to use.
    pub method: AggregationMethod,

    /// If true, pad the last beat interval to include remaining frames.
    pub pad_end: bool,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            method: AggregationMethod::Mean,
            pad_end: true,
        }
    }
}

/// Synchronize frame-level features to beat times.
///
/// # Arguments
/// * `features` - Frame-level features (frames × features)
/// * `beat_frames` - Beat positions in frame indices
/// * `cfg` - Synchronization configuration
///
/// # Returns
/// Beat-synchronized features (beats × features)
pub fn sync_to_beats(
    features: ArrayView2<'_, f32>,
    beat_frames: &[usize],
    cfg: SyncConfig,
) -> Array2<f32> {
    let (n_frames, n_features) = features.dim();

    if beat_frames.is_empty() {
        return Array2::zeros((0, n_features));
    }

    let n_beats = beat_frames.len();
    let mut output = Array2::zeros((n_beats, n_features));

    // Parallelize for large beat counts
    if n_beats > 50 {
        let rows: Vec<Vec<f32>> = (0..n_beats)
            .into_par_iter()
            .map(|beat_idx| {
                let beat_frame = beat_frames[beat_idx];

                // Determine the frame range for this beat
                let start_frame = if beat_idx == 0 {
                    0
                } else {
                    // Midpoint between previous and current beat
                    (beat_frames[beat_idx - 1] + beat_frame) / 2
                };

                let end_frame = if beat_idx + 1 < n_beats {
                    // Midpoint between current and next beat
                    (beat_frame + beat_frames[beat_idx + 1]) / 2
                } else if cfg.pad_end {
                    // Include all remaining frames
                    n_frames
                } else {
                    // Only up to current beat
                    beat_frame + 1
                };

                // Clamp to valid range
                let start = start_frame.min(n_frames);
                let end = end_frame.min(n_frames).max(start);

                if start >= end {
                    return vec![0.0; n_features];
                }

                // Aggregate features in this range
                (0..n_features)
                    .map(|feat_idx| {
                        aggregate_feature(features.slice(s![start..end, feat_idx]), cfg.method)
                    })
                    .collect()
            })
            .collect();

        // Copy results into output array
        for (beat_idx, row) in rows.iter().enumerate() {
            for (feat_idx, &value) in row.iter().enumerate() {
                output[[beat_idx, feat_idx]] = value;
            }
        }
    } else {
        // Sequential path for small beat counts
        for (beat_idx, &beat_frame) in beat_frames.iter().enumerate() {
            // Determine the frame range for this beat
            let start_frame = if beat_idx == 0 {
                0
            } else {
                // Midpoint between previous and current beat
                (beat_frames[beat_idx - 1] + beat_frame) / 2
            };

            let end_frame = if beat_idx + 1 < n_beats {
                // Midpoint between current and next beat
                (beat_frame + beat_frames[beat_idx + 1]) / 2
            } else if cfg.pad_end {
                // Include all remaining frames
                n_frames
            } else {
                // Only up to current beat
                beat_frame + 1
            };

            // Clamp to valid range
            let start = start_frame.min(n_frames);
            let end = end_frame.min(n_frames).max(start);

            if start >= end {
                continue;
            }

            // Aggregate features in this range
            for feat_idx in 0..n_features {
                let value = aggregate_feature(features.slice(s![start..end, feat_idx]), cfg.method);
                output[[beat_idx, feat_idx]] = value;
            }
        }
    }

    output
}

/// Synchronize a 1D time series to beat times.
///
/// # Arguments
/// * `series` - Frame-level time series
/// * `beat_frames` - Beat positions in frame indices
/// * `cfg` - Synchronization configuration
///
/// # Returns
/// Beat-synchronized series
pub fn sync_series_to_beats(series: &[f32], beat_frames: &[usize], cfg: SyncConfig) -> Vec<f32> {
    let n_frames = series.len();

    if beat_frames.is_empty() {
        return Vec::new();
    }

    let n_beats = beat_frames.len();

    // Parallelize for large beat counts
    if n_beats > 50 {
        (0..n_beats)
            .into_par_iter()
            .map(|beat_idx| {
                let beat_frame = beat_frames[beat_idx];

                // Determine the frame range for this beat
                let start_frame = if beat_idx == 0 {
                    0
                } else {
                    (beat_frames[beat_idx - 1] + beat_frame) / 2
                };

                let end_frame = if beat_idx + 1 < n_beats {
                    (beat_frame + beat_frames[beat_idx + 1]) / 2
                } else if cfg.pad_end {
                    n_frames
                } else {
                    beat_frame + 1
                };

                let start = start_frame.min(n_frames);
                let end = end_frame.min(n_frames).max(start);

                if start >= end {
                    return 0.0;
                }

                match cfg.method {
                    AggregationMethod::Mean => {
                        let sum: f32 = series[start..end].iter().sum();
                        sum / (end - start) as f32
                    }
                    AggregationMethod::Median => {
                        let mut segment: Vec<f32> = series[start..end].to_vec();
                        median(&mut segment)
                    }
                    AggregationMethod::Max => series[start..end]
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max),
                    AggregationMethod::First => series[start],
                    AggregationMethod::Last => series[end - 1],
                }
            })
            .collect()
    } else {
        // Sequential path for small beat counts
        let mut output = Vec::with_capacity(n_beats);

        for (beat_idx, &beat_frame) in beat_frames.iter().enumerate() {
            // Determine the frame range for this beat
            let start_frame = if beat_idx == 0 {
                0
            } else {
                (beat_frames[beat_idx - 1] + beat_frame) / 2
            };

            let end_frame = if beat_idx + 1 < n_beats {
                (beat_frame + beat_frames[beat_idx + 1]) / 2
            } else if cfg.pad_end {
                n_frames
            } else {
                beat_frame + 1
            };

            let start = start_frame.min(n_frames);
            let end = end_frame.min(n_frames).max(start);

            if start >= end {
                output.push(0.0);
                continue;
            }

            let value = match cfg.method {
                AggregationMethod::Mean => {
                    let sum: f32 = series[start..end].iter().sum();
                    sum / (end - start) as f32
                }
                AggregationMethod::Median => {
                    let mut segment: Vec<f32> = series[start..end].to_vec();
                    median(&mut segment)
                }
                AggregationMethod::Max => series[start..end]
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max),
                AggregationMethod::First => series[start],
                AggregationMethod::Last => series[end - 1],
            };

            output.push(value);
        }

        output
    }
}

/// Synchronize features using explicit beat boundaries.
///
/// This version takes explicit start/end frame indices for each beat interval.
///
/// # Arguments
/// * `features` - Frame-level features (frames × features)
/// * `beat_intervals` - List of (start_frame, end_frame) tuples
/// * `method` - Aggregation method
///
/// # Returns
/// Beat-synchronized features (beats × features)
pub fn sync_to_intervals(
    features: ArrayView2<'_, f32>,
    beat_intervals: &[(usize, usize)],
    method: AggregationMethod,
) -> Array2<f32> {
    let (n_frames, n_features) = features.dim();
    let n_beats = beat_intervals.len();

    let mut output = Array2::zeros((n_beats, n_features));

    // Parallelize for large beat counts
    if n_beats > 50 {
        let rows: Vec<Vec<f32>> = beat_intervals
            .par_iter()
            .map(|&(start, end)| {
                let start = start.min(n_frames);
                let end = end.min(n_frames).max(start);

                if start >= end {
                    return vec![0.0; n_features];
                }

                (0..n_features)
                    .map(|feat_idx| {
                        aggregate_feature(features.slice(s![start..end, feat_idx]), method)
                    })
                    .collect()
            })
            .collect();

        // Copy results into output array
        for (beat_idx, row) in rows.iter().enumerate() {
            for (feat_idx, &value) in row.iter().enumerate() {
                output[[beat_idx, feat_idx]] = value;
            }
        }
    } else {
        // Sequential path for small beat counts
        for (beat_idx, &(start, end)) in beat_intervals.iter().enumerate() {
            let start = start.min(n_frames);
            let end = end.min(n_frames).max(start);

            if start >= end {
                continue;
            }

            for feat_idx in 0..n_features {
                let value = aggregate_feature(features.slice(s![start..end, feat_idx]), method);
                output[[beat_idx, feat_idx]] = value;
            }
        }
    }

    output
}

/// Aggregate a feature slice using the specified method.
fn aggregate_feature(values: ndarray::ArrayView1<'_, f32>, method: AggregationMethod) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    match method {
        AggregationMethod::Mean => {
            let sum: f32 = values.iter().sum();
            sum / values.len() as f32
        }
        AggregationMethod::Median => {
            let mut sorted: Vec<f32> = values.to_vec();
            median(&mut sorted)
        }
        AggregationMethod::Max => values.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        AggregationMethod::First => values[0],
        AggregationMethod::Last => values[values.len() - 1],
    }
}

/// Compute median of a mutable slice (sorts in-place).
fn median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;

    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

/// Convert beat times (in seconds) to frame indices.
///
/// # Arguments
/// * `beat_times` - Beat positions in seconds
/// * `sample_rate` - Audio sample rate (Hz)
/// * `hop_size` - Hop size in samples
///
/// # Returns
/// Beat positions as frame indices
pub fn beat_times_to_frames(beat_times: &[f64], sample_rate: u32, hop_size: usize) -> Vec<usize> {
    let sr = sample_rate as f64;
    let hop = hop_size as f64;

    beat_times
        .iter()
        .map(|&time| (time * sr / hop).round() as usize)
        .collect()
}

/// Convert beat frames to times (in seconds).
///
/// # Arguments
/// * `beat_frames` - Beat positions as frame indices
/// * `sample_rate` - Audio sample rate (Hz)
/// * `hop_size` - Hop size in samples
///
/// # Returns
/// Beat positions in seconds
pub fn beat_frames_to_times(beat_frames: &[usize], sample_rate: u32, hop_size: usize) -> Vec<f64> {
    let sr = sample_rate as f64;
    let hop = hop_size as f64;

    beat_frames
        .iter()
        .map(|&frame| frame as f64 * hop / sr)
        .collect()
}

// Import for slice macro in sync functions
use ndarray::s;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_sync_to_beats_basic() {
        // Create simple features: 10 frames × 2 features
        let mut features = Array2::zeros((10, 2));
        for i in 0..10 {
            features[[i, 0]] = i as f32;
            features[[i, 1]] = (i * 2) as f32;
        }

        // Beat at frames 0, 3, 6, 9
        let beat_frames = vec![0, 3, 6, 9];

        let synced = sync_to_beats(features.view(), &beat_frames, SyncConfig::default());

        assert_eq!(synced.dim(), (4, 2));

        // First beat (at frame 0) aggregates frames [0, 1) = just frame 0
        // Mean of [0] = 0.0
        assert!((synced[[0, 0]] - 0.0).abs() < 1e-5);

        // Second beat (at frame 3) aggregates frames [1, 4) = frames 1, 2, 3
        // Mean of [1, 2, 3] = 2.0
        assert!((synced[[1, 0]] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_sync_series_to_beats() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let beat_frames = vec![0, 3];

        let synced = sync_series_to_beats(
            &series,
            &beat_frames,
            SyncConfig {
                method: AggregationMethod::Mean,
                pad_end: true,
            },
        );

        assert_eq!(synced.len(), 2);
    }

    #[test]
    fn test_beat_time_conversion() {
        let beat_times = vec![0.0, 0.5, 1.0];
        let sr = 22050;
        let hop = 512;

        let frames = beat_times_to_frames(&beat_times, sr, hop);
        assert_eq!(frames.len(), 3);

        let times = beat_frames_to_times(&frames, sr, hop);
        assert_eq!(times.len(), 3);

        // Should be approximately equal after round-trip
        // Note: Due to rounding in frame conversion, precision is limited
        for (orig, converted) in beat_times.iter().zip(times.iter()) {
            assert!(
                (orig - converted).abs() < 0.02,
                "Original: {}, Converted: {}, Diff: {}",
                orig,
                converted,
                (orig - converted).abs()
            );
        }
    }

    #[test]
    fn test_aggregation_methods() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Mean
        let series = values.clone();
        let beats = vec![2];
        let synced = sync_series_to_beats(
            &series,
            &beats,
            SyncConfig {
                method: AggregationMethod::Mean,
                pad_end: true,
            },
        );
        assert!((synced[0] - 3.0).abs() < 1e-5); // mean of 1..5

        // Max
        let synced = sync_series_to_beats(
            &series,
            &beats,
            SyncConfig {
                method: AggregationMethod::Max,
                pad_end: true,
            },
        );
        assert!((synced[0] - 5.0).abs() < 1e-5);
    }
}
