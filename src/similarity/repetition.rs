//! Repetition curve extraction from a self-similarity matrix.
//!
//! A repetition curve measures how strongly each frame repeats
//! later in the track.

use super::SelfSimilarity;

/// How repetition is aggregated across future frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepetitionAggregation {
    /// Maximum similarity to any future frame.
    Max,
    /// Mean similarity across future frames.
    Mean,
    /// Sum of similarities across future frames.
    Sum,
}

/// Configuration for repetition curve extraction.
#[derive(Debug, Clone)]
pub struct RepetitionConfig {
    /// Ignore similarities where |i - j| < min_lag_frames.
    ///
    /// Prevents trivial near-diagonal matches.
    pub min_lag_frames: usize,

    /// How to aggregate similarity across future frames.
    pub aggregation: RepetitionAggregation,

    /// Optional smoothing window (frames).
    pub smooth: Option<usize>,
}

impl Default for RepetitionConfig {
    fn default() -> Self {
        Self {
            min_lag_frames: 20,
            aggregation: RepetitionAggregation::Max,
            smooth: Some(7),
        }
    }
}

/// Repetition curve (one value per frame).
#[derive(Debug, Clone)]
pub struct RepetitionCurve {
    n_frames: usize,
    values: Vec<f32>,
}

impl RepetitionCurve {
    /// Number of frames.
    #[inline]
    pub fn n_frames(&self) -> usize {
        self.n_frames
    }

    /// Values.
    #[inline]
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// Compute a repetition curve from a self-similarity matrix.
///
/// For each frame `i`, this aggregates similarity to frames `j > i + min_lag_frames`.
pub fn repetition_curve(ssm: &SelfSimilarity, cfg: RepetitionConfig) -> RepetitionCurve {
    let n = ssm.n_frames();
    let mut values = vec![0.0f32; n];

    for i in 0..n {
        let start_j = i + cfg.min_lag_frames;
        if start_j >= n {
            values[i] = 0.0;
            continue;
        }

        let mut acc = match cfg.aggregation {
            RepetitionAggregation::Max => 0.0,
            RepetitionAggregation::Mean | RepetitionAggregation::Sum => 0.0,
        };
        let mut count = 0usize;

        for j in start_j..n {
            let v = ssm.value(i, j);
            match cfg.aggregation {
                RepetitionAggregation::Max => {
                    if v > acc {
                        acc = v;
                    }
                }
                RepetitionAggregation::Sum | RepetitionAggregation::Mean => {
                    acc += v;
                    count += 1;
                }
            }
        }

        values[i] = match cfg.aggregation {
            RepetitionAggregation::Max => acc,
            RepetitionAggregation::Sum => acc,
            RepetitionAggregation::Mean => {
                if count > 0 {
                    acc / count as f32
                } else {
                    0.0
                }
            }
        };
    }

    // Optional smoothing
    if let Some(win) = cfg.smooth
        && win > 1
    {
        values = moving_average(&values, win);
    }

    RepetitionCurve {
        n_frames: n,
        values,
    }
}

// --- helpers ---

fn moving_average(x: &[f32], win: usize) -> Vec<f32> {
    let n = x.len();
    if win <= 1 || n == 0 {
        return x.to_vec();
    }

    let half = win / 2;
    let mut out = vec![0.0f32; n];

    for i in 0..n {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(n);

        let mut sum = 0.0f32;
        for v in &x[start..end] {
            sum += *v;
        }
        out[i] = sum / (end - start) as f32;
    }

    out
}
