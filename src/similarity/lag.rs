//! Diagonal energy / lag histogram extraction from a self-similarity matrix.
//!
//! This is a key primitive for repetition and loop-period estimation.
//!
//! For each lag `L > 0`, we aggregate S[i, i+L] over i, optionally ignoring
//! very small lags and optionally normalizing by the number of terms.

use super::SelfSimilarity;

/// How to aggregate similarity along a diagonal (lag).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LagAggregation {
    /// Mean of diagonal entries (recommended).
    Mean,
    /// Sum of diagonal entries (biased toward shorter available diagonals).
    Sum,
    /// Maximum along the diagonal (more sensitive to single matches).
    Max,
}

/// Configuration for diagonal lag energy computation.
#[derive(Debug, Clone)]
pub struct LagEnergyConfig {
    /// Minimum lag (frames) to consider.
    pub min_lag: usize,
    /// Maximum lag (frames) to consider (inclusive).
    pub max_lag: usize,

    /// Aggregate method along each diagonal.
    pub aggregation: LagAggregation,

    /// If true, subtract the global mean similarity before aggregation.
    /// This helps reduce bias from globally "similar" tracks.
    pub mean_center: bool,

    /// Optional smoothing over lag axis (moving average over lag).
    pub smooth_lags: Option<usize>,
}

impl Default for LagEnergyConfig {
    fn default() -> Self {
        Self {
            min_lag: 20,
            max_lag: 800,
            aggregation: LagAggregation::Mean,
            mean_center: true,
            smooth_lags: Some(7),
        }
    }
}

/// Lag energy results (one score per lag).
#[derive(Debug, Clone)]
pub struct LagEnergy {
    /// min lag included
    pub min_lag: usize,
    /// max lag included
    pub max_lag: usize,
    /// energy[l] corresponds to lag = min_lag + l
    pub energy: Vec<f32>,
}

impl LagEnergy {
    /// Returns the number of lags included.
    #[inline]
    pub fn lag_count(&self) -> usize {
        self.energy.len()
    }

    /// Returns the energy at a given lag.
    #[inline]
    pub fn energy_at_lag(&self, lag: usize) -> Option<f32> {
        if lag < self.min_lag || lag > self.max_lag {
            None
        } else {
            Some(self.energy[lag - self.min_lag])
        }
    }

    /// Returns (best_lag, best_energy) using simple max.
    pub fn argmax(&self) -> Option<(usize, f32)> {
        self.energy
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, &v)| (self.min_lag + i, v))
    }
}

/// Estimate of the strongest repeat lag (loop period) from lag energy.
#[derive(Debug, Clone)]
pub struct RepeatLagEstimate {
    /// Best lag in frames.
    pub best_lag: usize,
    /// Score at best lag.
    pub best_score: f32,
    /// Confidence in [0,1] (heuristic).
    pub confidence: f32,
    /// Full lag energy (for debugging/visualization).
    pub lag_energy: LagEnergy,
}

/// Compute diagonal lag energy (a "lag histogram") from an SSM.
///
/// For lag L: aggregate values S[i, i+L] for i in 0..n-L.
pub fn diagonal_lag_energy(ssm: &SelfSimilarity, cfg: LagEnergyConfig) -> LagEnergy {
    let num_frames = ssm.n_frames();
    if num_frames == 0 {
        return LagEnergy {
            min_lag: cfg.min_lag,
            max_lag: cfg.max_lag,
            energy: Vec::new(),
        };
    }

    let min_lag = cfg.min_lag.min(num_frames.saturating_sub(1));
    let max_lag = cfg.max_lag.min(num_frames.saturating_sub(1));
    if max_lag < min_lag {
        return LagEnergy {
            min_lag,
            max_lag,
            energy: Vec::new(),
        };
    }

    // Optional mean-centering: compute global mean similarity over the considered lags.
    let global_mean = if cfg.mean_center {
        let mut sum = 0.0f64;
        let mut cnt = 0u64;
        for lag in min_lag..=max_lag {
            for frame_idx in 0..(num_frames - lag) {
                sum += ssm.value(frame_idx, frame_idx + lag) as f64;
                cnt += 1;
            }
        }
        if cnt > 0 {
            (sum / cnt as f64) as f32
        } else {
            0.0
        }
    } else {
        0.0
    };

    let mut energy = Vec::with_capacity(max_lag - min_lag + 1);

    for lag in min_lag..=max_lag {
        let mut acc = match cfg.aggregation {
            LagAggregation::Mean | LagAggregation::Sum => 0.0f64,
            LagAggregation::Max => f64::NEG_INFINITY,
        };
        let mut cnt = 0u64;

        for frame_idx in 0..(num_frames - lag) {
            let mut value = ssm.value(frame_idx, frame_idx + lag) - global_mean;
            // after mean-centering, negatives can happen; clamp for stability
            if value < 0.0 {
                value = 0.0;
            }

            match cfg.aggregation {
                LagAggregation::Mean | LagAggregation::Sum => {
                    acc += value as f64;
                    cnt += 1;
                }
                LagAggregation::Max => {
                    if (value as f64) > acc {
                        acc = value as f64;
                    }
                }
            }
        }

        let energy_value = match cfg.aggregation {
            LagAggregation::Sum => acc as f32,
            LagAggregation::Mean => {
                if cnt > 0 {
                    (acc / cnt as f64) as f32
                } else {
                    0.0
                }
            }
            LagAggregation::Max => {
                if acc.is_finite() {
                    acc as f32
                } else {
                    0.0
                }
            }
        };

        energy.push(energy_value);
    }

    if let Some(window_size) = cfg.smooth_lags
        && window_size > 1
        && !energy.is_empty()
    {
        energy = moving_average(&energy, window_size);
    }

    LagEnergy {
        min_lag,
        max_lag,
        energy,
    }
}

/// Convenience: estimate strongest repeat lag + heuristic confidence.
///
/// Confidence is computed as:
///   (best - median) / (best + 1e-9), clamped to [0,1]
/// This is a simple robustness measure (not probabilistic).
pub fn estimate_repeat_lag(
    ssm: &SelfSimilarity,
    cfg: LagEnergyConfig,
) -> Option<RepeatLagEstimate> {
    let le = diagonal_lag_energy(ssm, cfg);
    let (best_lag, best_score) = le.argmax()?;

    if le.energy.is_empty() {
        return None;
    }

    let mut sorted = le.energy.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];

    let conf = ((best_score - median) / (best_score + 1e-9)).clamp(0.0, 1.0);

    Some(RepeatLagEstimate {
        best_lag,
        best_score,
        confidence: conf,
        lag_energy: le,
    })
}

fn moving_average(x: &[f32], window_size: usize) -> Vec<f32> {
    let num_samples = x.len();
    if window_size <= 1 || num_samples == 0 {
        return x.to_vec();
    }
    let half = window_size / 2;
    let mut out = vec![0.0f32; num_samples];

    for i in 0..num_samples {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(num_samples);

        let mut sum = 0.0f32;
        for value in &x[start..end] {
            sum += *value;
        }
        out[i] = sum / (end - start) as f32;
    }

    out
}

/// Configuration for repeat phase (loop start) estimation.
#[derive(Debug, Clone)]
pub struct RepeatPhaseConfig {
    /// Window length (frames) over which to average diagonal energy.
    ///
    /// Larger values favor stability; smaller values favor precision.
    pub window: usize,

    /// Optional smoothing over phase scores (frames).
    pub smooth: Option<usize>,
}

impl Default for RepeatPhaseConfig {
    fn default() -> Self {
        Self {
            window: 8,
            smooth: Some(5),
        }
    }
}

/// Best repeat phase estimate for a given lag.
#[derive(Debug, Clone)]
pub struct RepeatPhaseEstimate {
    /// Start frame of the repeating segment.
    pub start_frame: usize,
    /// End frame (= start_frame + lag).
    pub end_frame: usize,
    /// Raw score at best phase.
    pub score: f32,
    /// Confidence in [0,1] (heuristic).
    pub confidence: f32,
    /// Per-phase scores (for debugging/visualization).
    pub phase_scores: Vec<f32>,
}

/// Find the best repeat phase (loop start) for a given lag.
///
/// This scans possible start frames `i` and scores how well
/// frames `[i .. i+window)` match `[i+lag .. i+lag+window)`.
pub fn best_repeat_phase(
    ssm: &SelfSimilarity,
    lag: usize,
    cfg: RepeatPhaseConfig,
) -> Option<RepeatPhaseEstimate> {
    let num_frames = ssm.n_frames();
    if lag == 0 || lag >= num_frames {
        return None;
    }

    let window = cfg.window.max(1);
    let max_start = num_frames.saturating_sub(lag + window);
    if max_start == 0 {
        return None;
    }

    let mut scores = vec![0.0f32; max_start + 1];

    for frame_idx in 0..=max_start {
        let mut acc = 0.0f64;
        let mut cnt = 0u64;

        for offset in 0..window {
            let frame_a = frame_idx + offset;
            let frame_b = frame_a + lag;
            if frame_b < num_frames {
                acc += ssm.value(frame_a, frame_b) as f64;
                cnt += 1;
            }
        }

        scores[frame_idx] = if cnt > 0 {
            (acc / cnt as f64) as f32
        } else {
            0.0
        };
    }

    // Optional smoothing over phase axis
    let scores = if let Some(window_size) = cfg.smooth {
        if window_size > 1 {
            moving_average(&scores, window_size)
        } else {
            scores
        }
    } else {
        scores
    };

    // Pick best phase
    let (best_i, best_score) = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())?;

    // Confidence heuristic: contrast against median
    let mut sorted = scores.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];

    let confidence = ((best_score - median) / (best_score + 1e-9)).clamp(0.0, 1.0);

    Some(RepeatPhaseEstimate {
        start_frame: best_i,
        end_frame: best_i + lag,
        score: *best_score,
        confidence,
        phase_scores: scores,
    })
}
