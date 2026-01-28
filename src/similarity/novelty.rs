//! Novelty curve extraction from a self-similarity matrix (Foote-style).
//!
//! Uses a checkerboard kernel along the diagonal to detect section boundaries.

use super::SelfSimilarity;
use rayon::prelude::*;

/// Kernel window size (half-width in frames).
///
/// Total kernel size is (2w x 2w).
#[derive(Debug, Clone, Copy)]
pub enum KernelWindow {
    /// Half-width in frames.
    Frames(usize),
}

/// Novelty curve configuration.
#[derive(Debug, Clone)]
pub struct NoveltyConfig {
    /// Kernel half-width.
    pub window: KernelWindow,

    /// Optional diagonal band limit (in frames).
    ///
    /// If set, we only read S(i,j) when |i-j| <= band.
    /// Useful if your SSM was computed with max_lag.
    pub band: Option<usize>,

    /// Apply Gaussian weighting inside the kernel.
    pub gaussian: bool,

    /// Gaussian sigma as fraction of window (e.g. 0.5).
    pub gaussian_sigma_frac: f32,

    /// Normalize novelty by kernel energy (scale stability).
    pub normalize: bool,

    /// Optional moving-average smoothing (frames).
    pub smooth: Option<usize>,
}

impl Default for NoveltyConfig {
    fn default() -> Self {
        Self {
            window: KernelWindow::Frames(32),
            band: None,
            gaussian: true,
            gaussian_sigma_frac: 0.5,
            normalize: true,
            smooth: Some(5),
        }
    }
}

/// Novelty curve (one value per frame).
#[derive(Debug, Clone)]
pub struct NoveltyCurve {
    n_frames: usize,
    values: Vec<f32>,
}

impl NoveltyCurve {
    /// Number of frames.
    #[inline]
    pub fn n_frames(&self) -> usize {
        self.n_frames
    }

    /// Novelty values.
    #[inline]
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// Compute novelty curve from a self-similarity matrix.
///
/// Foote novelty: convolve a checkerboard kernel along the diagonal.
/// For each frame k:
///   novelty[k] = sum_{i,j in [-w..w)}  S[k+i, k+j] * sign(i,j) * weight(i,j)
///
/// Checkerboard sign:
///   (+) for UL and LR quadrants, (-) for UR and LL.
pub fn novelty_curve(ssm: &SelfSimilarity, cfg: NoveltyConfig) -> NoveltyCurve {
    let num_frames = ssm.n_frames();
    let half_width = match cfg.window {
        KernelWindow::Frames(half_width) => half_width.max(1),
    };

    // Precompute weights and signs for kernel (2*half_width x 2*half_width) with center between quadrants.
    let size = 2 * half_width;
    let weights = build_weights(size, half_width, cfg.gaussian, cfg.gaussian_sigma_frac);
    let signs = build_checkerboard_signs(size, half_width);

    // Parallelize novelty computation for better performance on large matrices
    let mut out: Vec<f32> = if num_frames > 100 {
        // Parallel path for large matrices
        (0..num_frames)
            .into_par_iter()
            .map(|frame_idx| {
                // kernel block spans rows [frame_idx-half_width, frame_idx+half_width) and cols [frame_idx-half_width, frame_idx+half_width)
                let row_start = frame_idx.saturating_sub(half_width);
                let col_start = frame_idx.saturating_sub(half_width);
                let row_end = (frame_idx + half_width).min(num_frames);
                let col_end = (frame_idx + half_width).min(num_frames);

                let mut accumulator = 0.0f32;
                let mut energy = 0.0f32;

                for (row_offset, row) in (row_start..row_end).enumerate() {
                    for (col_offset, col) in (col_start..col_end).enumerate() {
                        // band gate if needed
                        if let Some(band) = cfg.band {
                            let dist = row.abs_diff(col);
                            if dist > band {
                                continue;
                            }
                        }

                        let similarity = ssm.value(row, col);
                        let weight = weights[row_offset * size + col_offset];
                        let sign = signs[row_offset * size + col_offset];

                        accumulator += similarity * weight * sign;
                        if cfg.normalize {
                            energy += weight * weight;
                        }
                    }
                }

                if cfg.normalize && energy > 1e-12 {
                    accumulator /= energy.sqrt();
                }

                // Novelty should be non-negative for boundary strength;
                // taking abs is standard because sign depends on direction.
                accumulator.abs()
            })
            .collect()
    } else {
        // Sequential path for small matrices to avoid parallelization overhead
        let mut out = vec![0.0f32; num_frames];

        for frame_idx in 0..num_frames {
            // kernel block spans rows [frame_idx-half_width, frame_idx+half_width) and cols [frame_idx-half_width, frame_idx+half_width)
            let row_start = frame_idx.saturating_sub(half_width);
            let col_start = frame_idx.saturating_sub(half_width);
            let row_end = (frame_idx + half_width).min(num_frames);
            let col_end = (frame_idx + half_width).min(num_frames);

            let mut accumulator = 0.0f32;
            let mut energy = 0.0f32;

            for (row_offset, row) in (row_start..row_end).enumerate() {
                for (col_offset, col) in (col_start..col_end).enumerate() {
                    // band gate if needed
                    if let Some(band) = cfg.band {
                        let dist = row.abs_diff(col);
                        if dist > band {
                            continue;
                        }
                    }

                    let similarity = ssm.value(row, col);
                    let weight = weights[row_offset * size + col_offset];
                    let sign = signs[row_offset * size + col_offset];

                    accumulator += similarity * weight * sign;
                    if cfg.normalize {
                        energy += weight * weight;
                    }
                }
            }

            if cfg.normalize && energy > 1e-12 {
                accumulator /= energy.sqrt();
            }

            // Novelty should be non-negative for boundary strength;
            // taking abs is standard because sign depends on direction.
            out[frame_idx] = accumulator.abs();
        }

        out
    };

    if let Some(window_size) = cfg.smooth
        && window_size > 1
    {
        out = moving_average(&out, window_size);
    }

    NoveltyCurve {
        n_frames: num_frames,
        values: out,
    }
}

fn build_checkerboard_signs(size: usize, half_width: usize) -> Vec<f32> {
    let mut signs = vec![0.0f32; size * size];
    for row in 0..size {
        for col in 0..size {
            // quadrants relative to half_width
            let top = row < half_width;
            let left = col < half_width;
            let sign = match (top, left) {
                (true, true) => 1.0,   // UL
                (false, false) => 1.0, // LR
                (true, false) => -1.0, // UR
                (false, true) => -1.0, // LL
            };
            signs[row * size + col] = sign;
        }
    }
    signs
}

fn build_weights(size: usize, half_width: usize, gaussian: bool, sigma_frac: f32) -> Vec<f32> {
    let mut weights = vec![1.0f32; size * size];
    if !gaussian {
        return weights;
    }

    let sigma = (half_width as f32 * sigma_frac).max(1e-6);
    // center at kernel midpoint (half_width-0.5), but we can approximate by half_width as int center
    let center_x = (size as f32 - 1.0) / 2.0;
    let center_y = center_x;

    for row in 0..size {
        for col in 0..size {
            let delta_x = row as f32 - center_x;
            let delta_y = col as f32 - center_y;
            let gaussian_weight =
                (-0.5 * (delta_x * delta_x + delta_y * delta_y) / (sigma * sigma)).exp();
            weights[row * size + col] = gaussian_weight;
        }
    }
    weights
}

fn moving_average(x: &[f32], window_size: usize) -> Vec<f32> {
    let num_samples = x.len();
    if window_size <= 1 || num_samples == 0 {
        return x.to_vec();
    }
    let half = window_size / 2;

    // Parallelize for large arrays
    if num_samples > 1000 {
        (0..num_samples)
            .into_par_iter()
            .map(|i| {
                let start = i.saturating_sub(half);
                let end = (i + half + 1).min(num_samples);

                let sum: f32 = x[start..end].iter().sum();
                sum / (end - start) as f32
            })
            .collect()
    } else {
        // Sequential path for small arrays
        let mut out = vec![0.0f32; num_samples];

        for i in 0..num_samples {
            let start = i.saturating_sub(half);
            let end = (i + half + 1).min(num_samples);

            let sum: f32 = x[start..end].iter().sum();
            out[i] = sum / (end - start) as f32;
        }
        out
    }
}
