//! Novelty curve extraction from a self-similarity matrix (Foote-style).
//!
//! Uses a checkerboard kernel along the diagonal to detect section boundaries.

use super::SelfSimilarity;

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
    let n = ssm.n_frames();
    let w = match cfg.window {
        KernelWindow::Frames(w) => w.max(1),
    };

    // Precompute weights and signs for kernel (2w x 2w) with center between quadrants.
    let size = 2 * w;
    let weights = build_weights(size, w, cfg.gaussian, cfg.gaussian_sigma_frac);
    let signs = build_checkerboard_signs(size, w);

    let mut out = vec![0.0f32; n];

    // For each diagonal position k, apply kernel to local block around (k,k).
    for k in 0..n {
        // kernel block spans rows [k-w, k+w) and cols [k-w, k+w)
        let r0 = k.saturating_sub(w);
        let c0 = k.saturating_sub(w);
        let r1 = (k + w).min(n);
        let c1 = (k + w).min(n);

        let mut acc = 0.0f32;
        let mut energy = 0.0f32;

        for (rr, r) in (r0..r1).enumerate() {
            for (cc, c) in (c0..c1).enumerate() {
                // band gate if needed
                if let Some(band) = cfg.band {
                    let dist = r.abs_diff(c);
                    if dist > band {
                        continue;
                    }
                }

                let s = ssm.value(r, c);
                let wgt = weights[rr * size + cc];
                let sgn = signs[rr * size + cc];

                acc += s * wgt * sgn;
                if cfg.normalize {
                    energy += wgt * wgt;
                }
            }
        }

        if cfg.normalize && energy > 1e-12 {
            acc /= energy.sqrt();
        }

        // Novelty should be non-negative for boundary strength;
        // taking abs is standard because sign depends on direction.
        out[k] = acc.abs();
    }

    if let Some(win) = cfg.smooth
        && win > 1
    {
        out = moving_average(&out, win);
    }

    NoveltyCurve {
        n_frames: n,
        values: out,
    }
}

fn build_checkerboard_signs(size: usize, w: usize) -> Vec<f32> {
    let mut s = vec![0.0f32; size * size];
    for r in 0..size {
        for c in 0..size {
            // quadrants relative to w
            let top = r < w;
            let left = c < w;
            let sign = match (top, left) {
                (true, true) => 1.0,   // UL
                (false, false) => 1.0, // LR
                (true, false) => -1.0, // UR
                (false, true) => -1.0, // LL
            };
            s[r * size + c] = sign;
        }
    }
    s
}

fn build_weights(size: usize, w: usize, gaussian: bool, sigma_frac: f32) -> Vec<f32> {
    let mut out = vec![1.0f32; size * size];
    if !gaussian {
        return out;
    }

    let sigma = (w as f32 * sigma_frac).max(1e-6);
    // center at kernel midpoint (w-0.5), but we can approximate by w as int center
    let cx = (size as f32 - 1.0) / 2.0;
    let cy = cx;

    for r in 0..size {
        for c in 0..size {
            let dx = r as f32 - cx;
            let dy = c as f32 - cy;
            let g = (-0.5 * (dx * dx + dy * dy) / (sigma * sigma)).exp();
            out[r * size + c] = g;
        }
    }
    out
}

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
