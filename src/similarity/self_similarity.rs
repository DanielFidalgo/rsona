//! Self-similarity matrix computation.
//!
//! This module computes frame-to-frame similarity using configurable metrics.
//! It is designed to support structure analysis (loops, sections, repetition)
//! without baking in any musical assumptions.

use rayon::prelude::*;

/// Frame-aligned feature vectors (frames × dims).
///
/// Storage is row-major:
/// frame 0: [f0, f1, ...]
/// frame 1: [f0, f1, ...]
#[derive(Debug)]
pub struct FrameFeatures<'a> {
    n_frames: usize,
    n_dims: usize,
    data: &'a [f32],
}

impl<'a> FrameFeatures<'a> {
    /// Create a new FrameFeatures instance.
    pub fn new(n_frames: usize, n_dims: usize, data: &'a [f32]) -> Self {
        assert_eq!(n_frames * n_dims, data.len(), "FrameFeatures size mismatch");
        Self {
            n_frames,
            n_dims,
            data,
        }
    }

    /// Number of frames.
    #[inline]
    pub fn n_frames(&self) -> usize {
        self.n_frames
    }

    /// Number of dimensions.
    #[inline]
    pub fn n_dims(&self) -> usize {
        self.n_dims
    }

    /// Get a reference to a frame.
    #[inline]
    pub fn frame(&self, i: usize) -> &[f32] {
        let start = i * self.n_dims;
        &self.data[start..start + self.n_dims]
    }

    /// Get the raw backing slice.
    #[inline]
    pub fn as_slice(&self) -> &'a [f32] {
        self.data
    }
}

/// Similarity metric between frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimilarityMetric {
    /// Cosine similarity (dot product of unit vectors).
    Cosine,
}

/// Configuration for self-similarity computation.
#[derive(Debug, Clone)]
pub struct SelfSimilarityConfig {
    /// Similarity metric.
    pub metric: SimilarityMetric,

    /// Optional maximum lag (in frames).
    ///
    /// If set, only similarities where |i - j| <= max_lag
    /// will be computed. Others are set to 0.
    pub max_lag: Option<usize>,

    /// Whether to normalize feature vectors to unit norm.
    ///
    /// Strongly recommended for cosine similarity.
    pub normalize: bool,
}

impl Default for SelfSimilarityConfig {
    fn default() -> Self {
        Self {
            metric: SimilarityMetric::Cosine,
            max_lag: None,
            normalize: true,
        }
    }
}

/// Self-similarity matrix.
///
/// Stored row-major: S[i * n_frames + j]
#[derive(Debug, Clone)]
pub struct SelfSimilarity {
    n_frames: usize,
    data: Vec<f32>,
}

impl SelfSimilarity {
    /// Number of frames.
    #[inline]
    pub fn n_frames(&self) -> usize {
        self.n_frames
    }

    /// Value at (i, j).
    #[inline]
    pub fn value(&self, i: usize, j: usize) -> f32 {
        self.data[i * self.n_frames + j]
    }

    /// Get underlying data slice.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

/// Compute a self-similarity matrix for frame-aligned features.
///
/// This computes S[i, j] = similarity(frame_i, frame_j).
///
/// If `max_lag` is set, only values within that diagonal band
/// are computed; others are set to 0.0.
pub fn self_similarity(features: &FrameFeatures<'_>, cfg: SelfSimilarityConfig) -> SelfSimilarity {
    let n = features.n_frames();
    let d = features.n_dims();

    // 1) Optional normalization
    let normed: Vec<f32>;
    let data: &[f32] = if cfg.normalize {
        normed = normalize_frames_parallel(features);
        &normed
    } else {
        features.as_slice()
    };

    // 2) Similarity computation with parallelization
    let mut out = vec![0.0f32; n * n];

    // Use parallelization for large matrices
    if n > 10 {
        out.par_chunks_mut(n).enumerate().for_each(|(i, row_out)| {
            let fi = &data[i * d..(i + 1) * d];

            let j_start = match cfg.max_lag {
                Some(lag) => i.saturating_sub(lag),
                None => 0,
            };
            let j_end = match cfg.max_lag {
                Some(lag) => (i + lag + 1).min(n),
                None => n,
            };

            for j in j_start..j_end {
                let fj = &data[j * d..(j + 1) * d];
                let sim = match cfg.metric {
                    SimilarityMetric::Cosine => dot_optimized(fi, fj),
                };
                row_out[j] = sim;
            }
        });
    } else {
        // Sequential for small matrices to avoid parallelization overhead
        for i in 0..n {
            let fi = &data[i * d..(i + 1) * d];

            let j_start = match cfg.max_lag {
                Some(lag) => i.saturating_sub(lag),
                None => 0,
            };
            let j_end = match cfg.max_lag {
                Some(lag) => (i + lag + 1).min(n),
                None => n,
            };

            for j in j_start..j_end {
                let fj = &data[j * d..(j + 1) * d];
                let sim = match cfg.metric {
                    SimilarityMetric::Cosine => dot_optimized(fi, fj),
                };
                out[i * n + j] = sim;
            }
        }
    }

    SelfSimilarity {
        n_frames: n,
        data: out,
    }
}

// --- helpers ---

/// Optimized parallel normalization
fn normalize_frames_parallel(features: &FrameFeatures<'_>) -> Vec<f32> {
    let n_frames = features.n_frames();
    let n_dims = features.n_dims();
    let data = features.as_slice();

    if n_frames > 10 {
        // Parallel path
        (0..n_frames)
            .into_par_iter()
            .flat_map(|i| {
                let row = &data[i * n_dims..(i + 1) * n_dims];

                // Compute norm with manual unrolling
                let mut norm = 0.0f32;
                let chunks = n_dims / 4;
                let remainder = n_dims % 4;

                for chunk_idx in 0..chunks {
                    let base = chunk_idx * 4;
                    norm += row[base] * row[base];
                    norm += row[base + 1] * row[base + 1];
                    norm += row[base + 2] * row[base + 2];
                    norm += row[base + 3] * row[base + 3];
                }

                for idx in (chunks * 4)..(chunks * 4 + remainder) {
                    norm += row[idx] * row[idx];
                }

                norm = norm.sqrt().max(1e-12);

                // Normalize
                let mut normalized_row = Vec::with_capacity(n_dims);
                for &v in row {
                    normalized_row.push(v / norm);
                }
                normalized_row
            })
            .collect()
    } else {
        // Sequential path
        let mut out = vec![0.0f32; data.len()];

        for i in 0..n_frames {
            let row = &data[i * n_dims..(i + 1) * n_dims];

            let mut norm = 0.0f32;
            let chunks = n_dims / 4;
            let remainder = n_dims % 4;

            for chunk_idx in 0..chunks {
                let base = chunk_idx * 4;
                norm += row[base] * row[base];
                norm += row[base + 1] * row[base + 1];
                norm += row[base + 2] * row[base + 2];
                norm += row[base + 3] * row[base + 3];
            }

            for idx in (chunks * 4)..(chunks * 4 + remainder) {
                norm += row[idx] * row[idx];
            }

            norm = norm.sqrt().max(1e-12);

            let start = i * n_dims;
            let dst = &mut out[start..start + n_dims];
            for k in 0..n_dims {
                dst[k] = row[k] / norm;
            }
        }

        out
    }
}

/// Optimized dot product with manual unrolling for better vectorization
#[inline(always)]
fn dot_optimized(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let len = a.len();
    let mut sum = 0.0f32;

    // Process in chunks of 8 for SIMD-friendly code
    const CHUNK: usize = 8;
    let main_chunks = len / CHUNK;
    let remainder = len % CHUNK;

    // Main loop with manual unrolling
    for chunk_idx in 0..main_chunks {
        let base = chunk_idx * CHUNK;

        // Unrolled accumulation - compiler can vectorize this
        sum += a[base] * b[base];
        sum += a[base + 1] * b[base + 1];
        sum += a[base + 2] * b[base + 2];
        sum += a[base + 3] * b[base + 3];
        sum += a[base + 4] * b[base + 4];
        sum += a[base + 5] * b[base + 5];
        sum += a[base + 6] * b[base + 6];
        sum += a[base + 7] * b[base + 7];
    }

    // Handle remainder
    let remainder_start = main_chunks * CHUNK;
    for i in 0..remainder {
        let idx = remainder_start + i;
        sum += a[idx] * b[idx];
    }

    sum
}
