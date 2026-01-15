//! MFCC computation from a mel spectrogram.
//!
//! MFCC pipeline (typical):
//! 1) power mel spectrogram
//! 2) log compression
//! 3) DCT-II along mel axis
//! 4) optional liftering

use crate::spectrum::MelSpectrogram;

use ndarray::{ArrayView2, ShapeBuilder};
use rayon::prelude::*;
use std::cell::RefCell;

// Thread-local buffers to avoid repeated allocation in parallel MFCC computation
thread_local! {
    static MFCC_BUFFERS: RefCell<(Vec<f32>, Vec<f32>)> = RefCell::new((Vec::new(), Vec::new()));
}

/// DCT normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DctNorm {
    /// No normalization.
    None,
    /// Orthonormal normalization (common default in MFCC implementations).
    Ortho,
}

/// MFCC configuration.
#[derive(Debug, Clone)]
pub struct MfccConfig {
    /// Number of MFCC coefficients to return.
    pub n_mfcc: usize,
    /// DCT normalization.
    pub dct_norm: DctNorm,
    /// Small floor value to avoid log(0).
    pub log_floor: f32,
    /// Maximum dB range (clips values more than this many dB below the max).
    pub top_db: f32,
    /// If > 0, apply liftering (cepstral lifter).
    pub lifter: usize,
}

impl Default for MfccConfig {
    fn default() -> Self {
        Self {
            n_mfcc: 20,
            dct_norm: DctNorm::Ortho,
            log_floor: 1e-10,
            top_db: 80.0,
            lifter: 0,
        }
    }
}

/// MFCC result: frames × n_mfcc.
#[derive(Debug, Clone)]
pub struct MfccResult {
    n_frames: usize,
    #[allow(unused)]
    n_mels: usize,
    n_mfcc: usize,
    /// Row-major: frames × n_mfcc.
    data: Vec<f32>,
}

impl MfccResult {
    /// Number of frames.
    #[inline]
    pub fn n_frames(&self) -> usize {
        self.n_frames
    }

    /// Number of MFCC coefficients.
    #[inline]
    pub fn n_mfcc(&self) -> usize {
        self.n_mfcc
    }

    /// View as ndarray shape (n_frames, n_mfcc).
    pub fn as_ndarray(&self) -> ArrayView2<'_, f32> {
        let shape = (self.n_frames, self.n_mfcc).strides((self.n_mfcc, 1));
        ArrayView2::from_shape(shape, &self.data).expect("mfcc storage is contiguous")
    }

    /// View as slice.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

/// Compute MFCCs from a mel spectrogram (power).
///
/// Input is assumed to be **power mel spectrogram** (not dB).
pub fn mfcc(mel: &MelSpectrogram, cfg: MfccConfig) -> MfccResult {
    assert!(cfg.n_mfcc > 0, "n_mfcc must be > 0");

    let n_frames = mel.n_frames();
    let n_mels = mel.n_mels();
    let n_mfcc = cfg.n_mfcc.min(n_mels);

    // Step 1: Convert mel spectrogram to log scale (dB)
    // We need to do this in two passes to apply top_db clipping correctly
    let mut log_mel = vec![0.0f32; n_frames * n_mels];

    // First pass: compute log values
    for t in 0..n_frames {
        let x = mel.frame(t).expect("n_frames mismatch");
        let log_row = &mut log_mel[t * n_mels..(t + 1) * n_mels];

        for m in 0..n_mels {
            let v = x[m].max(cfg.log_floor);
            log_row[m] = 10.0 * v.log10();
        }
    }

    // Apply top_db clipping: ensure no value is more than top_db below the maximum
    if cfg.top_db > 0.0 {
        let max_db = log_mel.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let threshold = max_db - cfg.top_db;

        for val in log_mel.iter_mut() {
            *val = val.max(threshold);
        }
    }

    // Step 2: Apply DCT-II to get MFCCs
    let dct = dct2_matrix(n_mfcc, n_mels, cfg.dct_norm);
    let mut out = vec![0.0f32; n_frames * n_mfcc];

    // Parallelize for large frame counts
    if n_frames > 10 {
        out.par_chunks_mut(n_mfcc)
            .enumerate()
            .for_each(|(t, out_row)| {
                let log_row = &log_mel[t * n_mels..(t + 1) * n_mels];

                // DCT: y[k] = sum_m dct[k,m] * log_row[m]
                for k in 0..n_mfcc {
                    let dct_row = &dct[k * n_mels..(k + 1) * n_mels];
                    let mut acc = 0.0f32;
                    for m in 0..n_mels {
                        acc += dct_row[m] * log_row[m];
                    }
                    out_row[k] = acc;
                }
            });
    } else {
        // Sequential path
        for t in 0..n_frames {
            let log_row = &log_mel[t * n_mels..(t + 1) * n_mels];
            let out_row = &mut out[t * n_mfcc..(t + 1) * n_mfcc];

            // DCT: y[k] = sum_m dct[k,m] * log_row[m]
            for k in 0..n_mfcc {
                let dct_row = &dct[k * n_mels..(k + 1) * n_mels];
                let mut acc = 0.0f32;
                for m in 0..n_mels {
                    acc += dct_row[m] * log_row[m];
                }
                out_row[k] = acc;
            }
        }
    }

    // Optional liftering
    if cfg.lifter > 0 {
        apply_lifter(&mut out, n_frames, n_mfcc, cfg.lifter);
    }

    MfccResult {
        n_frames,
        n_mels,
        n_mfcc,
        data: out,
    }
}

/// Create a DCT-II matrix of shape (n_mfcc, n_mels), row-major.
///
/// DCT-II:
/// X_k = sum_{n=0}^{N-1} x_n * cos(pi/N * (n + 0.5) * k)
///
/// Ortho norm:
/// k=0: sqrt(1/N)
/// k>0: sqrt(2/N)
fn dct2_matrix(n_mfcc: usize, n_mels: usize, norm: DctNorm) -> Vec<f32> {
    let n = n_mels as f32;
    let mut mat = vec![0.0f32; n_mfcc * n_mels];

    for k in 0..n_mfcc {
        let scale = match norm {
            DctNorm::None => 1.0,
            DctNorm::Ortho => {
                if k == 0 {
                    (1.0 / n).sqrt()
                } else {
                    (2.0 / n).sqrt()
                }
            }
        };

        for m in 0..n_mels {
            let angle = std::f32::consts::PI / n * (m as f32 + 0.5) * k as f32;
            mat[k * n_mels + m] = scale * angle.cos();
        }
    }

    mat
}

/// Apply cepstral lifter in-place.
///
/// lifter formula (common):
/// L[n] = 1 + (L/2) * sin(pi * n / L)
fn apply_lifter(data: &mut [f32], n_frames: usize, n_mfcc: usize, lifter: usize) {
    let l = lifter as f32;
    let mut lift = vec![1.0f32; n_mfcc];
    for n in 0..n_mfcc {
        lift[n] = 1.0 + 0.5 * l * (std::f32::consts::PI * n as f32 / l).sin();
    }

    for t in 0..n_frames {
        let row = &mut data[t * n_mfcc..(t + 1) * n_mfcc];
        for n in 0..n_mfcc {
            row[n] *= lift[n];
        }
    }
}
