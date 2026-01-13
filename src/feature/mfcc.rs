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
    /// If > 0, apply liftering (cepstral lifter).
    pub lifter: usize,
}

impl Default for MfccConfig {
    fn default() -> Self {
        Self {
            n_mfcc: 20,
            dct_norm: DctNorm::Ortho,
            log_floor: 1e-10,
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

    // Precompute DCT-II matrix: (n_mfcc × n_mels).
    let dct = dct2_matrix(n_mfcc, n_mels, cfg.dct_norm);

    // Output frames × n_mfcc.
    let mut out = vec![0.0f32; n_frames * n_mfcc];

    // Parallelize for large frame counts
    if n_frames > 10 {
        // Parallel path - pre-allocate output and write directly to avoid cloning
        out.par_chunks_mut(n_mfcc)
            .enumerate()
            .for_each(|(t, out_row)| {
                let x = mel.frame(t).expect("n_frames mismatch");

                MFCC_BUFFERS.with(|buffers| {
                    let mut buffers = buffers.borrow_mut();
                    let (logm, _) = &mut *buffers;

                    // Ensure buffer is properly sized
                    logm.resize(n_mels, 0.0);

                    // log-mel vector (reusing buffer)
                    for m in 0..n_mels {
                        let v = x[m].max(cfg.log_floor);
                        logm[m] = v.ln();
                    }

                    // DCT: y[k] = sum_m dct[k,m] * logm[m]
                    for k in 0..n_mfcc {
                        let row = &dct[k * n_mels..(k + 1) * n_mels];
                        let mut acc = 0.0f32;
                        for m in 0..n_mels {
                            acc += row[m] * logm[m];
                        }
                        out_row[k] = acc;
                    }
                });
            });
    } else {
        // Sequential path for small frame counts - reuse logm buffer
        let mut logm = vec![0.0f32; n_mels]; // Reuse buffer across frames

        for t in 0..n_frames {
            let x = mel.frame(t).expect("n_frames mismatch");

            // log-mel vector (reusing buffer)
            for m in 0..n_mels {
                let v = x[m].max(cfg.log_floor);
                logm[m] = v.ln();
            }

            // DCT: y[k] = sum_m dct[k,m] * logm[m]
            let out_row = &mut out[t * n_mfcc..(t + 1) * n_mfcc];
            for k in 0..n_mfcc {
                let row = &dct[k * n_mels..(k + 1) * n_mels];
                let mut acc = 0.0f32;
                for m in 0..n_mels {
                    acc += row[m] * logm[m];
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
