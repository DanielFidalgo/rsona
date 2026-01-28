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
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;
use std::cell::RefCell;

// Thread-local cache for DCT computation
thread_local! {
    static MFCC_BUFFERS: RefCell<(Vec<f32>, Vec<f32>)> = const { RefCell::new((Vec::new(), Vec::new())) };
    static DCT_CACHE: RefCell<DctCache> = RefCell::new(DctCache::new());
}

/// Cache for DCT computation to avoid repeated allocations and FFT planning
struct DctCache {
    planner: FftPlanner<f32>,
    fft_buffer: Vec<Complex<f32>>,
    twiddle_factors: Vec<Complex<f32>>,
    cached_size: usize,
}

impl DctCache {
    fn new() -> Self {
        Self {
            planner: FftPlanner::new(),
            fft_buffer: Vec::new(),
            twiddle_factors: Vec::new(),
            cached_size: 0,
        }
    }

    fn prepare(&mut self, n: usize) {
        if self.cached_size != n {
            self.fft_buffer.resize(n, Complex::new(0.0, 0.0));
            self.twiddle_factors.resize(n, Complex::new(0.0, 0.0));

            // Pre-compute twiddle factors for phase correction
            // e^(-i*pi*k/(2n))
            let factor = -std::f32::consts::PI / (2.0 * n as f32);
            for k in 0..n {
                let angle = factor * k as f32;
                self.twiddle_factors[k] = Complex::new(angle.cos(), angle.sin());
            }

            self.cached_size = n;
        }
    }
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
    let num_mels = mel.n_mels();
    let n_mfcc = cfg.n_mfcc.min(num_mels);

    // Step 1: Convert mel spectrogram to log scale (dB)
    // We need to do this in two passes to apply top_db clipping correctly
    let mut log_mel = vec![0.0f32; n_frames * num_mels];

    // First pass: compute log values (parallelized for better performance)
    if n_frames > 10 {
        log_mel
            .par_chunks_mut(num_mels)
            .enumerate()
            .for_each(|(frame_index, log_row)| {
                let mel_frame = mel.frame(frame_index).expect("n_frames mismatch");
                for mel_idx in 0..num_mels {
                    let value = mel_frame[mel_idx].max(cfg.log_floor);
                    log_row[mel_idx] = 10.0 * value.log10();
                }
            });
    } else {
        for frame_index in 0..n_frames {
            let mel_frame = mel.frame(frame_index).expect("n_frames mismatch");
            let log_row = &mut log_mel[frame_index * num_mels..(frame_index + 1) * num_mels];

            for mel_idx in 0..num_mels {
                let value = mel_frame[mel_idx].max(cfg.log_floor);
                log_row[mel_idx] = 10.0 * value.log10();
            }
        }
    }

    // Apply top_db clipping: ensure no value is more than top_db below the maximum
    if cfg.top_db > 0.0 {
        // Parallelize max finding
        let max_db = log_mel
            .par_iter()
            .cloned()
            .reduce(|| f32::NEG_INFINITY, f32::max);
        let threshold = max_db - cfg.top_db;

        // Parallelize threshold application
        log_mel.par_iter_mut().for_each(|val| {
            *val = val.max(threshold);
        });
    }

    // Step 2: Apply DCT-II to get MFCCs using FFT-based approach
    let mut out = vec![0.0f32; n_frames * n_mfcc];

    // Parallelize for large frame counts
    if n_frames > 10 {
        out.par_chunks_mut(n_mfcc)
            .enumerate()
            .for_each(|(frame_index, out_row)| {
                let log_row = &log_mel[frame_index * num_mels..(frame_index + 1) * num_mels];
                dct2_fft_frame(log_row, out_row, n_mfcc, cfg.dct_norm);
            });
    } else {
        // Sequential path
        for frame_index in 0..n_frames {
            let log_row = &log_mel[frame_index * num_mels..(frame_index + 1) * num_mels];
            let out_row = &mut out[frame_index * n_mfcc..(frame_index + 1) * n_mfcc];
            dct2_fft_frame(log_row, out_row, n_mfcc, cfg.dct_norm);
        }
    }

    // Optional liftering
    if cfg.lifter > 0 {
        apply_lifter(&mut out, n_frames, n_mfcc, cfg.lifter);
    }

    MfccResult {
        n_frames,
        n_mels: num_mels,
        n_mfcc,
        data: out,
    }
}

/// FFT-based DCT-II computation for a single frame.
///
/// This uses an optimized FFT-based algorithm:
/// 1. Reorder input to prepare for FFT
/// 2. Apply n-point FFT (not 2n)
/// 3. Apply pre-computed phase correction
/// 4. Extract real parts
///
/// This is O(n log n) instead of O(n²) for the naive matrix approach.
///
/// DCT-II formula:
/// X_k = sum_{n=0}^{N-1} x_n * cos(pi/N * (n + 0.5) * k)
#[inline(always)]
fn dct2_fft_frame(input: &[f32], output: &mut [f32], n_mfcc: usize, norm: DctNorm) {
    let num_mels = input.len();

    DCT_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        cache.prepare(num_mels);

        let fft = cache.planner.plan_fft_forward(num_mels);

        // Reorder input for efficient DCT via FFT
        // Even indices: x[0], x[2], x[4], ...
        // Odd indices (reversed): x[num_mels-1], x[num_mels-3], x[num_mels-5], ...
        let half = num_mels.div_ceil(2);
        for i in 0..half {
            cache.fft_buffer[i] = Complex::new(input[2 * i], 0.0);
        }
        for i in 0..(num_mels - half) {
            let src_idx = num_mels - 2 * i - 1;
            cache.fft_buffer[half + i] = Complex::new(input[src_idx], 0.0);
        }

        // Apply FFT
        fft.process(&mut cache.fft_buffer);

        // Extract DCT coefficients with pre-computed twiddle factors
        let base_scale = match norm {
            DctNorm::None => 2.0,
            DctNorm::Ortho => (2.0 / num_mels as f32).sqrt(),
        };

        for coeff_idx in 0..n_mfcc.min(num_mels) {
            // Apply phase correction using pre-computed twiddle factors
            let twiddle = cache.twiddle_factors[coeff_idx];
            let real = cache.fft_buffer[coeff_idx].re * twiddle.re
                - cache.fft_buffer[coeff_idx].im * twiddle.im;

            // Apply normalization
            let scale = if norm == DctNorm::Ortho && coeff_idx == 0 {
                base_scale / 2.0f32.sqrt()
            } else {
                base_scale
            };

            output[coeff_idx] = real * scale;
        }
    });
}

/// Apply cepstral lifter in-place.
///
/// lifter formula (common):
/// L[n] = 1 + (L/2) * sin(pi * n / L)
fn apply_lifter(data: &mut [f32], n_frames: usize, n_mfcc: usize, lifter: usize) {
    let lifter_value = lifter as f32;
    let mut lift = vec![1.0f32; n_mfcc];
    for coeff_idx in 0..n_mfcc {
        lift[coeff_idx] = 1.0
            + 0.5 * lifter_value * (std::f32::consts::PI * coeff_idx as f32 / lifter_value).sin();
    }

    for frame_index in 0..n_frames {
        let row = &mut data[frame_index * n_mfcc..(frame_index + 1) * n_mfcc];
        for coeff_idx in 0..n_mfcc {
            row[coeff_idx] *= lift[coeff_idx];
        }
    }
}
