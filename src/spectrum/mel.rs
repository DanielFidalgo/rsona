//! Mel filter bank and mel spectrogram.
//!
//! This module converts a one-sided power spectrogram into mel bands.

use crate::spectrum::Spectrogram;

use ndarray::{ArrayView2, ShapeBuilder};
use rayon::prelude::*;
use std::cell::RefCell;

// Thread-local buffer to avoid repeated allocation in parallel mel computation
thread_local! {
    static MEL_POWER_BUFFER: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Mel scaling convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelScale {
    /// Slaney-style mel scale (standard default when `htk=false`).
    Slaney,
    /// HTK mel scale.
    Htk,
}

/// Configuration for mel spectrogram computation.
#[derive(Debug, Clone)]
pub struct MelConfig {
    /// Number of mel bands.
    pub n_mels: usize,
    /// Minimum frequency (Hz).
    pub fmin: f32,
    /// Maximum frequency (Hz). If None, uses Nyquist.
    pub fmax: Option<f32>,
    /// Mel scale convention.
    pub mel_scale: MelScale,
    /// Apply Slaney-style area normalization (recommended for Slaney).
    ///
    /// Standard implementations use Slaney mel scale by default and apply normalization.
    pub normalize: bool,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            n_mels: 128,
            fmin: 0.0,
            fmax: None, // Nyquist
            mel_scale: MelScale::Slaney,
            normalize: true,
        }
    }
}

/// Sparse filter representation for one mel band
#[derive(Debug, Clone)]
struct SparseFilter {
    /// Starting bin index (inclusive)
    start: usize,
    /// Ending bin index (exclusive)
    end: usize,
    /// Non-zero filter coefficients
    coeffs: Vec<f32>,
}

/// Precomputed mel filter bank matrix.
///
/// Shape is (n_mels, n_bins).
#[derive(Debug, Clone)]
pub struct MelFilterBank {
    /// Number of mel bins.
    pub n_mels: usize,
    /// Number of FFT bins.
    pub n_bins: usize,
    /// Sample rate (Hz).
    pub sample_rate: u32,
    /// Number of FFT bins.
    pub n_fft: usize,
    /// Minimum frequency (Hz).
    pub fmin: f32,
    /// Maximum frequency (Hz).
    pub fmax: f32,
    /// Mel scale.
    pub mel_scale: MelScale,
    /// Should normalize.
    pub normalize: bool,
    /// Sparse representation: one filter per mel band
    sparse_filters: Vec<SparseFilter>,
}

/// A mel spectrogram (power in mel bands).
///
/// Storage row-major: frames × mels.
#[derive(Debug, Clone)]
pub struct MelSpectrogram {
    sample_rate: u32,
    hop_size: usize,
    n_fft: usize,
    n_frames: usize,
    n_mels: usize,
    /// Row-major: `data[t * n_mels + m]`.
    data: Vec<f32>,
}

impl MelSpectrogram {
    #[inline]
    /// Returns the sample rate of the spectrogram.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    #[inline]
    /// Returns the hop size of the spectrogram.
    pub fn hop_size(&self) -> usize {
        self.hop_size
    }
    #[inline]
    /// Returns the number of FFT bins.
    pub fn n_fft(&self) -> usize {
        self.n_fft
    }
    #[inline]
    /// Returns the number of frames in the spectrogram.
    pub fn n_frames(&self) -> usize {
        self.n_frames
    }
    #[inline]
    /// Returns the number of mel bins.
    pub fn n_mels(&self) -> usize {
        self.n_mels
    }

    /// View as ndarray shape (n_frames, n_mels).
    pub fn as_ndarray(&self) -> ArrayView2<'_, f32> {
        let shape = (self.n_frames, self.n_mels).strides((self.n_mels, 1));
        ArrayView2::from_shape(shape, &self.data).expect("mel spec storage is contiguous")
    }

    #[inline]
    /// Returns the frame at the given index.
    pub fn frame(&self, index: usize) -> Option<&[f32]> {
        if index >= self.n_frames {
            return None;
        }
        let start = index * self.n_mels;
        let end = start + self.n_mels;
        Some(&self.data[start..end])
    }

    #[inline]
    /// Returns the mel spectrogram as a slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

/// Compute a mel spectrogram from a complex one-sided STFT spectrogram.
///
/// This uses power spectrum `|X|^2` and multiplies by the mel filter bank.
pub fn mel_spectrogram(spec: &Spectrogram, cfg: MelConfig) -> MelSpectrogram {
    let n_frames = spec.n_frames();
    let n_bins = spec.n_bins();
    let sr = spec.sample_rate();
    let n_fft = spec.n_fft();
    let hop = spec.hop_size();

    let nyquist = sr as f32 / 2.0;
    let fmax = cfg.fmax.unwrap_or(nyquist).min(nyquist).max(cfg.fmin);

    let bank = build_mel_filterbank(
        sr,
        n_fft,
        n_bins,
        cfg.n_mels,
        cfg.fmin,
        fmax,
        cfg.mel_scale,
        cfg.normalize,
    );

    // Output frames × mels.
    let mut out = vec![0.0f32; n_frames * cfg.n_mels];

    // For each frame, compute power bins and apply filters.
    // Parallelize for large frame counts
    if n_frames > 10 {
        out.par_chunks_mut(cfg.n_mels)
            .enumerate()
            .for_each(|(frame_index, out_row)| {
                let spectrum_frame = spec.frame(frame_index).expect("n_frames mismatch");

                // Apply sparse mel filters
                for mel_idx in 0..cfg.n_mels {
                    out_row[mel_idx] =
                        apply_sparse_filter(&bank.sparse_filters[mel_idx], spectrum_frame);
                }
            });
    } else {
        // Sequential path for small frame counts
        for frame_index in 0..n_frames {
            let spectrum_frame = spec.frame(frame_index).expect("n_frames mismatch");
            let out_row = &mut out[frame_index * cfg.n_mels..(frame_index + 1) * cfg.n_mels];

            // Apply sparse mel filters
            for mel_idx in 0..cfg.n_mels {
                out_row[mel_idx] =
                    apply_sparse_filter(&bank.sparse_filters[mel_idx], spectrum_frame);
            }
        }
    }

    MelSpectrogram {
        sample_rate: sr,
        hop_size: hop,
        n_fft,
        n_frames,
        n_mels: cfg.n_mels,
        data: out,
    }
}

/// Apply sparse mel filter: only compute power for non-zero filter coefficients
#[inline(always)]
fn apply_sparse_filter(
    filter: &SparseFilter,
    spectrum_frame: &[rustfft::num_complex::Complex<f32>],
) -> f32 {
    let len = filter.end - filter.start;
    let mut sum = 0.0f32;

    // Process in chunks of 8 for SIMD
    const CHUNK: usize = 8;
    let main_chunks = len / CHUNK;
    let remainder = len % CHUNK;

    // Main loop with manual unrolling
    for chunk_idx in 0..main_chunks {
        let base_filter = chunk_idx * CHUNK;
        let base_bin = filter.start + base_filter;

        // Compute power and multiply by filter coefficient in one go
        sum += spectrum_frame[base_bin].norm_sqr() * filter.coeffs[base_filter];
        sum += spectrum_frame[base_bin + 1].norm_sqr() * filter.coeffs[base_filter + 1];
        sum += spectrum_frame[base_bin + 2].norm_sqr() * filter.coeffs[base_filter + 2];
        sum += spectrum_frame[base_bin + 3].norm_sqr() * filter.coeffs[base_filter + 3];
        sum += spectrum_frame[base_bin + 4].norm_sqr() * filter.coeffs[base_filter + 4];
        sum += spectrum_frame[base_bin + 5].norm_sqr() * filter.coeffs[base_filter + 5];
        sum += spectrum_frame[base_bin + 6].norm_sqr() * filter.coeffs[base_filter + 6];
        sum += spectrum_frame[base_bin + 7].norm_sqr() * filter.coeffs[base_filter + 7];
    }

    // Handle remainder
    for i in 0..remainder {
        let filter_idx = main_chunks * CHUNK + i;
        let bin_idx = filter.start + filter_idx;
        sum += spectrum_frame[bin_idx].norm_sqr() * filter.coeffs[filter_idx];
    }

    sum
}

/// Build mel filter bank (triangular) for one-sided FFT bins with sparse representation.
fn build_mel_filterbank(
    sample_rate: u32,
    n_fft: usize,
    n_bins: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
    mel_scale: MelScale,
    normalize: bool,
) -> MelFilterBank {
    assert!(n_mels > 0);
    assert!(n_bins == (n_fft / 2) + 1, "n_bins must be n_fft/2 + 1");

    // 1) Compute mel-spaced frequencies (n_mels + 2 points).
    let m_min = hz_to_mel(fmin, mel_scale);
    let m_max = hz_to_mel(fmax, mel_scale);

    let mut mel_points = Vec::with_capacity(n_mels + 2);
    for i in 0..(n_mels + 2) {
        let alpha = i as f32 / (n_mels + 1) as f32;
        mel_points.push(m_min + alpha * (m_max - m_min));
    }

    let hz_points: Vec<f32> = mel_points
        .into_iter()
        .map(|m| mel_to_hz(m, mel_scale))
        .collect();

    // 2) Map Hz points to FFT bin indices.
    // Bin frequency = b * sr / n_fft.
    let bin = |hz: f32| -> usize {
        let b = (hz * n_fft as f32 / sample_rate as f32).floor() as isize;
        b.clamp(0, (n_bins - 1) as isize) as usize
    };

    let mut bins = Vec::with_capacity(n_mels + 2);
    for &hz in &hz_points {
        bins.push(bin(hz));
    }

    // 3) Build sparse triangular filters.
    let mut sparse_filters = Vec::with_capacity(n_mels);

    for mel_idx in 0..n_mels {
        let left = bins[mel_idx];
        let center = bins[mel_idx + 1];
        let right = bins[mel_idx + 2];

        if left == center || center == right {
            // Degenerate; create empty filter
            sparse_filters.push(SparseFilter {
                start: left,
                end: left,
                coeffs: Vec::new(),
            });
            continue;
        }

        // Build coefficients for non-zero range [left, right)
        let len = right - left;
        let mut coeffs = Vec::with_capacity(len);

        // Rising slope: left -> center
        for b in left..center {
            let num = (b - left) as f32;
            let den = (center - left) as f32;
            coeffs.push(num / den);
        }

        // Falling slope: center -> right
        for b in center..right {
            let num = (right - b) as f32;
            let den = (right - center) as f32;
            coeffs.push(num / den);
        }

        // Optional Slaney-style normalization: scale each filter by 2/(f_{m+2}-f_m)
        if normalize {
            let f_left = hz_points[mel_idx];
            let f_right = hz_points[mel_idx + 2];
            let enorm = 2.0 / (f_right - f_left).max(1e-12);
            for coeff in &mut coeffs {
                *coeff *= enorm;
            }
        }

        sparse_filters.push(SparseFilter {
            start: left,
            end: right,
            coeffs,
        });
    }

    MelFilterBank {
        n_mels,
        n_bins,
        sample_rate,
        n_fft,
        fmin,
        fmax,
        mel_scale,
        normalize,
        sparse_filters,
    }
}

fn hz_to_mel(hz: f32, scale: MelScale) -> f32 {
    match scale {
        MelScale::Htk => 2595.0 * (1.0 + hz / 700.0).log10(),
        MelScale::Slaney => {
            // Slaney approximation: linear below 1 kHz, log above.
            // Matches common MIR implementations with htk=false setting.
            let f_sp = 200.0 / 3.0;
            let min_log_hz = 1000.0;
            let min_log_mel = min_log_hz / f_sp;
            let logstep = (6.4f32).ln() / 27.0;

            if hz < min_log_hz {
                hz / f_sp
            } else {
                min_log_mel + (hz / min_log_hz).ln() / logstep
            }
        }
    }
}

fn mel_to_hz(mel: f32, scale: MelScale) -> f32 {
    match scale {
        MelScale::Htk => 700.0 * (10f32.powf(mel / 2595.0) - 1.0),
        MelScale::Slaney => {
            let f_sp = 200.0 / 3.0;
            let min_log_hz = 1000.0;
            let min_log_mel = min_log_hz / f_sp;
            let logstep = (6.4f32).ln() / 27.0;

            if mel < min_log_mel {
                mel * f_sp
            } else {
                min_log_hz * ((mel - min_log_mel) * logstep).exp()
            }
        }
    }
}
