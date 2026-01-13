//! Chromagram (pitch class profile) extraction.
//!
//! Chromagrams represent the intensity of each of the 12 pitch classes
//! (C, C#, D, ..., B) over time, collapsing octave information.

use crate::spectrum::Spectrogram;
use ndarray::{ArrayView2, ShapeBuilder};
use rayon::prelude::*;

/// Tuning reference for A4 (Hz).
pub const A4_HZ: f32 = 440.0;

/// Number of pitch classes in Western music.
pub const N_CHROMA: usize = 12;

/// Configuration for chromagram computation.
#[derive(Debug, Clone)]
pub struct ChromaConfig {
    /// Number of chroma bins (typically 12 for Western music).
    pub n_chroma: usize,

    /// Tuning deviation from A440 (in fractions of a bin).
    /// 0.0 means A4 = 440 Hz exactly.
    pub tuning: f32,

    /// Normalization method.
    pub norm: ChromaNorm,

    /// Minimum frequency to consider (Hz).
    pub fmin: f32,
    /// Maximum frequency to consider (Hz). If None, uses Nyquist frequency.
    pub fmax: Option<f32>,
}

impl Default for ChromaConfig {
    fn default() -> Self {
        Self {
            n_chroma: N_CHROMA,
            tuning: 0.0,
            norm: ChromaNorm::None,
            fmin: 32.7, // C1
            fmax: None, // Nyquist
        }
    }
}

/// Normalization methods for chroma features.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaNorm {
    /// No normalization.
    None,
    /// Normalize each frame to sum to 1.
    L1,
    /// Normalize each frame to unit L2 norm.
    L2,
    /// Normalize each frame so max value is 1.
    Max,
}

/// A chromagram (pitch class profile over time).
///
/// Storage is row-major: frames × chroma bins.
#[derive(Debug, Clone)]
pub struct Chromagram {
    sample_rate: u32,
    hop_size: usize,
    n_fft: usize,
    n_frames: usize,
    n_chroma: usize,
    /// Row-major: `data[t * n_chroma + c]`.
    data: Vec<f32>,
}

impl Chromagram {
    /// Returns the sample rate of the chromagram.
    #[inline]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Returns the hop size of the chromagram.
    #[inline]
    pub fn hop_size(&self) -> usize {
        self.hop_size
    }

    /// Returns the number of FFT bins.
    #[inline]
    pub fn n_fft(&self) -> usize {
        self.n_fft
    }

    /// Returns the number of frames in the chromagram.
    #[inline]
    pub fn n_frames(&self) -> usize {
        self.n_frames
    }

    /// Returns the number of chroma bins.
    #[inline]
    pub fn n_chroma(&self) -> usize {
        self.n_chroma
    }

    /// View as ndarray shape (n_frames, n_chroma).
    pub fn as_ndarray(&self) -> ArrayView2<'_, f32> {
        let shape = (self.n_frames, self.n_chroma).strides((self.n_chroma, 1));
        ArrayView2::from_shape(shape, &self.data).expect("chroma storage is contiguous")
    }

    /// Returns the chroma frame at the given index.
    #[inline]
    pub fn frame(&self, index: usize) -> Option<&[f32]> {
        if index >= self.n_frames {
            return None;
        }
        let start = index * self.n_chroma;
        let end = start + self.n_chroma;
        Some(&self.data[start..end])
    }

    /// Returns the chromagram as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

/// Compute a chromagram from a complex STFT spectrogram.
///
/// This maps each frequency bin to its corresponding pitch class,
/// sums magnitude across octaves, and applies optional normalization.
pub fn chroma_stft(spec: &Spectrogram, cfg: ChromaConfig) -> Chromagram {
    let n_frames = spec.n_frames();
    let n_bins = spec.n_bins();
    let sr = spec.sample_rate();
    let n_fft = spec.n_fft();
    let hop = spec.hop_size();

    let nyquist = sr as f32 / 2.0;
    let fmax = cfg.fmax.unwrap_or(nyquist).min(nyquist);

    // Build chroma filter bank (maps FFT bins to chroma bins)
    let chroma_filters =
        build_chroma_filterbank(sr, n_fft, n_bins, cfg.n_chroma, cfg.tuning, cfg.fmin, fmax);

    // Output: frames × chroma
    let mut out = vec![0.0f32; n_frames * cfg.n_chroma];

    // Parallelize for large frame counts
    if n_frames > 10 {
        out.par_chunks_mut(cfg.n_chroma)
            .enumerate()
            .for_each(|(t, out_row)| {
                let x = spec.frame(t).expect("n_frames mismatch");
                apply_chroma_filters(&chroma_filters, x, out_row);
                normalize_chroma_frame(out_row, cfg.norm);
            });
    } else {
        for t in 0..n_frames {
            let x = spec.frame(t).expect("n_frames mismatch");
            let out_row = &mut out[t * cfg.n_chroma..(t + 1) * cfg.n_chroma];
            apply_chroma_filters(&chroma_filters, x, out_row);
            normalize_chroma_frame(out_row, cfg.norm);
        }
    }

    Chromagram {
        sample_rate: sr,
        hop_size: hop,
        n_fft,
        n_frames,
        n_chroma: cfg.n_chroma,
        data: out,
    }
}

/// Chroma filter bank: maps FFT bins to pitch classes.
///
/// Each filter is a vector of length n_bins, one per chroma class.
#[derive(Debug, Clone)]
struct ChromaFilterBank {
    _n_chroma: usize,
    _n_bins: usize,
    /// filters[c] contains the weights for chroma class c
    filters: Vec<Vec<(usize, f32)>>, // (bin_idx, weight) pairs
}

/// Build chroma filter bank using a Gaussian weighting around each pitch class.
fn build_chroma_filterbank(
    sample_rate: u32,
    n_fft: usize,
    n_bins: usize,
    n_chroma: usize,
    tuning: f32,
    fmin: f32,
    fmax: f32,
) -> ChromaFilterBank {
    assert!(n_bins == (n_fft / 2) + 1, "n_bins must be n_fft/2 + 1");
    assert_eq!(
        n_chroma, N_CHROMA,
        "Currently only n_chroma=12 is supported"
    );

    // Compute frequency for each FFT bin
    let freqs: Vec<f32> = (0..n_bins)
        .map(|b| (b as f32 * sample_rate as f32) / n_fft as f32)
        .collect();

    // Initialize filters (sparse representation)
    let mut filters: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_chroma];

    // For each FFT bin, determine which chroma class(es) it contributes to
    for (bin_idx, &freq) in freqs.iter().enumerate() {
        if freq < fmin || freq > fmax || freq <= 0.0 {
            continue;
        }

        // Convert frequency to pitch in semitones relative to C0 (16.35 Hz)
        // pitch = 12 * log2(freq / C0)
        let c0 = 16.35159783128741; // C0 in Hz
        let pitch = 12.0 * (freq / c0).log2() + tuning;

        // Chroma class (0..11)
        let chroma = pitch.rem_euclid(12.0);

        // Use a Gaussian window centered on the chroma class
        // This allows smooth blending when a bin falls between classes
        let chroma_center = chroma.round();
        let chroma_idx = (chroma_center as usize) % n_chroma;

        // Distance from center (in semitones within 0..12 range)
        let mut delta = chroma - chroma_center;
        if delta > 6.0 {
            delta -= 12.0;
        } else if delta < -6.0 {
            delta += 12.0;
        }

        // Gaussian weight (sigma = 0.5 semitones for reasonable spread)
        let sigma = 0.5f32;
        let weight = (-0.5 * (delta / sigma).powi(2)).exp();

        // Only add if weight is significant
        if weight > 0.01 {
            filters[chroma_idx].push((bin_idx, weight));
        }

        // Also consider neighboring chroma classes for smooth transitions
        if delta.abs() > 0.1 {
            let neighbor_idx = if delta > 0.0 {
                (chroma_idx + 1) % n_chroma
            } else {
                (chroma_idx + n_chroma - 1) % n_chroma
            };

            let neighbor_delta = if delta > 0.0 {
                delta - 1.0
            } else {
                delta + 1.0
            };
            let neighbor_weight = (-0.5 * (neighbor_delta / sigma).powi(2)).exp();

            if neighbor_weight > 0.01 {
                filters[neighbor_idx].push((bin_idx, neighbor_weight));
            }
        }
    }

    // Normalize each filter to sum to 1
    for filter in &mut filters {
        let sum: f32 = filter.iter().map(|(_, w)| w).sum();
        if sum > 1e-10 {
            for (_, w) in filter.iter_mut() {
                *w /= sum;
            }
        }
    }

    ChromaFilterBank {
        _n_chroma: n_chroma,
        _n_bins: n_bins,
        filters,
    }
}

/// Apply chroma filters to a single STFT frame.
#[inline]
fn apply_chroma_filters(
    bank: &ChromaFilterBank,
    x: &[rustfft::num_complex::Complex<f32>],
    out: &mut [f32],
) {
    // Zero output
    for val in out.iter_mut() {
        *val = 0.0;
    }

    // For each chroma class, sum weighted magnitudes
    for (chroma_idx, filter) in bank.filters.iter().enumerate() {
        let mut sum = 0.0f32;
        for &(bin_idx, weight) in filter {
            let magnitude = x[bin_idx].norm();
            sum += magnitude * weight;
        }
        out[chroma_idx] = sum;
    }
}

/// Normalize a chroma frame according to the specified method.
#[inline]
fn normalize_chroma_frame(frame: &mut [f32], norm: ChromaNorm) {
    match norm {
        ChromaNorm::None => {}
        ChromaNorm::L1 => {
            let sum: f32 = frame.iter().sum();
            if sum > 1e-10 {
                for val in frame.iter_mut() {
                    *val /= sum;
                }
            }
        }
        ChromaNorm::L2 => {
            let norm: f32 = frame.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 1e-10 {
                for val in frame.iter_mut() {
                    *val /= norm;
                }
            }
        }
        ChromaNorm::Max => {
            let max = frame.iter().cloned().fold(0.0f32, f32::max);
            if max > 1e-10 {
                for val in frame.iter_mut() {
                    *val /= max;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::Buffer;
    use crate::signal::{self, FrameConfig};
    use crate::spectrum::{StftConfig, stft};
    use std::f32::consts::PI;

    #[test]
    fn test_chroma_basic() {
        // Create a simple test signal (sine wave at A440)
        let sr = 22050;
        let duration = 1.0;
        let n_samples = (sr as f32 * duration) as usize;
        let freq = 440.0; // A4

        let mut samples = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let t = i as f32 / sr as f32;
            samples.push((2.0 * PI * freq * t).sin());
        }

        let audio = Buffer::new(sr, 1, samples);
        let frames = signal::frame(
            &audio,
            FrameConfig {
                frame_size: 2048,
                hop_size: 512,
                ..Default::default()
            },
        )
        .expect("Failed to frame audio");

        let spec = stft(&frames, StftConfig::default()).expect("Failed to compute STFT");
        let chroma = chroma_stft(&spec, ChromaConfig::default());

        // Should have the expected dimensions
        assert_eq!(chroma.n_chroma(), 12);
        assert!(chroma.n_frames() > 0);

        // For a pure A440 tone, chroma bin 9 (A) should dominate
        let frame = chroma.frame(chroma.n_frames() / 2).unwrap();
        let max_idx = frame
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        assert_eq!(max_idx, 9, "A440 should activate chroma bin 9 (A)");
    }

    #[test]
    fn test_chroma_normalization() {
        let sr = 22050;
        let n_samples = 2048;
        let samples = vec![0.5f32; n_samples];

        let audio = Buffer::new(sr, 1, samples);
        let frames = signal::frame(&audio, FrameConfig::default()).expect("Failed to frame audio");
        let spec = stft(&frames, StftConfig::default()).expect("Failed to compute STFT");

        // Test L1 normalization
        let chroma_l1 = chroma_stft(
            &spec,
            ChromaConfig {
                norm: ChromaNorm::L1,
                ..Default::default()
            },
        );

        if let Some(frame) = chroma_l1.frame(0) {
            let sum: f32 = frame.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "L1 norm should sum to 1.0");
        }
    }
}
