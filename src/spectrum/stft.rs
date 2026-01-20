//! Short-time Fourier transform (STFT).
//!
//! Design notes:
//! - Consumes `signal::Frames` (time-domain, windowed).
//! - Computes one-sided spectra for real inputs: `n_fft/2 + 1` bins.
//! - Stores output as row-major contiguous: frames × bins.
//! - Provides ndarray views for interoperability.

use crate::signal::Frames;

use ndarray::{ArrayView2, ShapeBuilder};
use rayon::prelude::*;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

// Thread-local cache for FFT planners and buffers to avoid repeated allocation
thread_local! {
    static FFT_CACHE: RefCell<HashMap<usize, (Arc<dyn rustfft::Fft<f32>>, Vec<Complex<f32>>)>> = RefCell::new(HashMap::new());
}

/// Errors produced by spectrum operations.
#[derive(thiserror::Error, Debug)]
pub enum SpectrumError {
    /// Invalid STFT configuration.
    #[error("invalid STFT configuration: {0}")]
    InvalidConfig(&'static str),

    /// Frames are empty.
    #[error("frames are empty")]
    EmptyFrames,
}

/// Configuration for STFT.
#[derive(Debug, Clone)]
pub struct StftConfig {
    /// FFT size. If larger than frame_size, frames are zero-padded.
    /// If smaller than frame_size, frames are truncated.
    pub n_fft: usize,
}

impl StftConfig {
    /// Standard default: `n_fft = frame_size`.
    pub fn with_frame_size(frame_size: usize) -> Self {
        Self { n_fft: frame_size }
    }
}

impl Default for StftConfig {
    fn default() -> Self {
        Self { n_fft: 2048 }
    }
}

/// A complex STFT spectrogram (one-sided).
///
/// Storage is row-major: contiguous frames × bins:
/// `data[frame_index * n_bins + bin]`.
#[derive(Debug, Clone)]
pub struct Spectrogram {
    sample_rate: u32,
    n_fft: usize,
    hop_size: usize,
    n_frames: usize,
    n_bins: usize,
    data: Vec<Complex<f32>>,
}

impl Spectrogram {
    /// Sample rate (Hz) of the source audio.
    #[inline]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// FFT size.
    #[inline]
    pub fn n_fft(&self) -> usize {
        self.n_fft
    }

    /// Hop size (samples) from the source framing.
    #[inline]
    pub fn hop_size(&self) -> usize {
        self.hop_size
    }

    /// Number of time frames.
    #[inline]
    pub fn n_frames(&self) -> usize {
        self.n_frames
    }

    /// Number of frequency bins (one-sided).
    #[inline]
    pub fn n_bins(&self) -> usize {
        self.n_bins
    }

    /// Returns an ndarray view of shape `(n_frames, n_bins)` (complex).
    ///
    /// This is zero-copy.
    pub fn as_ndarray(&self) -> ArrayView2<'_, Complex<f32>> {
        let shape = (self.n_frames, self.n_bins).strides((self.n_bins, 1));
        ArrayView2::from_shape(shape, &self.data).expect("spectrogram storage is contiguous")
    }

    /// Returns the complex spectrum for a given frame (one-sided).
    #[inline]
    pub fn frame(&self, index: usize) -> Option<&[Complex<f32>]> {
        if index >= self.n_frames {
            return None;
        }
        let start = index * self.n_bins;
        let end = start + self.n_bins;
        Some(&self.data[start..end])
    }

    /// Convert a bin index to frequency in Hz.
    ///
    /// For one-sided spectra: bins are in `[0..=n_fft/2]`.
    #[inline]
    pub fn bin_frequency_hz(&self, bin: usize) -> Option<f64> {
        if bin >= self.n_bins {
            return None;
        }
        Some(bin as f64 * self.sample_rate as f64 / self.n_fft as f64)
    }

    /// Compute magnitude spectrogram (|X|) as `Vec<f32>` in the same layout:
    /// frames × bins (row-major).
    pub fn magnitude(&self) -> Vec<f32> {
        self.data.iter().map(|c| c.norm()).collect()
    }

    /// Compute power spectrogram (|X|^2) as `Vec<f32>` in the same layout:
    /// frames × bins (row-major).
    pub fn power(&self) -> Vec<f32> {
        self.data.iter().map(|c| c.norm_sqr()).collect()
    }

    /// Internal access to contiguous storage.
    #[inline]
    pub fn as_slice(&self) -> &[Complex<f32>] {
        &self.data
    }
}

/// Compute one-sided STFT of `frames` with `config`.
///
/// Notes:
/// - Frames are assumed to be already windowed (see `signal::frame`).
/// - Uses forward FFT.
/// - Returns one-sided bins: `0..=n_fft/2`.
pub fn stft(frames: &Frames, config: StftConfig) -> Result<Spectrogram, SpectrumError> {
    validate_config(frames, &config)?;

    let n_frames = frames.n_frames();
    if n_frames == 0 {
        return Err(SpectrumError::EmptyFrames);
    }

    let sample_rate = frames.sample_rate();
    let hop_size = frames.hop_size();
    let frame_size = frames.frame_size();
    let n_fft = config.n_fft;

    let n_bins = (n_fft / 2) + 1;

    // Output: contiguous frames × bins.
    // Pre-allocate full output buffer to avoid per-frame allocations
    let mut out = vec![Complex::<f32>::new(0.0, 0.0); n_frames * n_bins];

    // Parallelize FFT computation across frames using thread-local cached planners
    if n_frames > 10 {
        // Parallel path - write directly to pre-allocated buffer
        out.par_chunks_mut(n_bins)
            .enumerate()
            .for_each(|(t, out_row)| {
                let frame = frames
                    .frame(t)
                    .expect("Frames reported n_frames but frame(t) returned None");

                // Use thread-local cache for FFT planner and buffer
                FFT_CACHE.with(|cache| {
                    let mut cache = cache.borrow_mut();

                    // Get or create FFT planner and buffer for this n_fft size
                    let (fft, buf) = cache.entry(n_fft).or_insert_with(|| {
                        let mut planner = FftPlanner::<f32>::new();
                        let fft = planner.plan_fft_forward(n_fft);
                        // Allocate buffer (will be filled on first use)
                        let buf = vec![Complex::<f32>::new(0.0, 0.0); n_fft];
                        (fft, buf)
                    });

                    let copy_len = frame_size.min(n_fft);

                    // Optimized real → complex conversion with SIMD-friendly unrolling
                    const CHUNK: usize = 8;
                    let main_chunks = copy_len / CHUNK;
                    let _remainder = copy_len % CHUNK;

                    for chunk_idx in 0..main_chunks {
                        let base = chunk_idx * CHUNK;
                        buf[base] = Complex::new(frame[base], 0.0);
                        buf[base + 1] = Complex::new(frame[base + 1], 0.0);
                        buf[base + 2] = Complex::new(frame[base + 2], 0.0);
                        buf[base + 3] = Complex::new(frame[base + 3], 0.0);
                        buf[base + 4] = Complex::new(frame[base + 4], 0.0);
                        buf[base + 5] = Complex::new(frame[base + 5], 0.0);
                        buf[base + 6] = Complex::new(frame[base + 6], 0.0);
                        buf[base + 7] = Complex::new(frame[base + 7], 0.0);
                    }

                    for i in (main_chunks * CHUNK)..copy_len {
                        buf[i] = Complex::new(frame[i], 0.0);
                    }

                    // Zero-pad remainder if needed (efficiently)
                    if copy_len < n_fft {
                        buf[copy_len..n_fft].fill(Complex::new(0.0, 0.0));
                    }

                    // FFT in place
                    fft.process(buf);

                    // Copy one-sided bins to output
                    out_row.copy_from_slice(&buf[..n_bins]);
                });
            });
    } else {
        // Sequential path for small frame counts to avoid parallelization overhead
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);
        // Allocate FFT buffer (will be filled each iteration)
        let mut buf = vec![Complex::<f32>::new(0.0, 0.0); n_fft];

        for t in 0..n_frames {
            let frame = frames
                .frame(t)
                .expect("Frames reported n_frames but frame(t) returned None");

            let copy_len = frame_size.min(n_fft);

            // Optimized real → complex conversion with unrolling
            const CHUNK: usize = 8;
            let main_chunks = copy_len / CHUNK;
            let _remainder = copy_len % CHUNK;

            for chunk_idx in 0..main_chunks {
                let base = chunk_idx * CHUNK;
                buf[base] = Complex::new(frame[base], 0.0);
                buf[base + 1] = Complex::new(frame[base + 1], 0.0);
                buf[base + 2] = Complex::new(frame[base + 2], 0.0);
                buf[base + 3] = Complex::new(frame[base + 3], 0.0);
                buf[base + 4] = Complex::new(frame[base + 4], 0.0);
                buf[base + 5] = Complex::new(frame[base + 5], 0.0);
                buf[base + 6] = Complex::new(frame[base + 6], 0.0);
                buf[base + 7] = Complex::new(frame[base + 7], 0.0);
            }

            for i in (main_chunks * CHUNK)..copy_len {
                buf[i] = Complex::new(frame[i], 0.0);
            }

            // Zero-pad remainder if needed (efficiently)
            if copy_len < n_fft {
                buf[copy_len..n_fft].fill(Complex::new(0.0, 0.0));
            }

            // FFT in place.
            fft.process(&mut buf);

            // Store one-sided bins.
            let row = &mut out[t * n_bins..(t + 1) * n_bins];
            row.copy_from_slice(&buf[..n_bins]);
        }
    }

    Ok(Spectrogram {
        sample_rate,
        n_fft,
        hop_size,
        n_frames,
        n_bins,
        data: out,
    })
}

fn validate_config(frames: &Frames, config: &StftConfig) -> Result<(), SpectrumError> {
    if config.n_fft == 0 {
        return Err(SpectrumError::InvalidConfig("n_fft must be > 0"));
    }
    // FFT sizes are typically powers of two, but rustfft supports any size.
    // We don't enforce power-of-two here.
    if frames.frame_size() == 0 {
        return Err(SpectrumError::InvalidConfig(
            "frames.frame_size must be > 0",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::buffer::Buffer;
    use crate::signal::Window;
    use crate::signal::{ChannelMode, FrameConfig, Padding, frame};

    fn sine_buffer(sample_rate: u32, freq_hz: f32, seconds: f32) -> Buffer {
        let n = (sample_rate as f32 * seconds) as usize;
        let mut samples = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / sample_rate as f32;
            let x = (2.0 * std::f32::consts::PI * freq_hz * t).sin();
            samples.push(x);
        }
        Buffer::new(sample_rate, 1, samples)
    }

    #[test]
    fn stft_has_expected_bin_peak_for_sine() {
        // Small, deterministic test:
        // sr=8000, n_fft=8 => bin resolution = 1000 Hz
        // use freq=1000 Hz => should peak at bin 1
        let audio = sine_buffer(8000, 1000.0, 0.01);

        let cfg = FrameConfig {
            frame_size: 8,
            hop_size: 8,
            window: Window::Rectangular,
            padding: Padding::None,
            channel_mode: ChannelMode::Channel(0),
            center: false,
        };

        let frames = frame(&audio, cfg).unwrap();
        assert!(frames.n_frames() > 0);

        let spec = stft(&frames, StftConfig { n_fft: 8 }).unwrap();
        assert_eq!(spec.n_bins(), 5); // 8/2+1

        // Check first frame peak bin
        let first = spec.frame(0).unwrap();
        let mags: Vec<f32> = first.iter().map(|c| c.norm()).collect();

        let (max_bin, _max_val) = mags
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        assert_eq!(max_bin, 1);

        // Bin 1 frequency should be 1000 Hz
        let f = spec.bin_frequency_hz(1).unwrap();
        assert!((f - 1000.0).abs() < 1e-6);
    }
}
