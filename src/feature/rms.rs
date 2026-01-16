//! Root Mean Square (RMS) energy computation.

use crate::signal::Frames;
use crate::spectrum::Spectrogram;
use rayon::prelude::*;

/// RMS energy result.
#[derive(Debug, Clone)]
pub struct Rms {
    n_frames: usize,
    values: Vec<f32>,
}

impl Rms {
    /// Frame count.
    #[inline]
    pub fn n_frames(&self) -> usize {
        self.n_frames
    }

    /// RMS energy values.
    #[inline]
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// Compute root-mean-square (RMS) energy for each frame from time-domain signal.
///
/// The RMS value for a frame is computed as:
/// ```text
/// RMS = sqrt(mean(x^2)) = sqrt((1/N) * sum(x_i^2))
/// ```
///
/// This matches `librosa.feature.rms(y=...)` behavior.
///
/// # Arguments
/// * `frames` - Time-domain audio frames
///
/// # Returns
/// RMS energy for each frame
pub fn rms(frames: &Frames) -> Rms {
    let n_frames = frames.n_frames();
    let frame_size = frames.frame_size();

    // Direct access to frame data for better performance
    let data = frames.as_slice();
    let inv_frame_size = 1.0 / frame_size as f32;

    // Compute RMS for each frame in parallel for large datasets
    let values = if n_frames > 10 {
        (0..n_frames)
            .into_par_iter()
            .map(|t| {
                let start = t * frame_size;
                let end = start + frame_size;
                let frame_data = &data[start..end];

                // Compute sum of squares using optimized iteration
                let sum_squares: f32 = frame_data.iter().map(|&x| x * x).sum();

                // RMS = sqrt(mean(x^2))
                (sum_squares * inv_frame_size).sqrt()
            })
            .collect()
    } else {
        // Sequential for small frame counts to avoid threading overhead
        (0..n_frames)
            .map(|t| {
                let start = t * frame_size;
                let end = start + frame_size;
                let frame_data = &data[start..end];

                let sum_squares: f32 = frame_data.iter().map(|&x| x * x).sum();
                (sum_squares * inv_frame_size).sqrt()
            })
            .collect()
    };

    Rms { n_frames, values }
}

/// Compute root-mean-square (RMS) energy for each frame from a spectrogram.
///
/// The RMS value for a frame is computed from the spectrogram magnitude as:
/// ```text
/// RMS = sqrt((1/N) * sum(|S(f)|^2))
/// ```
///
/// This matches `librosa.feature.rms(S=...)` behavior.
///
/// # Arguments
/// * `spec` - Magnitude spectrogram (STFT or other)
///
/// # Returns
/// RMS energy for each frame
pub fn rms_from_spectrogram(spec: &Spectrogram) -> Rms {
    let n_frames = spec.n_frames();
    let n_bins = spec.n_bins();
    let n_fft = spec.n_fft();
    let norm_factor = 2.0 / (n_fft * n_fft) as f32;
    let is_even_fft = n_fft % 2 == 0;

    // Parallelize for large frame counts
    let values = if n_frames > 10 {
        (0..n_frames)
            .into_par_iter()
            .map(|t| {
                let frame = spec.frame(t).expect("frame index out of bounds");
                compute_rms_from_frame(frame, n_bins, is_even_fft, norm_factor)
            })
            .collect()
    } else {
        // Sequential for small frame counts
        (0..n_frames)
            .map(|t| {
                let frame = spec.frame(t).expect("frame index out of bounds");
                compute_rms_from_frame(frame, n_bins, is_even_fft, norm_factor)
            })
            .collect()
    };

    Rms { n_frames, values }
}

/// Helper function to compute RMS from a single spectrogram frame.
#[inline]
fn compute_rms_from_frame(
    frame: &[rustfft::num_complex::Complex<f32>],
    n_bins: usize,
    is_even_fft: bool,
    norm_factor: f32,
) -> f32 {
    let mut sum_power = 0.0f32;

    // Process DC component (bin 0)
    let mag = frame[0].norm();
    sum_power += 0.5 * mag * mag;

    // Process middle bins (full power)
    let end = if is_even_fft { n_bins - 1 } else { n_bins };
    for b in 1..end {
        let mag = frame[b].norm();
        sum_power += mag * mag;
    }

    // Process Nyquist component (last bin) if n_fft is even
    if is_even_fft {
        let mag = frame[n_bins - 1].norm();
        sum_power += 0.5 * mag * mag;
    }

    // RMS = sqrt(2 * sum(power) / n_fft^2)
    (sum_power * norm_factor).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::Buffer;
    use crate::signal::{FrameConfig, Window, frame};
    use crate::spectrum::{StftConfig, stft};

    #[test]
    fn rms_constant_signal() {
        // A constant signal should have RMS equal to its absolute value
        // Use 4096 samples to ensure we have complete frames
        let samples = vec![0.5f32; 4096];
        let audio = Buffer::new(16000, 1, samples);
        let config = FrameConfig {
            frame_size: 512,
            hop_size: 256,
            window: Window::Rectangular,
            center: false,
            ..Default::default()
        };

        let frames = frame(&audio, config).unwrap();
        let result = rms(&frames);

        // RMS of constant 0.5 should be 0.5
        // Check all frames except potentially the last one which might be zero-padded
        let n_complete_frames = ((4096 - 512) / 256) + 1; // All complete frames
        for i in 0..n_complete_frames {
            let val = result.values()[i];
            assert!(
                (val - 0.5).abs() < 1e-6,
                "Frame {}: expected 0.5, got {}",
                i,
                val
            );
        }
    }

    #[test]
    fn rms_zero_signal() {
        // Zero signal should have zero RMS
        let samples = vec![0.0f32; 4096];
        let audio = Buffer::new(16000, 1, samples);
        let config = FrameConfig {
            frame_size: 512,
            hop_size: 256,
            window: Window::Rectangular,
            center: false,
            ..Default::default()
        };

        let frames = frame(&audio, config).unwrap();
        let result = rms(&frames);

        for &val in result.values() {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn rms_from_spectrogram_matches_time_domain() {
        // RMS from spectrogram should match time domain with rectangular window
        // due to Parseval's theorem
        let mut samples = vec![0.0f32; 8192];
        for (i, x) in samples.iter_mut().enumerate() {
            *x = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin();
        }
        let audio = Buffer::new(16000, 1, samples);

        let frame_config = FrameConfig {
            frame_size: 512,
            hop_size: 256,
            window: Window::Rectangular,
            center: false,
            ..Default::default()
        };

        let frames = frame(&audio, frame_config).unwrap();
        let rms_time = rms(&frames);

        let stft_config = StftConfig { n_fft: 512 };

        let spec = stft(&frames, stft_config).unwrap();
        let rms_spec = rms_from_spectrogram(&spec);

        assert_eq!(rms_time.n_frames(), rms_spec.n_frames());

        // With rectangular window, Parseval's theorem ensures they should match closely
        // The RMS of a sine wave is amplitude/sqrt(2) ≈ 0.707
        // Check all frames except potentially the last one which might be zero-padded
        let n_complete_frames = ((8192 - 512) / 256) + 1;
        for i in 0..n_complete_frames.min(rms_time.n_frames()) {
            let t = rms_time.values()[i];
            let s = rms_spec.values()[i];
            if t > 1e-6 {
                // Allow small numerical differences (within 1%)
                let rel_error = (t - s).abs() / t;
                assert!(
                    rel_error < 0.01,
                    "Frame {}: Relative error {} too large (t={}, s={})",
                    i,
                    rel_error,
                    t,
                    s
                );
            }
        }
    }
}
