//! Spectral flux and onset strength.

use crate::spectrum::Spectrogram;

/// Spectral flux result.
///
/// One scalar value per frame.
/// The first frame is always 0.0 (no previous frame).
#[derive(Debug, Clone)]
pub struct SpectralFlux {
    n_frames: usize,
    values: Vec<f32>,
}

impl SpectralFlux {
    /// Frame count.
    #[inline]
    pub fn n_frames(&self) -> usize {
        self.n_frames
    }

    /// Values.
    #[inline]
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// Compute spectral flux from a complex STFT spectrogram.
///
/// This computes **positive-only spectral flux**:
///
/// ```text
/// flux[t] = sum_b max(|X_t(b)|^2 - |X_{t-1}(b)|^2, 0)
/// ```
///
/// - Uses power spectrum
/// - Frame 0 is defined as 0.0
pub fn spectral_flux(spec: &Spectrogram) -> SpectralFlux {
    let n_frames = spec.n_frames();
    let n_bins = spec.n_bins();

    let mut values = vec![0.0f32; n_frames];

    if n_frames < 2 {
        return SpectralFlux { n_frames, values };
    }

    // Precompute power spectrum for frame 0
    let mut prev_power = vec![0.0f32; n_bins];
    {
        let frame0 = spec.frame(0).expect("frame index out of bounds");
        for b in 0..n_bins {
            prev_power[b] = frame0[b].norm_sqr();
        }
    }

    // Iterate from frame 1 onward
    for t in 1..n_frames {
        let frame = spec.frame(t).expect("frame index out of bounds");

        let mut flux = 0.0f32;

        for b in 0..n_bins {
            let power = frame[b].norm_sqr();
            let diff = power - prev_power[b];
            if diff > 0.0 {
                flux += diff;
            }
            prev_power[b] = power;
        }

        values[t] = flux;
    }

    SpectralFlux { n_frames, values }
}
