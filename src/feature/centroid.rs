//! Spectral centroid.

use crate::spectrum::Spectrogram;

/// Spectral centroid result.
#[derive(Debug, Clone)]
pub struct SpectralCentroid {
    n_frames: usize,
    values: Vec<f32>,
}

impl SpectralCentroid {
    /// Frame count.
    #[inline]
    pub fn n_frames(&self) -> usize {
        self.n_frames
    }

    /// Spectral centroid values.ß
    #[inline]
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// Compute spectral centroid (Hz) for each frame.
///
/// Centroid = sum(f * |X(f)|) / sum(|X(f)|)
pub fn spectral_centroid(spec: &Spectrogram) -> SpectralCentroid {
    let n_frames = spec.n_frames();
    let n_bins = spec.n_bins();

    let mut values = Vec::with_capacity(n_frames);

    for t in 0..n_frames {
        let frame = spec.frame(t).expect("frame index out of bounds");

        let mut num = 0.0f32;
        let mut den = 0.0f32;

        for b in 0..n_bins {
            let freq = spec.bin_frequency_hz(b).unwrap() as f32;
            let magnitude = frame[b].norm();

            num += freq * magnitude;
            den += magnitude;
        }

        values.push(if den > 0.0 { num / den } else { 0.0 });
    }

    SpectralCentroid { n_frames, values }
}
