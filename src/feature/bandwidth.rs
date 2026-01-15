//! Spectral bandwidth.

use crate::spectrum::Spectrogram;

/// Spectral bandwidth result.
#[derive(Debug, Clone)]
pub struct SpectralBandwidth {
    n_frames: usize,
    values: Vec<f32>,
}

impl SpectralBandwidth {
    /// Frame count.
    #[inline]
    pub fn n_frames(&self) -> usize {
        self.n_frames
    }

    ///
    #[inline]
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// Compute spectral bandwidth (Hz) for each frame.
///
/// Bandwidth = sqrt( sum((f - centroid)^2 * |X(f)|^2) / sum(|X(f)|^2) )
pub fn spectral_bandwidth(spec: &Spectrogram) -> SpectralBandwidth {
    let n_frames = spec.n_frames();
    let n_bins = spec.n_bins();

    let mut values = Vec::with_capacity(n_frames);

    for t in 0..n_frames {
        let frame = spec.frame(t).expect("frame index out of bounds");

        // First pass: centroid
        let mut num = 0.0f32;
        let mut den = 0.0f32;

        for b in 0..n_bins {
            let freq = spec.bin_frequency_hz(b).unwrap() as f32;
            let power = frame[b].norm();

            num += freq * power;
            den += power;
        }

        let centroid = if den > 0.0 { num / den } else { 0.0 };

        // Second pass: variance
        let mut var = 0.0f32;
        for b in 0..n_bins {
            let freq = spec.bin_frequency_hz(b).unwrap() as f32;
            let power = frame[b].norm();
            let diff = freq - centroid;
            var += diff * diff * power;
        }

        let bw = if den > 0.0 { (var / den).sqrt() } else { 0.0 };
        values.push(bw);
    }

    SpectralBandwidth { n_frames, values }
}
