//! Spectral bandwidth.

use crate::spectrum::Spectrogram;
use crate::utils::parallel::auto_map;

// Use macro to generate time-series feature struct
time_series_feature!(SpectralBandwidth);

/// Compute spectral bandwidth (Hz) for each frame.
///
/// Bandwidth = sqrt( sum((f - centroid)^2 * |X(f)|^2) / sum(|X(f)|^2) )
pub fn spectral_bandwidth(spec: &Spectrogram) -> SpectralBandwidth {
    let n_frames = spec.n_frames();
    let n_bins = spec.n_bins();

    let values = auto_map(n_frames, |frame_index| {
        let frame = spec.frame(frame_index).expect("frame index out of bounds");

        // First pass: centroid
        let mut numerator = 0.0f32;
        let mut denominator = 0.0f32;

        for bin_idx in 0..n_bins {
            let freq = spec.bin_frequency_hz(bin_idx).unwrap() as f32;
            let power = frame[bin_idx].norm();

            numerator += freq * power;
            denominator += power;
        }

        let centroid = if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        };

        // Second pass: variance
        let mut variance = 0.0f32;
        for bin_idx in 0..n_bins {
            let freq = spec.bin_frequency_hz(bin_idx).unwrap() as f32;
            let power = frame[bin_idx].norm();
            let diff = freq - centroid;
            variance += diff * diff * power;
        }

        if denominator > 0.0 {
            (variance / denominator).sqrt()
        } else {
            0.0
        }
    });

    SpectralBandwidth::new(values)
}
