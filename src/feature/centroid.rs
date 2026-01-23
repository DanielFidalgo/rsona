//! Spectral centroid.

use crate::spectrum::Spectrogram;
use crate::utils::parallel::auto_map;

// Use macro to generate time-series feature struct
time_series_feature!(SpectralCentroid);

/// Compute spectral centroid (Hz) for each frame.
///
/// Centroid = sum(f * |X(f)|) / sum(|X(f)|)
pub fn spectral_centroid(spec: &Spectrogram) -> SpectralCentroid {
    let n_frames = spec.n_frames();
    let n_bins = spec.n_bins();

    let values = auto_map(n_frames, |frame_index| {
        let frame = spec.frame(frame_index).expect("frame index out of bounds");

        let mut numerator = 0.0f32;
        let mut denominator = 0.0f32;

        for bin_idx in 0..n_bins {
            let freq = spec.bin_frequency_hz(bin_idx).unwrap() as f32;
            let magnitude = frame[bin_idx].norm();

            numerator += freq * magnitude;
            denominator += magnitude;
        }

        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    });

    SpectralCentroid::new(values)
}
