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

    let values = auto_map(n_frames, |t| {
        let frame = spec.frame(t).expect("frame index out of bounds");

        let mut num = 0.0f32;
        let mut den = 0.0f32;

        for b in 0..n_bins {
            let freq = spec.bin_frequency_hz(b).unwrap() as f32;
            let magnitude = frame[b].norm();

            num += freq * magnitude;
            den += magnitude;
        }

        if den > 0.0 { num / den } else { 0.0 }
    });

    SpectralCentroid::new(values)
}
