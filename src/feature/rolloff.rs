//! Spectral rolloff.

use crate::spectrum::Spectrogram;
use crate::utils::parallel::auto_map;

// Use macro to generate time-series feature struct
time_series_feature!(SpectralRolloff);

/// Compute spectral rolloff (Hz) for each frame.
///
/// `roll_percent` should be in (0, 1], e.g. 0.85 for 85%.
pub fn spectral_rolloff(spec: &Spectrogram, roll_percent: f32) -> SpectralRolloff {
    assert!(
        roll_percent > 0.0 && roll_percent <= 1.0,
        "roll_percent must be in (0, 1]"
    );

    let n_frames = spec.n_frames();
    let n_bins = spec.n_bins();

    let values = auto_map(n_frames, |t| {
        let frame = spec.frame(t).expect("frame index out of bounds");

        let total_energy: f32 = frame.iter().map(|c| c.norm()).sum();
        let threshold = total_energy * roll_percent;

        let mut cumulative = 0.0f32;
        let mut roll_freq = 0.0f32;

        for b in 0..n_bins {
            cumulative += frame[b].norm();
            if cumulative >= threshold {
                roll_freq = spec.bin_frequency_hz(b).unwrap() as f32;
                break;
            }
        }

        roll_freq
    });

    SpectralRolloff::new(values)
}
