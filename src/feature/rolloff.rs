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

    let values = auto_map(n_frames, |frame_index| {
        let frame = spec.frame(frame_index).expect("frame index out of bounds");

        let total_energy: f32 = frame.iter().map(|complex_bin| complex_bin.norm()).sum();
        let threshold = total_energy * roll_percent;

        let mut cumulative = 0.0f32;
        let mut roll_freq = 0.0f32;

        for bin_idx in 0..n_bins {
            cumulative += frame[bin_idx].norm();
            if cumulative >= threshold {
                roll_freq = spec.bin_frequency_hz(bin_idx).unwrap() as f32;
                break;
            }
        }

        roll_freq
    });

    SpectralRolloff::new(values)
}
