//! Spectral rolloff.

use crate::spectrum::Spectrogram;

/// Spectral rolloff result.
#[derive(Debug, Clone)]
pub struct SpectralRolloff {
    n_frames: usize,
    values: Vec<f32>,
}

impl SpectralRolloff {
    /// Frame count.
    #[inline]
    pub fn n_frames(&self) -> usize {
        self.n_frames
    }

    /// Spectral rolloff values.
    #[inline]
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

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

    let mut values = Vec::with_capacity(n_frames);

    for t in 0..n_frames {
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

        values.push(roll_freq);
    }

    SpectralRolloff { n_frames, values }
}
