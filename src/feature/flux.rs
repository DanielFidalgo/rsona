//! Spectral flux and onset strength.
//! Spectral flux.

use crate::spectrum::Spectrogram;
use rayon::prelude::*;

// Use macro to generate time-series feature struct
time_series_feature!(SpectralFlux);

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

    // Precompute ALL power spectra in parallel for better performance
    let all_power: Vec<Vec<f32>> = if n_frames > 10 {
        (0..n_frames)
            .into_par_iter()
            .map(|frame_index| {
                let frame = spec.frame(frame_index).expect("frame index out of bounds");
                let mut power = vec![0.0f32; n_bins];
                for bin_idx in 0..n_bins {
                    power[bin_idx] = frame[bin_idx].norm_sqr();
                }
                power
            })
            .collect()
    } else {
        (0..n_frames)
            .map(|frame_index| {
                let frame = spec.frame(frame_index).expect("frame index out of bounds");
                let mut power = vec![0.0f32; n_bins];
                for bin_idx in 0..n_bins {
                    power[bin_idx] = frame[bin_idx].norm_sqr();
                }
                power
            })
            .collect()
    };

    // Compute flux differences in parallel
    if n_frames > 10 {
        values[1..]
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, flux_val)| {
                let frame_index = idx + 1;
                let curr_power = &all_power[frame_index];
                let prev_power = &all_power[frame_index - 1];

                let mut flux = 0.0f32;
                for bin_idx in 0..n_bins {
                    let diff = curr_power[bin_idx] - prev_power[bin_idx];
                    if diff > 0.0 {
                        flux += diff;
                    }
                }
                *flux_val = flux;
            });
    } else {
        for frame_index in 1..n_frames {
            let curr_power = &all_power[frame_index];
            let prev_power = &all_power[frame_index - 1];

            let mut flux = 0.0f32;
            for bin_idx in 0..n_bins {
                let diff = curr_power[bin_idx] - prev_power[bin_idx];
                if diff > 0.0 {
                    flux += diff;
                }
            }
            values[frame_index] = flux;
        }
    }

    SpectralFlux::new(values)
}
