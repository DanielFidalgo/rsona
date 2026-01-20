//! Onset strength (onset envelope) computation.
//!
//! This computes onset strength using:
//! - Mel spectrogram
//! - Log compression
//! - Positive spectral differences
//! - Optional temporal smoothing

use crate::spectrum::power_to_db;
use crate::spectrum::{DbConfig, MelConfig, MelSpectrogram, Spectrogram, mel_spectrogram};
use rayon::prelude::*;

/// Configuration for onset strength computation.
#[derive(Debug, Clone)]
pub struct OnsetConfig {
    /// Mel spectrogram configuration.
    pub mel: MelConfig,

    /// dB conversion configuration.
    pub db: DbConfig,

    /// Apply moving-average smoothing (window size in frames).
    ///
    /// Standard implementations use a small smoothing window.
    pub smooth: Option<usize>,
}

impl Default for OnsetConfig {
    fn default() -> Self {
        Self {
            mel: MelConfig {
                n_mels: 128,
                ..Default::default()
            },
            db: DbConfig::default(),
            smooth: Some(3),
        }
    }
}

// Use macro to generate time-series feature struct
time_series_feature!(OnsetEnvelope);

/// Compute onset strength envelope from a complex spectrogram.
///
/// Standard onset strength computation:
/// - mel spectrogram
/// - log / dB scaling
/// - positive spectral flux
/// - sum across mel bands
/// - optional smoothing
pub fn onset_strength(spec: &Spectrogram, cfg: OnsetConfig) -> OnsetEnvelope {
    // 1) Mel spectrogram (power)
    let mel = mel_spectrogram(spec, cfg.mel.clone());

    // 2) Compute onset from mel spectrogram
    let mut envelope = onset_strength_from_mel(&mel, cfg.db);

    // 3) Apply centering shift to match standard center=True behavior
    // Shift by n_fft / (2 * hop_length) frames to align with centered STFT
    let n_fft = spec.n_fft();
    let hop_size = spec.hop_size();
    let shift = n_fft / (2 * hop_size);

    if shift > 0 && shift < envelope.n_frames {
        // Prepend `shift` zeros and trim the end
        let mut shifted_values = vec![0.0f32; shift];
        shifted_values.extend_from_slice(&envelope.values[..envelope.n_frames - shift]);
        envelope.values = shifted_values;
    }

    envelope
}

/// Compute onset strength envelope from a pre-computed mel spectrogram.
///
/// This is more efficient when you already have a mel spectrogram computed,
/// avoiding redundant computation.
///
/// Standard onset strength computation:
/// - log / dB scaling
/// - positive spectral flux
/// - sum across mel bands
/// - optional smoothing
pub fn onset_strength_from_mel(mel: &MelSpectrogram, db_cfg: DbConfig) -> OnsetEnvelope {
    let n_frames = mel.n_frames();
    let n_mels = mel.n_mels();

    // 1) Log / dB compression
    let mel_db = power_to_db(mel.as_slice(), db_cfg);

    // 3) Positive differences across frames with optimized computation
    let mut onset = vec![0.0f32; n_frames];

    if n_frames >= 2 {
        if n_frames > 100 {
            // Parallel path for large frame counts
            onset[1..]
                .par_iter_mut()
                .enumerate()
                .for_each(|(idx, onset_val)| {
                    let t = idx + 1;
                    let cur = &mel_db[t * n_mels..(t + 1) * n_mels];
                    let prev = &mel_db[(t - 1) * n_mels..t * n_mels];

                    *onset_val = compute_positive_diff_sum(cur, prev) / n_mels as f32;
                });
        } else {
            // Sequential path for smaller frame counts
            for t in 1..n_frames {
                let cur = &mel_db[t * n_mels..(t + 1) * n_mels];
                let prev = &mel_db[(t - 1) * n_mels..t * n_mels];

                onset[t] = compute_positive_diff_sum(cur, prev) / n_mels as f32;
            }
        }
    }

    OnsetEnvelope::new(onset)
}

/// Compute sum of positive differences with vectorization-friendly code
#[inline(always)]
fn compute_positive_diff_sum(cur: &[f32], prev: &[f32]) -> f32 {
    debug_assert_eq!(cur.len(), prev.len());

    let len = cur.len();
    let mut acc = 0.0f32;

    // Process in chunks of 8 for better SIMD
    const CHUNK: usize = 8;
    let main_chunks = len / CHUNK;
    let remainder = len % CHUNK;

    // Main loop with manual unrolling
    for chunk_idx in 0..main_chunks {
        let base = chunk_idx * CHUNK;

        // Unrolled positive difference accumulation
        let d0 = cur[base] - prev[base];
        let d1 = cur[base + 1] - prev[base + 1];
        let d2 = cur[base + 2] - prev[base + 2];
        let d3 = cur[base + 3] - prev[base + 3];
        let d4 = cur[base + 4] - prev[base + 4];
        let d5 = cur[base + 5] - prev[base + 5];
        let d6 = cur[base + 6] - prev[base + 6];
        let d7 = cur[base + 7] - prev[base + 7];

        acc += if d0 > 0.0 { d0 } else { 0.0 };
        acc += if d1 > 0.0 { d1 } else { 0.0 };
        acc += if d2 > 0.0 { d2 } else { 0.0 };
        acc += if d3 > 0.0 { d3 } else { 0.0 };
        acc += if d4 > 0.0 { d4 } else { 0.0 };
        acc += if d5 > 0.0 { d5 } else { 0.0 };
        acc += if d6 > 0.0 { d6 } else { 0.0 };
        acc += if d7 > 0.0 { d7 } else { 0.0 };
    }

    // Handle remainder
    let remainder_start = main_chunks * CHUNK;
    for i in 0..remainder {
        let idx = remainder_start + i;
        let diff = cur[idx] - prev[idx];
        if diff > 0.0 {
            acc += diff;
        }
    }

    acc
}
