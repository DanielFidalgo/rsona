//! Decibel (dB) conversions for spectral data.
//!
//! This module provides standard conversions from power or amplitude
//! spectrograms to decibel scale.

use rayon::prelude::*;

/// Configuration for decibel conversion.
///
/// Standard defaults:
/// - `ref = 1.0`
/// - `amin = 1e-10`
/// - `top_db = Some(80.0)`
#[derive(Debug, Clone)]
pub struct DbConfig {
    /// Reference value. Usually 1.0 or the maximum of the input.
    pub ref_value: f32,

    /// Minimum amplitude/power to clamp to before taking log.
    pub amin: f32,

    /// Maximum dynamic range in dB.
    ///
    /// If set, values more than `top_db` below the maximum
    /// will be clipped.
    pub top_db: Option<f32>,
}

impl Default for DbConfig {
    fn default() -> Self {
        Self {
            ref_value: 1.0,
            amin: 1e-10,
            top_db: Some(80.0),
        }
    }
}

/// Convert a power spectrogram to decibel (dB) scale.
///
/// Formula:
/// ```text
/// S_db = 10 * log10(max(S, amin) / ref)
/// ```
///
/// If `top_db` is set, values are clipped to:
/// ```text
/// max(S_db) - top_db
/// ```
pub fn power_to_db(input: &[f32], cfg: DbConfig) -> Vec<f32> {
    assert!(cfg.ref_value > 0.0, "ref_value must be > 0");
    assert!(cfg.amin > 0.0, "amin must be > 0");

    let n = input.len();
    if n == 0 {
        return Vec::new();
    }

    // Precompute constant factor
    let log_ref = cfg.ref_value.log10();
    let const_factor = -10.0 * log_ref;

    let mut out = if n > 1000 {
        // Parallel path for large arrays
        input
            .par_iter()
            .map(|&v| {
                let clamped = v.max(cfg.amin);
                10.0 * clamped.log10() + const_factor
            })
            .collect()
    } else {
        // Sequential path with manual unrolling for better SIMD
        let mut out = Vec::with_capacity(n);

        const CHUNK: usize = 8;
        let main_chunks = n / CHUNK;
        let _remainder = n % CHUNK;

        // Main loop - unrolled for SIMD
        for chunk_idx in 0..main_chunks {
            let base = chunk_idx * CHUNK;

            let v0 = input[base].max(cfg.amin);
            let v1 = input[base + 1].max(cfg.amin);
            let v2 = input[base + 2].max(cfg.amin);
            let v3 = input[base + 3].max(cfg.amin);
            let v4 = input[base + 4].max(cfg.amin);
            let v5 = input[base + 5].max(cfg.amin);
            let v6 = input[base + 6].max(cfg.amin);
            let v7 = input[base + 7].max(cfg.amin);

            out.push(10.0 * v0.log10() + const_factor);
            out.push(10.0 * v1.log10() + const_factor);
            out.push(10.0 * v2.log10() + const_factor);
            out.push(10.0 * v3.log10() + const_factor);
            out.push(10.0 * v4.log10() + const_factor);
            out.push(10.0 * v5.log10() + const_factor);
            out.push(10.0 * v6.log10() + const_factor);
            out.push(10.0 * v7.log10() + const_factor);
        }

        // Handle remainder
        for idx in (main_chunks * CHUNK)..n {
            let v = input[idx].max(cfg.amin);
            out.push(10.0 * v.log10() + const_factor);
        }

        out
    };

    // Optional dynamic range clipping.
    if let Some(top_db) = cfg.top_db {
        // Find max in parallel if large enough
        let max_db = if n > 1000 {
            out.par_iter()
                .copied()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(f32::NEG_INFINITY)
        } else {
            out.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        };

        let min_db = max_db - top_db;

        // Clipping with vectorization hints
        if n > 1000 {
            out.par_iter_mut().for_each(|v| {
                if *v < min_db {
                    *v = min_db;
                }
            });
        } else {
            const CHUNK: usize = 8;
            let main_chunks = n / CHUNK;
            let _remainder = n % CHUNK;

            for chunk_idx in 0..main_chunks {
                let base = chunk_idx * CHUNK;

                out[base] = out[base].max(min_db);
                out[base + 1] = out[base + 1].max(min_db);
                out[base + 2] = out[base + 2].max(min_db);
                out[base + 3] = out[base + 3].max(min_db);
                out[base + 4] = out[base + 4].max(min_db);
                out[base + 5] = out[base + 5].max(min_db);
                out[base + 6] = out[base + 6].max(min_db);
                out[base + 7] = out[base + 7].max(min_db);
            }

            for idx in (main_chunks * CHUNK)..n {
                out[idx] = out[idx].max(min_db);
            }
        }
    }

    out
}

/// Convert an amplitude spectrogram to decibel (dB) scale.
///
/// Formula:
/// ```text
/// A_db = 20 * log10(max(A, amin) / ref)
/// ```
pub fn amplitude_to_db(input: &[f32], cfg: DbConfig) -> Vec<f32> {
    assert!(cfg.ref_value > 0.0, "ref_value must be > 0");
    assert!(cfg.amin > 0.0, "amin must be > 0");

    let n = input.len();
    if n == 0 {
        return Vec::new();
    }

    // Precompute constant factor
    let log_ref = cfg.ref_value.log10();
    let const_factor = -20.0 * log_ref;

    let mut out = if n > 1000 {
        // Parallel path for large arrays
        input
            .par_iter()
            .map(|&v| {
                let clamped = v.max(cfg.amin);
                20.0 * clamped.log10() + const_factor
            })
            .collect()
    } else {
        // Sequential path with manual unrolling
        let mut out = Vec::with_capacity(n);

        const CHUNK: usize = 8;
        let main_chunks = n / CHUNK;
        let _remainder = n % CHUNK;

        for chunk_idx in 0..main_chunks {
            let base = chunk_idx * CHUNK;

            let v0 = input[base].max(cfg.amin);
            let v1 = input[base + 1].max(cfg.amin);
            let v2 = input[base + 2].max(cfg.amin);
            let v3 = input[base + 3].max(cfg.amin);
            let v4 = input[base + 4].max(cfg.amin);
            let v5 = input[base + 5].max(cfg.amin);
            let v6 = input[base + 6].max(cfg.amin);
            let v7 = input[base + 7].max(cfg.amin);

            out.push(20.0 * v0.log10() + const_factor);
            out.push(20.0 * v1.log10() + const_factor);
            out.push(20.0 * v2.log10() + const_factor);
            out.push(20.0 * v3.log10() + const_factor);
            out.push(20.0 * v4.log10() + const_factor);
            out.push(20.0 * v5.log10() + const_factor);
            out.push(20.0 * v6.log10() + const_factor);
            out.push(20.0 * v7.log10() + const_factor);
        }

        for idx in (main_chunks * CHUNK)..n {
            let v = input[idx].max(cfg.amin);
            out.push(20.0 * v.log10() + const_factor);
        }

        out
    };

    // Optional dynamic range clipping
    if let Some(top_db) = cfg.top_db {
        let max_db = if n > 1000 {
            out.par_iter()
                .copied()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(f32::NEG_INFINITY)
        } else {
            out.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        };

        let min_db = max_db - top_db;

        if n > 1000 {
            out.par_iter_mut().for_each(|v| {
                *v = (*v).max(min_db);
            });
        } else {
            const CHUNK: usize = 8;
            let main_chunks = n / CHUNK;

            for chunk_idx in 0..main_chunks {
                let base = chunk_idx * CHUNK;

                out[base] = out[base].max(min_db);
                out[base + 1] = out[base + 1].max(min_db);
                out[base + 2] = out[base + 2].max(min_db);
                out[base + 3] = out[base + 3].max(min_db);
                out[base + 4] = out[base + 4].max(min_db);
                out[base + 5] = out[base + 5].max(min_db);
                out[base + 6] = out[base + 6].max(min_db);
                out[base + 7] = out[base + 7].max(min_db);
            }

            for idx in (main_chunks * CHUNK)..n {
                out[idx] = out[idx].max(min_db);
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_to_db_basic() {
        let power = vec![1.0, 0.1, 0.01];
        let db = power_to_db(&power, DbConfig::default());

        // 10 * log10([1, 0.1, 0.01]) = [0, -10, -20]
        assert!((db[0] - 0.0).abs() < 1e-6);
        assert!((db[1] + 10.0).abs() < 1e-6);
        assert!((db[2] + 20.0).abs() < 1e-6);
    }

    #[test]
    fn amplitude_to_db_basic() {
        let amp = vec![1.0, 0.1];
        let db = amplitude_to_db(&amp, DbConfig::default());

        // 20 * log10([1, 0.1]) = [0, -20]
        assert!((db[0] - 0.0).abs() < 1e-6);
        assert!((db[1] + 20.0).abs() < 1e-6);
    }

    #[test]
    fn top_db_clipping() {
        let power = vec![1.0, 1e-10];
        let cfg = DbConfig {
            ref_value: 1.0,
            amin: 1e-10,
            top_db: Some(40.0),
        };

        let db = power_to_db(&power, cfg);

        // max = 0 dB, min clipped to -40 dB
        assert!((db[1] + 40.0).abs() < 1e-6);
    }
}
