use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

/// Tempo estimation from an onset strength envelope using autocorrelation.
#[derive(Debug, Clone)]
pub struct TempoConfig {
    /// Minimum BPM to consider.
    pub min_bpm: f32,
    /// Maximum BPM to consider.
    pub max_bpm: f32,
    /// Optional smoothing window applied to onset envelope (frames).
    pub smooth: Option<usize>,
    /// If true, normalize autocorrelation by energy (roughly scale-invariant).
    pub normalize_acf: bool,
    /// Starting BPM for prior (center of Gaussian).
    pub start_bpm: f32,
    /// Standard deviation for tempo prior (in BPM).
    pub std_bpm: f32,
}

impl Default for TempoConfig {
    fn default() -> Self {
        Self {
            min_bpm: 60.0,
            max_bpm: 320.0,
            smooth: Some(3),
            normalize_acf: true,
            start_bpm: 120.0,
            std_bpm: 40.0,
        }
    }
}

/// Tempo estimation from an onset strength envelope using autocorrelation.
#[derive(Debug, Clone)]
pub struct TempoEstimate {
    /// Estimated tempo in beats per minute.
    pub bpm: f32,
    /// Period (frames per beat).
    pub period_frames: usize,
    /// Autocorrelation values for lags 0..=max_lag (for debugging/visualization).
    pub acf: Vec<f32>,
    /// Lag chosen (frames).
    pub best_lag: usize,
}

/// Estimate tempo from an onset envelope.
///
/// Inputs:
/// - `onset_env`: onset strength per frame
/// - `sample_rate`: audio sample rate (Hz)
/// - `hop_size`: hop size in samples (per frame)
pub fn estimate_tempo(
    onset_env: &[f32],
    sample_rate: u32,
    hop_size: usize,
    cfg: TempoConfig,
) -> TempoEstimate {
    let n = onset_env.len();
    if n == 0 {
        return TempoEstimate {
            bpm: 0.0,
            period_frames: 0,
            acf: vec![0.0],
            best_lag: 0,
        };
    }

    let env = if let Some(win) = cfg.smooth {
        if win > 1 {
            moving_average(onset_env, win)
        } else {
            onset_env.to_vec()
        }
    } else {
        onset_env.to_vec()
    };

    // Convert BPM range to lag range in frames:
    // bpm = 60 * sr / (hop * lag)
    // lag = 60 * sr / (hop * bpm)
    let sr = sample_rate as f32;
    let hop = hop_size as f32;

    let max_lag = ((60.0 * sr) / (hop * cfg.min_bpm)).round().max(1.0) as usize;
    let min_lag = ((60.0 * sr) / (hop * cfg.max_bpm)).round().max(1.0) as usize;

    // Use FFT-based ACF for better performance
    let acf = compute_acf_fft(&env, max_lag, cfg.normalize_acf);

    // Find best lag within [min_lag, max_lag] with tempo prior
    let best_lag =
        find_best_tempo_with_prior(&acf, min_lag, max_lag, sr, hop, cfg.start_bpm, cfg.std_bpm);

    let bpm = if best_lag > 0 {
        (60.0 * sr) / (hop * best_lag as f32)
    } else {
        0.0
    };

    TempoEstimate {
        bpm,
        period_frames: best_lag,
        acf,
        best_lag,
    }
}

/// FFT-based autocorrelation computation (much faster than naive approach)
///
/// This uses the Wiener-Khinchin theorem: autocorrelation = IFFT(|FFT(x)|^2)
#[inline]
fn compute_acf_fft(env: &[f32], max_lag: usize, normalize: bool) -> Vec<f32> {
    let n = env.len();

    // For FFT-based ACF, we need to zero-pad to at least 2*n to avoid circular correlation
    let fft_size = (2 * n).next_power_of_two();

    // Prepare input for FFT
    let mut fft_input = vec![Complex::new(0.0f32, 0.0f32); fft_size];
    for i in 0..n {
        fft_input[i] = Complex::new(env[i], 0.0);
    }

    // Forward FFT
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_size);
    fft.process(&mut fft_input);

    // Compute power spectrum: |X|^2
    for i in 0..fft_size {
        let mag_sq = fft_input[i].norm_sqr();
        fft_input[i] = Complex::new(mag_sq, 0.0);
    }

    // Inverse FFT to get autocorrelation
    let ifft = planner.plan_fft_inverse(fft_size);
    ifft.process(&mut fft_input);

    // Extract autocorrelation values (normalize by FFT size)
    let fft_norm = fft_size as f32;
    let mut acf = vec![0.0f32; max_lag + 1];

    if normalize {
        // Normalize by energy and lag-dependent sample count
        let energy: f32 = env.iter().map(|v| v * v).sum::<f32>().max(1e-12);

        for lag in 0..=max_lag.min(n - 1) {
            let valid_len = n - lag;
            let raw_acf = fft_input[lag].re / fft_norm;
            acf[lag] = raw_acf / (energy * valid_len as f32).sqrt();
        }
    } else {
        for lag in 0..=max_lag.min(n - 1) {
            let valid_len = n - lag;
            acf[lag] = (fft_input[lag].re / fft_norm) / valid_len as f32;
        }
    }

    acf
}

/// Find best tempo using ACF with octave correction.
#[inline]
fn find_best_tempo_with_prior(
    acf: &[f32],
    min_lag: usize,
    max_lag: usize,
    sr: f32,
    hop: f32,
    start_bpm: f32,
    std_bpm: f32,
) -> usize {
    if min_lag > max_lag || max_lag >= acf.len() {
        return min_lag.min(max_lag).min(acf.len() - 1);
    }

    // First, find the strongest ACF peak (ignoring prior)
    let mut raw_best_lag = min_lag;
    let mut raw_best_val = f32::NEG_INFINITY;

    for lag in min_lag..=max_lag {
        if acf[lag] > raw_best_val {
            raw_best_val = acf[lag];
            raw_best_lag = lag;
        }
    }

    // Generate octave candidates: check multiples of the detected period
    // This handles the common case where ACF finds a sub-harmonic
    let mut candidates = Vec::with_capacity(10);

    // Check integer multiples: 1x, 2x, 3x, 4x
    for multiplier in 1..=4 {
        let candidate_lag = raw_best_lag * multiplier;
        if candidate_lag >= min_lag && candidate_lag <= max_lag {
            candidates.push(candidate_lag);
        }
    }

    // Also check sub-multiples: 0.5x, 0.33x
    for divisor in 2..=3 {
        let candidate_lag = raw_best_lag / divisor;
        if candidate_lag >= min_lag && candidate_lag <= max_lag && candidate_lag > 0 {
            candidates.push(candidate_lag);
        }
    }

    // If no candidates, fall back to raw detection
    if candidates.is_empty() {
        return raw_best_lag;
    }

    // Score each candidate by combining ACF strength with musical likelihood
    let mut best_candidate = raw_best_lag;
    let mut best_score = f32::NEG_INFINITY;

    // Precompute constants for Gaussian
    let inv_std_bpm = 1.0 / std_bpm;
    let gaussian_factor = -0.5 * inv_std_bpm * inv_std_bpm;

    for &candidate_lag in &candidates {
        if candidate_lag >= acf.len() {
            continue;
        }

        let candidate_acf = acf[candidate_lag];
        let candidate_bpm = (60.0 * sr) / (hop * candidate_lag as f32);

        // Musical likelihood: prefer tempos in typical range
        // Use a Gaussian centered on start_bpm with std_bpm
        let bpm_diff = candidate_bpm - start_bpm;
        let musical_likelihood = (gaussian_factor * bpm_diff * bpm_diff).exp();

        // Final score: Give strong weight to musical likelihood to avoid octave errors
        // When ACF values are close, musical likelihood should dominate
        let score = candidate_acf * (0.3 + 0.7 * musical_likelihood);

        if score > best_score {
            best_score = score;
            best_candidate = candidate_lag;
        }
    }

    // Explicit octave error correction:
    // If 2x the best lag has a similar ACF value (within 5%), prefer the longer period
    // BUT only if it brings us closer to the musical prior
    let double_lag = best_candidate * 2;
    if double_lag <= max_lag && double_lag < acf.len() {
        let best_acf = acf[best_candidate];
        let double_acf = acf[double_lag];

        // If the ACF at 2x lag is within 5% of the best, consider preferring it
        let acf_ratio = double_acf / best_acf.max(1e-10);
        if acf_ratio > 0.95 {
            // Compute BPMs for both candidates
            let best_bpm = (60.0 * sr) / (hop * best_candidate as f32);
            let double_bpm = (60.0 * sr) / (hop * double_lag as f32);

            // Only prefer the doubled lag if it's closer to the musical prior
            let best_diff = (best_bpm - start_bpm).abs();
            let double_diff = (double_bpm - start_bpm).abs();

            if double_diff < best_diff {
                best_candidate = double_lag;
            }
        }
    }

    best_candidate
}

/// Optimized moving average with reduced allocations
#[inline]
fn moving_average(x: &[f32], win: usize) -> Vec<f32> {
    let n = x.len();
    if win <= 1 || n == 0 {
        return x.to_vec();
    }

    let half = win / 2;
    let mut out = Vec::with_capacity(n);

    // Use a sliding window approach for better cache locality
    for i in 0..n {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(n);

        let mut sum = 0.0f32;

        // Manual unrolling for small windows
        let len = end - start;
        let chunk_size = 4;
        let num_chunks = len / chunk_size;

        for chunk_idx in 0..num_chunks {
            let base = start + chunk_idx * chunk_size;
            sum += x[base];
            sum += x[base + 1];
            sum += x[base + 2];
            sum += x[base + 3];
        }

        // Handle remainder
        for idx in (start + num_chunks * chunk_size)..end {
            sum += x[idx];
        }

        out.push(sum / len as f32);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimates_tempo_for_pulse_train() {
        // Synthetic onset envelope: pulses every 10 frames
        let period = 10usize;
        let mut env = vec![0.0f32; 300];
        for i in (0..env.len()).step_by(period) {
            env[i] = 1.0;
        }

        // Suppose sr=1000 Hz, hop=10 samples => frame_rate=100 fps
        // period=10 frames => 10/100 sec per beat => 0.1s => 600 BPM
        // We'll set range around that.
        let est = estimate_tempo(
            &env,
            1000,
            10,
            TempoConfig {
                min_bpm: 400.0,
                max_bpm: 800.0,
                ..Default::default()
            },
        );

        assert!((est.period_frames as i32 - 10).abs() <= 1);
        assert!(est.bpm > 0.0);
    }
}
