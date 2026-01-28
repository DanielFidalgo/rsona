/// Beat tracking from an onset envelope and estimated period.
#[derive(Debug, Clone)]
pub struct BeatConfig {
    /// If true, snap each predicted beat to nearest local maximum in onset envelope.
    pub snap_to_local_max: bool,

    /// When snapping, search +/- this many frames around each beat.
    pub snap_radius_frames: usize,

    /// Minimum onset value to allow snapping.
    pub snap_min_rel: f32,
}

impl Default for BeatConfig {
    fn default() -> Self {
        Self {
            snap_to_local_max: true,
            snap_radius_frames: 2,
            snap_min_rel: 0.15,
        }
    }
}

/// Beat tracking from an onset envelope and estimated period.
#[derive(Debug, Clone)]
pub struct BeatTrack {
    /// Beat positions as frame indices.
    pub beat_frames: Vec<usize>,
    /// Beat positions in seconds.
    pub beat_times: Vec<f64>,
    /// Chosen phase offset in frames.
    pub phase_offset: usize,
}

/// Track beats using a fixed period (frames per beat).
///
/// Algorithm (v1):
/// 1) Find best phase offset that maximizes onset strength on the grid
/// 2) Emit beats at offset + k * period
/// 3) Optionally snap each beat to nearest local maximum in onset envelope
pub fn track_beats(
    onset_env: &[f32],
    sample_rate: u32,
    hop_size: usize,
    period_frames: usize,
    cfg: BeatConfig,
) -> BeatTrack {
    let num_frames = onset_env.len();
    if num_frames == 0 || period_frames == 0 {
        return BeatTrack {
            beat_frames: Vec::new(),
            beat_times: Vec::new(),
            phase_offset: 0,
        };
    }

    // 1) Choose phase offset
    let phase_offset = best_phase(onset_env, period_frames);

    // 2) Generate beat grid
    let mut beats = Vec::new();
    let mut beat_position = phase_offset;
    while beat_position < num_frames {
        beats.push(beat_position);
        beat_position += period_frames;
    }

    // 3) Optional snapping
    if cfg.snap_to_local_max && !beats.is_empty() {
        let max_onset = onset_env.iter().cloned().fold(0.0f32, f32::max);
        let min_thresh = max_onset * cfg.snap_min_rel;

        for beat in &mut beats {
            if let Some(snapped) =
                snap_local_max(onset_env, *beat, cfg.snap_radius_frames, min_thresh)
            {
                *beat = snapped;
            }
        }

        // Dedup and re-sort after snapping
        beats.sort_unstable();
        beats.dedup();
    }

    // 4) Convert to times
    let sample_rate_f64 = sample_rate as f64;
    let hop = hop_size as f64;
    let beat_times: Vec<f64> = beats
        .iter()
        .map(|&f| (f as f64 * hop) / sample_rate_f64)
        .collect();

    BeatTrack {
        beat_frames: beats,
        beat_times,
        phase_offset,
    }
}

fn best_phase(env: &[f32], period: usize) -> usize {
    let num_frames = env.len();
    let mut best_offset = 0usize;
    let mut best_score = f32::NEG_INFINITY;

    for offset in 0..period.min(num_frames) {
        let mut sum = 0.0f32;
        let mut beat_position = offset;
        while beat_position < num_frames {
            sum += env[beat_position];
            beat_position += period;
        }
        if sum > best_score {
            best_score = sum;
            best_offset = offset;
        }
    }

    best_offset
}

/// Snap to nearest local maximum within +/- radius where env >= min_thresh.
fn snap_local_max(env: &[f32], idx: usize, radius: usize, min_thresh: f32) -> Option<usize> {
    let num_frames = env.len();
    if num_frames == 0 {
        return None;
    }
    let start = idx.saturating_sub(radius);
    let end = (idx + radius + 1).min(num_frames);

    let mut best: Option<(usize, f32)> = None; // (index, value)

    for i in start..end {
        let value = env[i];
        if value < min_thresh {
            continue;
        }
        let left = if i == 0 { value } else { env[i - 1] };
        let right = if i + 1 >= num_frames {
            value
        } else {
            env[i + 1]
        };
        if value >= left && value >= right {
            match best {
                None => best = Some((i, value)),
                Some((_, best_v)) if value > best_v => best = Some((i, value)),
                _ => {}
            }
        }
    }

    best.map(|(i, _)| i)
}
