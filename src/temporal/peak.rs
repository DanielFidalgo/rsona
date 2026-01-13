//! Generic peak picking for 1D curves.
//!
//! This implementation is inspired by common MIR practice and librosa-style
//! peak picking, but is fully generic and reusable.

/// Configuration for peak picking.
#[derive(Debug, Clone)]
pub struct PeakPickingConfig {
    /// Minimum height for a peak.
    ///
    /// Peaks below this absolute value are ignored.
    pub min_height: f32,

    /// Minimum height relative to the maximum value in the curve.
    ///
    /// Effective threshold = max(min_height, max(curve) * min_height_rel).
    pub min_height_rel: f32,

    /// Number of frames on each side that must be lower than the peak.
    ///
    /// Typical values: 1–5.
    pub local_max_radius: usize,

    /// Minimum distance (in frames) between consecutive peaks.
    ///
    /// Enforces sparsity.
    pub min_distance: usize,

    /// Optional moving-average smoothing window (frames).
    pub smooth: Option<usize>,
}

impl Default for PeakPickingConfig {
    fn default() -> Self {
        Self {
            min_height: 0.0,
            min_height_rel: 0.25,
            local_max_radius: 1,
            min_distance: 1,
            smooth: None,
        }
    }
}

/// A detected peak.
#[derive(Debug, Clone)]
pub struct Peak {
    /// Frame index of the peak.
    pub index: usize,
    /// Value of the curve at the peak.
    pub value: f32,
}

/// Collection of peaks.
#[derive(Debug, Clone)]
pub struct Peaks {
    values: Vec<Peak>,
}

impl Peaks {
    /// Length of the peak collection.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if the peak collection is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get the peak collection as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[Peak] {
        &self.values
    }

    /// Get the indices of the peaks.
    #[inline]
    pub fn indices(&self) -> Vec<usize> {
        self.values.iter().map(|p| p.index).collect()
    }
}

/// Pick peaks from a 1D curve.
///
/// A peak must:
/// - exceed the height threshold
/// - be a strict local maximum within `local_max_radius`
/// - respect `min_distance` from previous accepted peaks
pub fn peak_pick(curve: &[f32], cfg: PeakPickingConfig) -> Peaks {
    let n = curve.len();
    if n == 0 {
        return Peaks { values: Vec::new() };
    }

    // 1) Optional smoothing
    let data = if let Some(win) = cfg.smooth {
        if win > 1 {
            moving_average(curve, win)
        } else {
            curve.to_vec()
        }
    } else {
        curve.to_vec()
    };

    // 2) Compute threshold
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let height_thresh = cfg.min_height.max(max_val * cfg.min_height_rel);

    // 3) Candidate peak detection (local maxima)
    let mut candidates: Vec<Peak> = Vec::new();

    for i in 0..n {
        let v = data[i];
        if v < height_thresh {
            continue;
        }

        let start = i.saturating_sub(cfg.local_max_radius);
        let end = (i + cfg.local_max_radius + 1).min(n);

        let mut is_peak = true;
        for j in start..end {
            if j == i {
                continue;
            }
            if data[j] >= v {
                is_peak = false;
                break;
            }
        }

        if is_peak {
            candidates.push(Peak { index: i, value: v });
        }
    }

    // 4) Enforce minimum distance between peaks
    // Sort by descending peak height, then greedily select
    candidates.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap());

    let mut selected: Vec<Peak> = Vec::new();

    'outer: for peak in candidates {
        for sel in &selected {
            let dist = if peak.index > sel.index {
                peak.index - sel.index
            } else {
                sel.index - peak.index
            };
            if dist < cfg.min_distance {
                continue 'outer;
            }
        }
        selected.push(peak);
    }

    // Sort final peaks by time
    selected.sort_by_key(|p| p.index);

    Peaks { values: selected }
}

// --- helpers ---

fn moving_average(x: &[f32], win: usize) -> Vec<f32> {
    let n = x.len();
    if win <= 1 || n == 0 {
        return x.to_vec();
    }

    let half = win / 2;
    let mut out = vec![0.0f32; n];

    for i in 0..n {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(n);

        let mut sum = 0.0f32;
        for v in &x[start..end] {
            sum += *v;
        }
        out[i] = sum / (end - start) as f32;
    }

    out
}
