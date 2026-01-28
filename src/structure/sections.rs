//! Section boundary detection from novelty curves.

use crate::similarity::{NoveltyConfig, SelfSimilarity, novelty_curve};
use crate::temporal::{PeakPickingConfig, peak_pick};

/// Section boundary detection from novelty curves.
#[derive(Debug, Clone)]
pub struct SectionBoundaryConfig {
    /// Novelty curve configuration (kernel window etc.)
    pub novelty: NoveltyConfig,

    /// Peak picking configuration
    pub peaks: PeakPickingConfig,

    /// Drop boundaries within this many frames of start/end.
    pub guard_frames: usize,
}

impl Default for SectionBoundaryConfig {
    fn default() -> Self {
        Self {
            novelty: NoveltyConfig::default(),
            peaks: PeakPickingConfig {
                min_height_rel: 0.30,
                local_max_radius: 2,
                min_distance: 16,
                smooth: None,
                ..Default::default()
            },
            guard_frames: 8,
        }
    }
}

/// Section boundary detection from novelty curves.
#[derive(Debug, Clone)]
pub struct SectionBoundaries {
    /// Boundary frame indices (sorted).
    pub boundary_frames: Vec<usize>,
    /// Novelty curve for debugging/visualization.
    pub novelty: Vec<f32>,
}

/// Compute section boundaries from an SSM:
/// SSM → novelty curve → peak picking → boundary frames.
///
/// Returns boundary frame indices (e.g., for cutting segments).
pub fn section_boundaries_from_novelty(
    ssm: &SelfSimilarity,
    cfg: SectionBoundaryConfig,
) -> SectionBoundaries {
    let nov = novelty_curve(ssm, cfg.novelty);
    let n = nov.n_frames();

    let peaks = peak_pick(nov.values(), cfg.peaks);
    let mut idxs = peaks.indices();

    // Guard edges
    idxs.retain(|&i| i >= cfg.guard_frames && i + cfg.guard_frames < n);

    SectionBoundaries {
        boundary_frames: idxs,
        novelty: nov.values().to_vec(),
    }
}
