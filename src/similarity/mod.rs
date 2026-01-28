//! Frame-to-frame similarity analysis (self-similarity matrices).
mod lag;
mod novelty;
mod repetition;
mod self_similarity;

pub use lag::{
    LagEnergy, LagEnergyConfig, RepeatLagEstimate, RepeatPhaseConfig, best_repeat_phase,
    diagonal_lag_energy, estimate_repeat_lag,
};
pub use novelty::{KernelWindow, NoveltyConfig, NoveltyCurve, novelty_curve};
pub use repetition::{RepetitionAggregation, RepetitionConfig, RepetitionCurve, repetition_curve};
pub use self_similarity::{
    FrameFeatures, SelfSimilarity, SelfSimilarityConfig, SimilarityMetric, self_similarity,
};
