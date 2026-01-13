//! Temporal analysis utilities (peak picking, tempo, beats).

mod beat;
mod peak;
mod sync;
mod tempo;

pub use beat::{BeatConfig, BeatTrack, track_beats};
pub use peak::{Peak, PeakPickingConfig, Peaks, peak_pick};
pub use sync::{
    AggregationMethod, SyncConfig, beat_frames_to_times, beat_times_to_frames,
    sync_series_to_beats, sync_to_beats, sync_to_intervals,
};
pub use tempo::{TempoConfig, TempoEstimate, estimate_tempo};
