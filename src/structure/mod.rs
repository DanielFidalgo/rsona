//! Musical structure inference built on top of similarity + temporal primitives.

mod sections;
mod segmentation;

pub use sections::{SectionBoundaries, SectionBoundaryConfig, section_boundaries_from_novelty};
pub use segmentation::{
    Segment, SegmentKind, SegmentationConfig, SegmentationResult, segment_intro_loop_outro,
    segment_intro_loop_outro_with_onsets,
};
