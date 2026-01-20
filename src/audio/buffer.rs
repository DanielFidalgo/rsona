//! Time-domain audio buffer representation.
//!
//! This module defines the canonical audio container used throughout rsona.
//! It represents decoded PCM audio only. No resampling, mixing, windowing,
//! or frequency-domain processing occurs at this layer.

/// A decoded, time-domain audio buffer.
///
/// Samples are stored as interleaved `f32` values:
/// - Mono: `[M, M, M, ...]`
/// - Stereo: `[L, R, L, R, ...]`
///
/// This type represents raw PCM audio only.
/// All higher-level processing (framing, FFT, features, structure)
/// is built on top of this buffer.
///
/// ## Invariants
/// - `channels > 0`
/// - `samples.len() % channels == 0`
/// - Samples are normalized to `[-1.0, 1.0]` (as provided by the decoder)
#[derive(Debug)]
pub struct Buffer {
    /// Sample rate in Hz (e.g. 44100).
    pub sample_rate: u32,

    /// Number of audio channels.
    pub channels: usize,

    /// Interleaved PCM samples.
    pub samples: Vec<f32>,
}

impl Buffer {
    /// Create a new audio buffer.
    ///
    /// This constructor enforces basic invariants in debug builds.
    /// Callers are responsible for providing correctly decoded PCM data.
    #[inline]
    pub fn new(sample_rate: u32, channels: usize, samples: Vec<f32>) -> Self {
        debug_assert!(sample_rate > 0, "sample_rate must be > 0");
        debug_assert!(channels > 0, "channels must be > 0");
        debug_assert!(
            samples.len().is_multiple_of(channels),
            "samples length must be divisible by channels"
        );

        Self {
            sample_rate,
            channels,
            samples,
        }
    }

    /// Returns the total number of frames (samples per channel).
    ///
    /// For stereo audio, this is `samples.len() / 2`.
    #[inline]
    pub fn frames(&self) -> usize {
        self.samples.len() / self.channels
    }

    /// Returns the duration of the buffer in seconds.
    #[inline]
    pub fn duration_seconds(&self) -> f64 {
        self.frames() as f64 / self.sample_rate as f64
    }

    /// Returns `true` if the buffer contains no samples.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Returns the duration of a single frame in seconds.
    ///
    /// This is equivalent to `1.0 / sample_rate`.
    #[inline]
    pub fn frame_duration_seconds(&self) -> f64 {
        1.0 / self.sample_rate as f64
    }
}
