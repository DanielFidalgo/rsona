//! Audio processing pipeline helpers.
//!
//! This module provides convenience functions for common audio processing
//! workflows to reduce boilerplate in benchmarks and examples.
//!
//! # Examples
//!
//! Load audio with standard settings:
//!
//! ```no_run
//! use rsona::pipeline;
//!
//! let pipeline = pipeline::standard("audio.wav")?;
//! println!("Loaded {} frames", pipeline.frames.n_frames());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Load with custom configurations:
//!
//! ```no_run
//! use rsona::pipeline;
//! use rsona::signal::FrameConfig;
//! use rsona::spectrum::StftConfig;
//!
//! let pipeline = pipeline::load_and_analyze(
//!     "audio.wav",
//!     FrameConfig::music(),
//!     StftConfig::default(),
//! )?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::{audio, signal, spectrum};
use std::error::Error;

/// Standard audio processing pipeline components.
///
/// This struct holds the result of loading an audio file and computing
/// its frames and STFT, providing easy access to all intermediate
/// representations.
///
/// # Fields
///
/// - `buffer`: The original decoded audio buffer
/// - `frames`: Time-domain frames with windowing applied
/// - `spec`: Complex STFT spectrogram
pub struct Pipeline {
    /// Original audio buffer.
    pub buffer: audio::Buffer,

    /// Time-domain frames.
    pub frames: signal::Frames,

    /// STFT spectrogram.
    pub spec: spectrum::Spectrogram,
}

/// Load audio file and compute frames and STFT.
///
/// This is the primary entry point for custom pipeline configurations.
/// It performs the standard audio processing workflow:
///
/// 1. Load audio from file
/// 2. Frame the signal with the given configuration
/// 3. Compute STFT with the given configuration
///
/// # Arguments
///
/// * `path` - Path to audio file (WAV, MP3, FLAC, etc.)
/// * `frame_cfg` - Frame configuration (size, hop, windowing)
/// * `stft_cfg` - STFT configuration (FFT size)
///
/// # Returns
///
/// A `Pipeline` containing the buffer, frames, and spectrogram.
///
/// # Errors
///
/// Returns an error if:
/// - File cannot be loaded
/// - Audio format is unsupported
/// - Framing fails
/// - STFT computation fails
///
/// # Examples
///
/// ```no_run
/// use rsona::pipeline;
/// use rsona::signal::FrameConfig;
/// use rsona::spectrum::StftConfig;
///
/// let pipeline = pipeline::load_and_analyze(
///     "audio.wav",
///     FrameConfig::standard(),
///     StftConfig::default(),
/// )?;
///
/// println!("Sample rate: {} Hz", pipeline.buffer.sample_rate);
/// println!("Frames: {}", pipeline.frames.n_frames());
/// println!("Frequency bins: {}", pipeline.spec.n_bins());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn load_and_analyze(
    path: &str,
    frame_cfg: signal::FrameConfig,
    stft_cfg: spectrum::StftConfig,
) -> Result<Pipeline, Box<dyn Error>> {
    let buffer = audio::load(path)?;
    let frames = signal::frame(&buffer, frame_cfg)?;
    let spec = spectrum::stft(&frames, stft_cfg)?;

    Ok(Pipeline {
        buffer,
        frames,
        spec,
    })
}

/// Load audio with standard preset configurations.
///
/// This is the most convenient entry point for typical audio analysis.
/// It uses:
///
/// - Frame config: [`signal::FrameConfig::standard()`] (2048 frame size, 512 hop)
/// - STFT config: [`spectrum::StftConfig::default()`]
///
/// These settings provide a good balance between time and frequency
/// resolution for general audio analysis tasks.
///
/// # Arguments
///
/// * `path` - Path to audio file
///
/// # Examples
///
/// ```no_run
/// use rsona::pipeline;
/// use rsona::spectrum::mel_spectrogram;
/// use rsona::spectrum::MelConfig;
///
/// let pipeline = pipeline::standard("audio.wav")?;
/// let mel = mel_spectrogram(&pipeline.spec, MelConfig::default());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn standard(path: &str) -> Result<Pipeline, Box<dyn Error>> {
    load_and_analyze(
        path,
        signal::FrameConfig::standard(),
        spectrum::StftConfig::default(),
    )
}

/// Load audio for music analysis (higher resolution).
///
/// Uses higher resolution settings suitable for detailed harmonic and
/// spectral analysis:
///
/// - Frame config: [`signal::FrameConfig::music()`] (4096 frame size, 1024 hop)
/// - STFT config: 4096 FFT size
///
/// This provides better frequency resolution at the cost of lower
/// time resolution, which is typically acceptable for music analysis.
///
/// # Arguments
///
/// * `path` - Path to audio file
///
/// # Examples
///
/// ```no_run
/// use rsona::pipeline;
/// use rsona::feature::chroma_stft;
/// use rsona::feature::ChromaConfig;
///
/// let pipeline = pipeline::music("song.mp3")?;
/// let chroma = chroma_stft(&pipeline.spec, ChromaConfig::default());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn music(path: &str) -> Result<Pipeline, Box<dyn Error>> {
    load_and_analyze(
        path,
        signal::FrameConfig::music(),
        spectrum::StftConfig::with_frame_size(4096),
    )
}

/// Load audio for speech analysis (lower latency).
///
/// Uses lower latency settings suitable for real-time speech processing:
///
/// - Frame config: [`signal::FrameConfig::speech()`] (512 frame size, 256 hop)
/// - STFT config: 512 FFT size
///
/// This provides faster updates at the cost of lower frequency resolution,
/// which is typically acceptable for speech analysis.
///
/// # Arguments
///
/// * `path` - Path to audio file
///
/// # Examples
///
/// ```no_run
/// use rsona::pipeline;
/// use rsona::feature::mfcc;
/// use rsona::feature::MfccConfig;
/// use rsona::spectrum::mel_spectrogram;
/// use rsona::spectrum::MelConfig;
///
/// let pipeline = pipeline::speech("speech.wav")?;
/// let mel = mel_spectrogram(&pipeline.spec, MelConfig::default());
/// let mfcc_features = mfcc(&mel, MfccConfig::default());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn speech(path: &str) -> Result<Pipeline, Box<dyn Error>> {
    load_and_analyze(
        path,
        signal::FrameConfig::speech(),
        spectrum::StftConfig::with_frame_size(512),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require actual audio files and are marked as ignored.
    // Run with: cargo test -- --ignored

    #[test]
    #[ignore]
    fn test_standard_pipeline() {
        let result = standard("test_audio.wav");
        assert!(result.is_ok());

        let pipeline = result.unwrap();
        assert!(pipeline.frames.n_frames() > 0);
        assert!(pipeline.spec.n_bins() > 0);
    }

    #[test]
    #[ignore]
    fn test_music_pipeline() {
        let result = music("test_audio.wav");
        assert!(result.is_ok());

        let pipeline = result.unwrap();
        assert_eq!(pipeline.frames.frame_size(), 4096);
    }

    #[test]
    #[ignore]
    fn test_speech_pipeline() {
        let result = speech("test_audio.wav");
        assert!(result.is_ok());

        let pipeline = result.unwrap();
        assert_eq!(pipeline.frames.frame_size(), 512);
    }
}
