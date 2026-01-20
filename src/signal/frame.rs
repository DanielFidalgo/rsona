//! Framing: convert an audio buffer into overlapping frames with an optional window.
//!
//! This module is time-domain only. It does not compute FFTs.

use crate::{
    audio::buffer::Buffer,
    signal::window::{Window, WindowSpec},
};

use ndarray::{ArrayView2, ShapeBuilder};

/// Errors produced by signal operations.
#[derive(thiserror::Error, Debug)]
pub enum SignalError {
    /// Invalid frame configuration.
    #[error("invalid frame configuration: {0}")]
    InvalidConfig(&'static str),

    /// Invalid channel index.
    #[error("requested channel index {requested} but buffer has {available} channels")]
    ChannelOutOfRange {
        /// The requested channel index.
        requested: usize,
        /// The number of available channels.
        available: usize,
    },

    /// Audio buffer is empty.
    #[error("audio buffer is empty")]
    EmptyBuffer,
}

/// How to handle padding when the last frame extends past the end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Padding {
    /// Do not pad: only include frames fully contained in the signal.
    None,
    /// Pad the final frame with zeros so the last hop is included.
    ZeroPadEnd,
}

/// How to select channels for framing.
///
/// Important: rsona does **not** implicitly downmix to mono.
/// If you want mono framing, choose a `MixDown` mode explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelMode {
    /// Extract a single channel (0-based).
    Channel(usize),

    /// Mix down all channels into mono by averaging.
    ///
    /// This is explicit and deterministic.
    MixDownAverage,
}

/// Configuration for framing a signal.
#[derive(Debug, Clone)]
pub struct FrameConfig {
    /// Frame length in samples (per channel).
    pub frame_size: usize,

    /// Hop length in samples (per channel).
    pub hop_size: usize,

    /// Window type applied to each frame.
    pub window: Window,

    /// Padding strategy for the end of the signal.
    pub padding: Padding,

    /// Channel selection strategy.
    pub channel_mode: ChannelMode,

    /// If true, center the frames by padding the signal at both ends.
    ///
    /// When enabled, pads `frame_size / 2` zeros at the start and end of the signal,
    /// ensuring the first frame is centered at sample 0. This is the standard
    /// default behavior (`center=True`).
    ///
    /// Default: `true` (standard default).
    pub center: bool,
}

impl Default for FrameConfig {
    fn default() -> Self {
        Self {
            frame_size: 2048,
            hop_size: 512,
            window: Window::Hann,
            padding: Padding::ZeroPadEnd,
            channel_mode: ChannelMode::Channel(0),
            center: true,
        }
    }
}

impl FrameConfig {
    /// Standard audio analysis preset (2048 frame size, 512 hop, mono mix).
    ///
    /// This is the most common configuration used throughout rsona's
    /// benchmarks and examples. It provides a good balance between
    /// time and frequency resolution for general audio analysis.
    ///
    /// Configuration:
    /// - Frame size: 2048 samples (~46ms at 44.1kHz)
    /// - Hop size: 512 samples (~12ms at 44.1kHz)
    /// - Window: Hann
    /// - Center: true
    /// - Channel mode: Mix to mono (average)
    pub fn standard() -> Self {
        Self {
            frame_size: 2048,
            hop_size: 512,
            window: Window::Hann,
            padding: Padding::ZeroPadEnd,
            channel_mode: ChannelMode::MixDownAverage,
            center: true,
        }
    }

    /// Music analysis preset (4096 frame size, 1024 hop, mono mix).
    ///
    /// Higher resolution for detailed harmonic and spectral analysis.
    /// Provides better frequency resolution at the cost of lower
    /// time resolution.
    ///
    /// Configuration:
    /// - Frame size: 4096 samples (~93ms at 44.1kHz)
    /// - Hop size: 1024 samples (~23ms at 44.1kHz)
    /// - Window: Hann
    /// - Center: true
    /// - Channel mode: Mix to mono (average)
    pub fn music() -> Self {
        Self {
            frame_size: 4096,
            hop_size: 1024,
            window: Window::Hann,
            padding: Padding::ZeroPadEnd,
            channel_mode: ChannelMode::MixDownAverage,
            center: true,
        }
    }

    /// Speech analysis preset (512 frame size, 256 hop, mono mix).
    ///
    /// Lower latency configuration suitable for real-time speech
    /// processing and applications requiring fast updates.
    ///
    /// Configuration:
    /// - Frame size: 512 samples (~12ms at 44.1kHz)
    /// - Hop size: 256 samples (~6ms at 44.1kHz)
    /// - Window: Hann
    /// - Center: true
    /// - Channel mode: Mix to mono (average)
    pub fn speech() -> Self {
        Self {
            frame_size: 512,
            hop_size: 256,
            window: Window::Hann,
            padding: Padding::ZeroPadEnd,
            channel_mode: ChannelMode::MixDownAverage,
            center: true,
        }
    }
}

/// Framed representation of a signal.
///
/// Internally stores frames in a single contiguous Vec<f32> in row-major order:
/// `data[frame_index * frame_size + sample_in_frame]`.
#[derive(Debug, Clone)]
pub struct Frames {
    sample_rate: u32,
    frame_size: usize,
    hop_size: usize,
    window: WindowSpec,
    padding: Padding,
    channel_mode: ChannelMode,
    n_frames: usize,
    data: Vec<f32>,
}

impl Frames {
    /// Sample rate of the original audio (Hz).
    #[inline]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Frame size in samples.
    #[inline]
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    /// Hop size in samples.
    #[inline]
    pub fn hop_size(&self) -> usize {
        self.hop_size
    }

    /// Number of frames.
    #[inline]
    pub fn n_frames(&self) -> usize {
        self.n_frames
    }

    /// Window kind.
    #[inline]
    pub fn window(&self) -> Window {
        self.window.kind
    }

    /// Channel mode used to produce these frames.
    #[inline]
    pub fn channel_mode(&self) -> ChannelMode {
        self.channel_mode
    }

    /// Padding strategy.
    #[inline]
    pub fn padding(&self) -> Padding {
        self.padding
    }

    /// Returns a view of frames as an ndarray 2D array: shape `(n_frames, frame_size)`.
    ///
    /// This is zero-copy. The returned view references the internal contiguous storage.
    pub fn as_ndarray(&self) -> ArrayView2<'_, f32> {
        let shape = (self.n_frames, self.frame_size).strides((self.frame_size, 1));
        ArrayView2::from_shape(shape, &self.data).expect("internal frame storage is contiguous")
    }

    /// Get an immutable slice for a single frame.
    #[inline]
    pub fn frame(&self, index: usize) -> Option<&[f32]> {
        if index >= self.n_frames {
            return None;
        }
        let start = index * self.frame_size;
        let end = start + self.frame_size;
        Some(&self.data[start..end])
    }

    /// Time (seconds) of the start of a frame.
    #[inline]
    pub fn frame_start_seconds(&self, index: usize) -> Option<f64> {
        if index >= self.n_frames {
            return None;
        }
        Some((index * self.hop_size) as f64 / self.sample_rate as f64)
    }

    /// Internal access to contiguous frame storage.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

/// Convert an `audio::Buffer` to overlapping frames using `config`.
///
/// This performs an explicit channel selection or downmix, then frames in the time domain.
/// It applies the configured window to every frame.
///
/// No FFTs or frequency-domain transforms occur here.
pub fn frame(audio: &Buffer, config: FrameConfig) -> Result<Frames, SignalError> {
    validate_config(&config)?;

    if audio.samples.is_empty() {
        return Err(SignalError::EmptyBuffer);
    }
    if audio.channels == 0 {
        return Err(SignalError::InvalidConfig("audio.channels must be > 0"));
    }
    if !audio.samples.len().is_multiple_of(audio.channels) {
        return Err(SignalError::InvalidConfig(
            "audio.samples length must be divisible by audio.channels",
        ));
    }

    // 1) Extract a single-channel signal (explicit selection or mixdown).
    let mut x = extract_channel(audio, config.channel_mode)?;

    // 2) Apply center padding if requested (standard behavior).
    //    Pads frame_size/2 zeros at start and end, centering first frame at sample 0.
    if config.center {
        let pad = config.frame_size / 2;
        let mut padded = vec![0.0f32; x.len() + 2 * pad];
        padded[pad..pad + x.len()].copy_from_slice(&x);
        x = padded;
    }

    // 3) Determine number of frames.
    let n = x.len();
    // When center padding is enabled, use Padding::None logic for frame counting
    // because center padding already handles edge alignment (standard behavior).
    let effective_padding = if config.center {
        Padding::None
    } else {
        config.padding
    };
    #[allow(unused_variables)]
    let (n_frames, total_len) =
        compute_frame_count(n, config.frame_size, config.hop_size, effective_padding);

    // 4) Precompute window.
    let window = WindowSpec::new(config.window, config.frame_size);

    // 5) Allocate output: contiguous storage (n_frames * frame_size).
    let mut data = vec![0.0f32; n_frames * config.frame_size];

    // 6) Fill frames with fused copy-and-window operation.
    // Each frame starts at: i * hop_size
    // If it runs past n, pad zeros (if configured).

    if n_frames > 50 {
        // Parallel path for large frame counts
        use rayon::prelude::*;

        data.par_chunks_mut(config.frame_size)
            .enumerate()
            .for_each(|(i, out_row)| {
                let start = i * config.hop_size;
                let available = n.saturating_sub(start);
                let to_copy = available.min(config.frame_size);

                // Fused copy-and-window with SIMD-friendly unrolling
                if to_copy > 0 {
                    fused_copy_window_optimized(
                        &x[start..start + to_copy],
                        &window.coeffs[..to_copy],
                        &mut out_row[..to_copy],
                    );
                }

                // Window the zero-padded region (if any)
                if to_copy < config.frame_size {
                    // Already zeros, just need to apply window
                    for j in to_copy..config.frame_size {
                        out_row[j] = 0.0; // Already zero, but ensures it
                    }
                }
            });
    } else {
        // Sequential path for small frame counts
        for i in 0..n_frames {
            let start = i * config.hop_size;
            let out_row = &mut data[i * config.frame_size..(i + 1) * config.frame_size];

            let available = n.saturating_sub(start);
            let to_copy = available.min(config.frame_size);

            // Fused copy-and-window with SIMD-friendly unrolling
            if to_copy > 0 {
                fused_copy_window_optimized(
                    &x[start..start + to_copy],
                    &window.coeffs[..to_copy],
                    &mut out_row[..to_copy],
                );
            }

            // Zero-padded region already zeros, no need to process
        }
    }

    Ok(Frames {
        sample_rate: audio.sample_rate,
        frame_size: config.frame_size,
        hop_size: config.hop_size,
        window,
        padding: config.padding,
        channel_mode: config.channel_mode,
        n_frames,
        data,
    })
}

/// Fused copy-and-window operation with SIMD-friendly unrolling
/// This combines copying samples and applying window in one pass for better cache utilization
#[inline(always)]
fn fused_copy_window_optimized(input: &[f32], window: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), window.len());
    debug_assert_eq!(input.len(), output.len());

    let len = input.len();
    const CHUNK: usize = 8;
    let main_chunks = len / CHUNK;
    let _remainder = len % CHUNK;

    // Main loop with manual unrolling for SIMD
    for chunk_idx in 0..main_chunks {
        let base = chunk_idx * CHUNK;

        // Fused multiply: output[i] = input[i] * window[i]
        output[base] = input[base] * window[base];
        output[base + 1] = input[base + 1] * window[base + 1];
        output[base + 2] = input[base + 2] * window[base + 2];
        output[base + 3] = input[base + 3] * window[base + 3];
        output[base + 4] = input[base + 4] * window[base + 4];
        output[base + 5] = input[base + 5] * window[base + 5];
        output[base + 6] = input[base + 6] * window[base + 6];
        output[base + 7] = input[base + 7] * window[base + 7];
    }

    // Handle remainder
    for i in (main_chunks * CHUNK)..len {
        output[i] = input[i] * window[i];
    }
}

fn validate_config(config: &FrameConfig) -> Result<(), SignalError> {
    if config.frame_size == 0 {
        return Err(SignalError::InvalidConfig("frame_size must be > 0"));
    }
    if config.hop_size == 0 {
        return Err(SignalError::InvalidConfig("hop_size must be > 0"));
    }
    // Allow hop > frame_size (it can be used), but it’s unusual.
    Ok(())
}

fn extract_channel(audio: &Buffer, mode: ChannelMode) -> Result<Vec<f32>, SignalError> {
    let n_frames = audio.samples.len() / audio.channels;

    match mode {
        ChannelMode::Channel(ch) => {
            if ch >= audio.channels {
                return Err(SignalError::ChannelOutOfRange {
                    requested: ch,
                    available: audio.channels,
                });
            }

            // Pre-allocate output
            let mut out = vec![0.0f32; n_frames];
            let channels = audio.channels;

            if n_frames > 5000 {
                // Parallel path for large buffers
                use rayon::prelude::*;
                out.par_chunks_mut(512)
                    .enumerate()
                    .for_each(|(chunk_idx, chunk)| {
                        let start_frame = chunk_idx * 512;
                        let end_frame = (start_frame + chunk.len()).min(n_frames);

                        for (local_idx, frame_idx) in (start_frame..end_frame).enumerate() {
                            chunk[local_idx] = audio.samples[frame_idx * channels + ch];
                        }
                    });
            } else {
                // Sequential with SIMD-friendly unrolling
                const CHUNK: usize = 8;
                let main_chunks = n_frames / CHUNK;
                let _remainder = n_frames % CHUNK;

                for chunk_idx in 0..main_chunks {
                    let base = chunk_idx * CHUNK;
                    out[base] = audio.samples[base * channels + ch];
                    out[base + 1] = audio.samples[(base + 1) * channels + ch];
                    out[base + 2] = audio.samples[(base + 2) * channels + ch];
                    out[base + 3] = audio.samples[(base + 3) * channels + ch];
                    out[base + 4] = audio.samples[(base + 4) * channels + ch];
                    out[base + 5] = audio.samples[(base + 5) * channels + ch];
                    out[base + 6] = audio.samples[(base + 6) * channels + ch];
                    out[base + 7] = audio.samples[(base + 7) * channels + ch];
                }

                for i in (main_chunks * CHUNK)..n_frames {
                    out[i] = audio.samples[i * channels + ch];
                }
            }

            Ok(out)
        }
        ChannelMode::MixDownAverage => {
            let channels = audio.channels;
            let inv_channels = 1.0 / channels as f32;
            let mut out = vec![0.0f32; n_frames];

            if channels == 2 {
                // Optimized stereo mixdown (most common case)
                if n_frames > 5000 {
                    // Parallel path
                    use rayon::prelude::*;
                    out.par_chunks_mut(512)
                        .enumerate()
                        .for_each(|(chunk_idx, chunk)| {
                            let start_frame = chunk_idx * 512;
                            let end_frame = (start_frame + chunk.len()).min(n_frames);

                            for (local_idx, frame_idx) in (start_frame..end_frame).enumerate() {
                                let base = frame_idx * 2;
                                chunk[local_idx] =
                                    (audio.samples[base] + audio.samples[base + 1]) * 0.5;
                            }
                        });
                } else {
                    // Sequential with SIMD-friendly unrolling
                    const CHUNK: usize = 8;
                    let main_chunks = n_frames / CHUNK;
                    let _remainder = n_frames % CHUNK;

                    for chunk_idx in 0..main_chunks {
                        let base_out = chunk_idx * CHUNK;
                        let base_in = base_out * 2;

                        out[base_out] = (audio.samples[base_in] + audio.samples[base_in + 1]) * 0.5;
                        out[base_out + 1] =
                            (audio.samples[base_in + 2] + audio.samples[base_in + 3]) * 0.5;
                        out[base_out + 2] =
                            (audio.samples[base_in + 4] + audio.samples[base_in + 5]) * 0.5;
                        out[base_out + 3] =
                            (audio.samples[base_in + 6] + audio.samples[base_in + 7]) * 0.5;
                        out[base_out + 4] =
                            (audio.samples[base_in + 8] + audio.samples[base_in + 9]) * 0.5;
                        out[base_out + 5] =
                            (audio.samples[base_in + 10] + audio.samples[base_in + 11]) * 0.5;
                        out[base_out + 6] =
                            (audio.samples[base_in + 12] + audio.samples[base_in + 13]) * 0.5;
                        out[base_out + 7] =
                            (audio.samples[base_in + 14] + audio.samples[base_in + 15]) * 0.5;
                    }

                    for i in (main_chunks * CHUNK)..n_frames {
                        let base = i * 2;
                        out[i] = (audio.samples[base] + audio.samples[base + 1]) * 0.5;
                    }
                }
            } else {
                // General path for any channel count
                for i in 0..n_frames {
                    let base = i * channels;
                    let mut sum = 0.0f32;
                    for c in 0..channels {
                        sum += audio.samples[base + c];
                    }
                    out[i] = sum * inv_channels;
                }
            }

            Ok(out)
        }
    }
}

/// Compute number of frames to emit, and the effective total covered length.
///
/// Returns `(n_frames, total_len)`, where `total_len` is the maximum sample index
/// potentially referenced (for debugging/validation).
fn compute_frame_count(
    n_samples: usize,
    frame_size: usize,
    hop_size: usize,
    padding: Padding,
) -> (usize, usize) {
    match padding {
        Padding::None => {
            if n_samples < frame_size {
                return (0, 0);
            }
            // Only frames fully contained:
            // last start <= n_samples - frame_size
            let last_start = n_samples - frame_size;
            let n_frames = (last_start / hop_size) + 1;
            let total_len = (n_frames - 1) * hop_size + frame_size;
            (n_frames, total_len)
        }
        Padding::ZeroPadEnd => {
            if n_samples == 0 {
                return (0, 0);
            }
            // Include the last hop even if final frame is partial.
            // We want starts: 0, hop, 2hop, ... <= n_samples-1
            let last_start = n_samples - 1;
            let n_frames = (last_start / hop_size) + 1;
            let total_len = (n_frames - 1) * hop_size + frame_size;
            (n_frames, total_len)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::buffer::Buffer;

    fn stereo_buffer() -> Buffer {
        // 4 frames (per channel), stereo interleaved:
        // L: [1,2,3,4], R: [10,20,30,40]
        let samples = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0];
        Buffer::new(44100, 2, samples)
    }

    #[test]
    fn frame_extract_channel_0_rectangular() {
        let audio = stereo_buffer();
        let cfg = FrameConfig {
            frame_size: 2,
            hop_size: 1,
            window: Window::Rectangular,
            padding: Padding::None,
            channel_mode: ChannelMode::Channel(0),
            center: false,
        };

        let frames = frame(&audio, cfg).unwrap();
        assert_eq!(frames.n_frames(), 3);
        assert_eq!(frames.as_slice(), &[1.0, 2.0, 2.0, 3.0, 3.0, 4.0]);
    }

    #[test]
    fn frame_mixdown_average_rectangular() {
        let audio = stereo_buffer();
        let cfg = FrameConfig {
            frame_size: 2,
            hop_size: 2,
            window: Window::Rectangular,
            padding: Padding::ZeroPadEnd,
            channel_mode: ChannelMode::MixDownAverage,
            center: false,
        };

        // MixDownAverage per frame:
        // [(1+10)/2, (2+20)/2, (3+30)/2, (4+40)/2] = [5.5, 11.0, 16.5, 22.0]
        // frame_size=2, hop=2 => starts 0,2 => frames: [5.5,11.0], [16.5,22.0]
        let frames = frame(&audio, cfg).unwrap();
        assert_eq!(frames.n_frames(), 2);
        assert_eq!(frames.as_slice(), &[5.5, 11.0, 16.5, 22.0]);
    }

    #[test]
    fn frame_padding_zeropad_end_allows_partial() {
        let audio = Buffer::new(44100, 1, vec![1.0, 2.0, 3.0]);
        let cfg = FrameConfig {
            frame_size: 4,
            hop_size: 2,
            window: Window::Rectangular,
            padding: Padding::ZeroPadEnd,
            channel_mode: ChannelMode::Channel(0),
            center: false,
        };

        // starts 0,2 (<= n-1=2) => 2 frames
        // frame0: [1,2,3,0]
        // frame1: start=2 => [3,0,0,0]
        let frames = frame(&audio, cfg).unwrap();
        assert_eq!(frames.n_frames(), 2);
        assert_eq!(frames.as_slice(), &[1.0, 2.0, 3.0, 0.0, 3.0, 0.0, 0.0, 0.0]);
    }
}
