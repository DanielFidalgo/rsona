use std::fs::File;
use std::path::Path;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::default::{get_codecs, get_probe};

use super::{AudioError, Buffer};

/// Load audio file.
pub fn load<P: AsRef<Path>>(path: P) -> Result<Buffer, AudioError> {
    let file = File::open(path)?;
    let media_source_stream = MediaSourceStream::new(Box::new(file), Default::default());

    let hint = Hint::new();

    let probed = get_probe()
        .format(
            &hint,
            media_source_stream,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|_| AudioError::UnsupportedFormat)?;

    let mut format = probed.format;

    // --- IMPORTANT: extract track info, then drop the borrow ---
    let (track_id, codec_params) = {
        let track = format
            .default_track()
            .ok_or(AudioError::UnsupportedFormat)?;
        (track.id, track.codec_params.clone())
    };

    let sample_rate = codec_params.sample_rate.ok_or(AudioError::DecodeError)?;

    let channels = codec_params
        .channels
        .ok_or(AudioError::DecodeError)?
        .count();

    let mut decoder = get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|_| AudioError::DecodeError)?;

    // Estimate capacity: assume 3 minutes at 44.1kHz stereo = ~15.8M samples
    // This reduces reallocations significantly
    let estimated_capacity = codec_params
        .n_frames
        .map(|frames| (frames * channels as u64) as usize)
        .unwrap_or(16_000_000); // Default to ~3 min stereo at 44.1kHz

    let mut samples = Vec::<f32>::with_capacity(estimated_capacity);

    // Reuse sample buffer across packets to avoid repeated allocation
    let mut sample_buffer_cache: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(_)) => break, // EOF
            Err(_) => return Err(AudioError::DecodeError),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = decoder
            .decode(&packet)
            .map_err(|_| AudioError::DecodeError)?;

        // Reuse or create sample buffer
        let sample_buffer = sample_buffer_cache.get_or_insert_with(|| {
            SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec())
        });

        // Ensure buffer has enough capacity
        if sample_buffer.capacity() < decoded.capacity() {
            *sample_buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
        }

        sample_buffer.copy_interleaved_ref(decoded);

        samples.extend_from_slice(sample_buffer.samples());
    }

    Ok(Buffer::new(sample_rate, channels, samples))
}
