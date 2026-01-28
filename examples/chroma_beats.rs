//! Example: Chroma feature extraction and beat synchronization.
//!
//! This example demonstrates:
//! 1. Computing a chromagram (pitch class profile) from audio
//! 2. Detecting tempo and tracking beats
//! 3. Synchronizing chroma features to beat boundaries
//!
//! Usage:
//!   cargo run --example chroma_beats <audio_file>

use rsona::audio;
use rsona::feature::{ChromaConfig, ChromaNorm, chroma_stft, onset_strength_from_mel};
use rsona::signal::{self, FrameConfig};
use rsona::spectrum::{MelConfig, StftConfig, mel_spectrogram, stft};
use rsona::temporal::{
    AggregationMethod, SyncConfig, TempoConfig, estimate_tempo, sync_to_beats, track_beats,
};
use std::env;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <audio_file>", args[0]);
        process::exit(1);
    }

    let path = &args[1];
    println!("Loading audio from: {}", path);

    // Load audio
    let audio = match audio::load(path) {
        Ok(audio) => audio,
        Err(e) => {
            eprintln!("Error loading audio: {}", e);
            process::exit(1);
        }
    };

    let sr = audio.sample_rate;
    let duration = audio.samples.len() as f64 / sr as f64;
    println!("  Sample rate: {} Hz", sr);
    println!("  Duration: {:.2} seconds", duration);
    println!();

    // Frame audio
    let frame_cfg = FrameConfig {
        frame_size: 2048,
        hop_size: 512,
        ..Default::default()
    };
    let hop_size = frame_cfg.hop_size;
    let frames = signal::frame(&audio, frame_cfg).expect("Failed to frame audio");
    println!("Framed audio: {} frames", frames.n_frames());

    // Compute STFT
    let stft_cfg = StftConfig::default();
    let spec = stft(&frames, stft_cfg).expect("Failed to compute STFT");
    println!("STFT: {} frames × {} bins", spec.n_frames(), spec.n_bins());

    // Compute chromagram
    println!();
    println!("Computing chromagram...");
    let chroma_cfg = ChromaConfig {
        n_chroma: 12,
        norm: ChromaNorm::L2, // Normalize each frame to unit length
        ..Default::default()
    };
    let chroma = chroma_stft(&spec, chroma_cfg);
    println!(
        "  Chromagram: {} frames × {} chroma bins",
        chroma.n_frames(),
        chroma.n_chroma()
    );

    // Display first few frames of chroma
    println!("  First frame chroma values:");
    if let Some(frame) = chroma.frame(0) {
        let chroma_names = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];
        for (i, &value) in frame.iter().enumerate() {
            println!("    {}: {:.4}", chroma_names[i], value);
        }
    }

    // Compute mel spectrogram for onset detection
    println!();
    println!("Computing onset strength...");
    let mel_cfg = MelConfig::default();
    let mel_spec = mel_spectrogram(&spec, mel_cfg);
    let onset_env = onset_strength_from_mel(&mel_spec, Default::default());
    println!("  Onset envelope: {} frames", onset_env.values().len());

    // Estimate tempo
    println!();
    println!("Estimating tempo...");
    let tempo_cfg = TempoConfig::default();
    let tempo = estimate_tempo(onset_env.values(), sr, hop_size, tempo_cfg);
    println!("  Estimated tempo: {:.2} BPM", tempo.bpm);
    println!("  Period: {} frames per beat", tempo.period_frames);

    // Track beats
    println!();
    println!("Tracking beats...");
    let beat_cfg = Default::default();
    let beats = track_beats(
        onset_env.values(),
        sr,
        hop_size,
        tempo.period_frames,
        beat_cfg,
    );
    println!("  Found {} beats", beats.beat_frames.len());

    if !beats.beat_times.is_empty() {
        println!(
            "  First beat: {:.2} s (frame {})",
            beats.beat_times[0], beats.beat_frames[0]
        );
        let last_idx = beats.beat_times.len() - 1;
        println!(
            "  Last beat:  {:.2} s (frame {})",
            beats.beat_times[last_idx], beats.beat_frames[last_idx]
        );
    }

    // Synchronize chroma to beats
    println!();
    println!("Synchronizing chroma to beats...");
    let sync_cfg = SyncConfig {
        method: AggregationMethod::Mean,
        pad_end: true,
    };

    let chroma_array = chroma.as_ndarray();
    let beat_chroma = sync_to_beats(chroma_array, &beats.beat_frames, sync_cfg);

    println!(
        "  Beat-synchronized chroma: {} beats × {} chroma bins",
        beat_chroma.nrows(),
        beat_chroma.ncols()
    );

    // Display chroma for first few beats
    println!();
    println!("Chroma values for first 5 beats:");
    let chroma_names = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    let num_beats_to_show = 5.min(beat_chroma.nrows());

    for beat_idx in 0..num_beats_to_show {
        println!(
            "  Beat {} (@ {:.2}s):",
            beat_idx, beats.beat_times[beat_idx]
        );

        // Find dominant chroma
        let mut max_val = 0.0f32;
        let mut max_idx = 0;
        for chroma_idx in 0..12 {
            let val = beat_chroma[[beat_idx, chroma_idx]];
            if val > max_val {
                max_val = val;
                max_idx = chroma_idx;
            }
        }

        println!(
            "    Dominant pitch class: {} ({:.4})",
            chroma_names[max_idx], max_val
        );

        // Print all values
        print!("    [");
        for chroma_idx in 0..12 {
            print!("{:.3}", beat_chroma[[beat_idx, chroma_idx]]);
            if chroma_idx < 11 {
                print!(", ");
            }
        }
        println!("]");
    }

    // Compute statistics
    println!();
    println!("Statistics:");

    // Average chroma across all beats
    let mut avg_chroma = vec![0.0f32; 12];
    for chroma_idx in 0..12 {
        let mut sum = 0.0f32;
        for beat_idx in 0..beat_chroma.nrows() {
            sum += beat_chroma[[beat_idx, chroma_idx]];
        }
        avg_chroma[chroma_idx] = sum / beat_chroma.nrows() as f32;
    }

    println!("  Average chroma across all beats:");
    let mut sorted_chroma: Vec<(usize, f32)> = avg_chroma
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    sorted_chroma.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("    Top 3 pitch classes:");
    for i in 0..3.min(sorted_chroma.len()) {
        let (idx, val) = sorted_chroma[i];
        println!("      {}: {:.4}", chroma_names[idx], val);
    }

    println!();
    println!("Done!");
}
