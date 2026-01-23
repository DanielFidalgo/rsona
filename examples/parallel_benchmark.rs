//! Comprehensive parallel performance benchmark
//!
//! Tests the performance improvements from Rayon parallelization across
//! various workload sizes.

use rsona::audio::{Buffer, decode_audio_file};
use rsona::feature::{Chromagram, MfccConfig, chroma_stft, mfcc, onset_strength};
use rsona::signal::{FrameConfig, frame_audio};
use rsona::similarity::{NoveltyConfig, SelfSimilarityConfig, self_similarity};
use rsona::spectrum::{MelConfig, StftConfig, mel_spectrogram, stft};
use rsona::structure::{BeatLoopConfig, find_loop_by_beats};
use rsona::temporal::{SyncConfig, beat_tracking_from_onset, sync_series_to_beats, sync_to_beats};
use std::time::Instant;

fn generate_test_audio(sample_rate: u32, duration_sec: f32) -> Buffer {
    let n_samples = (sample_rate as f32 * duration_sec) as usize;
    let mut samples = vec![0.0f32; n_samples];

    // Generate complex audio with multiple frequencies
    let freqs = vec![220.0, 440.0, 880.0, 1320.0];
    let amplitudes = vec![0.3, 0.25, 0.2, 0.15];

    for i in 0..n_samples {
        let t = i as f32 / sample_rate as f32;
        let mut sample = 0.0f32;

        for (freq, amp) in freqs.iter().zip(amplitudes.iter()) {
            sample += amp * (2.0 * std::f32::consts::PI * freq * t).sin();
        }

        // Add some noise
        sample += 0.05 * ((i as f32 * 0.1).sin() * (i as f32 * 0.01).cos());

        samples[i] = sample;
    }

    Buffer::new(sample_rate, 1, samples)
}

fn benchmark_novelty_curve(ssm_sizes: &[usize]) {
    println!("\n=== Novelty Curve Benchmark ===");
    println!("Testing parallelized novelty computation");
    println!(
        "{:<12} {:>15} {:>15}",
        "SSM Size", "Time (ms)", "Speedup Note"
    );
    println!("{:-<42}", "");

    for &size in ssm_sizes {
        // Create synthetic self-similarity matrix
        let data = vec![0.5f32; size * size];
        let ssm = self_similarity::SelfSimilarity::from_dense(size, data);

        let config = NoveltyConfig {
            window: self_similarity::KernelWindow::Frames(32),
            band: None,
            gaussian: true,
            gaussian_sigma_frac: 0.5,
            normalize: true,
            smooth: Some(5),
        };

        let start = Instant::now();
        let _novelty = self_similarity::novelty_curve(&ssm, config);
        let duration = start.elapsed();

        let speedup_note = if size > 100 { "Parallel" } else { "Sequential" };

        println!(
            "{:<12} {:>12.3} ms {:>15}",
            size,
            duration.as_secs_f64() * 1000.0,
            speedup_note
        );
    }
}

fn benchmark_beat_loop_finder(audio_durations: &[f32]) {
    println!("\n=== Beat Loop Finder Benchmark ===");
    println!("Testing parallelized distance computation");
    println!(
        "{:<15} {:>12} {:>15} {:>15}",
        "Duration (s)", "N Beats", "Time (ms)", "Speedup Note"
    );
    println!("{:-<57}", "");

    let sample_rate = 22050;

    for &duration in audio_durations {
        let audio = generate_test_audio(sample_rate, duration);

        // Extract features needed for loop finding
        let frame_cfg = FrameConfig::default();
        let frames = frame_audio(&audio, frame_cfg);

        let stft_cfg = StftConfig::default();
        let spec = stft(&frames, stft_cfg).unwrap();

        let mel_cfg = MelConfig::default();
        let mel_spec = mel_spectrogram(&spec, mel_cfg);

        let onset_env = onset_strength(&mel_spec, None);

        // Estimate tempo and track beats
        let tempo =
            rsona::temporal::estimate_tempo(&onset_env.values(), sample_rate, frames.hop_size());
        let beats = beat_tracking_from_onset(
            &onset_env.values(),
            sample_rate,
            frames.hop_size(),
            tempo.period_frames,
            Default::default(),
        );

        let n_beats = beats.beat_frames.len();

        // Extract chroma and MFCC
        let chroma = chroma_stft(&spec, Default::default());
        let mfcc_cfg = MfccConfig::default();
        let mfcc = mfcc(&mel_spec, mfcc_cfg);

        let loop_cfg = BeatLoopConfig {
            min_length_samples: sample_rate as usize * 2,
            max_length_samples: Some(sample_rate as usize * 20),
            feature_window_frames: 50,
            max_candidates: 100,
            metric: rsona::structure::DistanceMetric::Manhattan,
        };

        let start = Instant::now();
        let _loop_result =
            find_loop_by_beats(&frames, &beats, &chroma, &mfcc, &onset_env, loop_cfg);
        let duration = start.elapsed();

        let speedup_note = if n_beats > 50 {
            "Parallel"
        } else {
            "Sequential"
        };

        println!(
            "{:<15} {:>12} {:>12.3} ms {:>15}",
            duration,
            n_beats,
            duration.as_secs_f64() * 1000.0,
            speedup_note
        );
    }
}

fn benchmark_beat_sync(audio_durations: &[f32]) {
    println!("\n=== Beat Synchronization Benchmark ===");
    println!("Testing parallelized beat aggregation");
    println!(
        "{:<15} {:>12} {:>15} {:>15}",
        "Duration (s)", "N Beats", "Time (ms)", "Speedup Note"
    );
    println!("{:-<57}", "");

    let sample_rate = 22050;

    for &duration in audio_durations {
        let audio = generate_test_audio(sample_rate, duration);

        let frame_cfg = FrameConfig::default();
        let frames = frame_audio(&audio, frame_cfg);

        let stft_cfg = StftConfig::default();
        let spec = stft(&frames, stft_cfg).unwrap();

        let mel_cfg = MelConfig::default();
        let mel_spec = mel_spectrogram(&spec, mel_cfg);

        let chroma = chroma_stft(&spec, Default::default());
        let onset_env = onset_strength(&mel_spec, None);

        // Track beats
        let tempo =
            rsona::temporal::estimate_tempo(&onset_env.values(), sample_rate, frames.hop_size());
        let beats = beat_tracking_from_onset(
            &onset_env.values(),
            sample_rate,
            frames.hop_size(),
            tempo.period_frames,
            Default::default(),
        );

        let n_beats = beats.beat_frames.len();

        let sync_cfg = SyncConfig::default();

        let start = Instant::now();
        let _synced_chroma =
            sync_to_beats(chroma.as_ndarray(), &beats.beat_frames, sync_cfg.clone());
        let _synced_onset = sync_series_to_beats(&onset_env.values(), &beats.beat_frames, sync_cfg);
        let duration = start.elapsed();

        let speedup_note = if n_beats > 50 {
            "Parallel"
        } else {
            "Sequential"
        };

        println!(
            "{:<15} {:>12} {:>12.3} ms {:>15}",
            duration,
            n_beats,
            duration.as_secs_f64() * 1000.0,
            speedup_note
        );
    }
}

fn benchmark_chroma_extraction(audio_durations: &[f32]) {
    println!("\n=== Chroma Extraction Benchmark ===");
    println!("Testing parallelized chroma filter application");
    println!(
        "{:<15} {:>12} {:>15}",
        "Duration (s)", "N Frames", "Time (ms)"
    );
    println!("{:-<42}", "");

    let sample_rate = 22050;

    for &duration in audio_durations {
        let audio = generate_test_audio(sample_rate, duration);

        let frame_cfg = FrameConfig::default();
        let frames = frame_audio(&audio, frame_cfg);

        let stft_cfg = StftConfig::default();
        let spec = stft(&frames, stft_cfg).unwrap();

        let n_frames = spec.n_frames();

        let start = Instant::now();
        let _chroma = chroma_stft(&spec, Default::default());
        let duration = start.elapsed();

        println!(
            "{:<15} {:>12} {:>12.3} ms",
            duration,
            n_frames,
            duration.as_secs_f64() * 1000.0
        );
    }
}

fn benchmark_full_pipeline(audio_file: Option<&str>) {
    println!("\n=== Full Pipeline Benchmark ===");
    println!("Testing complete analysis pipeline");

    if let Some(path) = audio_file {
        println!("Audio file: {}", path);

        let start_load = Instant::now();
        let audio = match decode_audio_file(path, Some(22050), Some(1)) {
            Ok(audio) => audio,
            Err(e) => {
                eprintln!("Failed to load audio: {}", e);
                return;
            }
        };
        let load_time = start_load.elapsed();

        let duration_sec = audio.duration_seconds();
        println!("Duration: {:.2} seconds", duration_sec);
        println!("Load time: {:.3} ms", load_time.as_secs_f64() * 1000.0);

        let start_analysis = Instant::now();

        // Frame audio
        let frame_cfg = FrameConfig::default();
        let frames = frame_audio(&audio, frame_cfg);
        let n_frames = frames.n_frames();

        // STFT
        let stft_cfg = StftConfig::default();
        let spec = stft(&frames, stft_cfg).unwrap();

        // Mel spectrogram
        let mel_cfg = MelConfig::default();
        let mel_spec = mel_spectrogram(&spec, mel_cfg);

        // Features
        let chroma = chroma_stft(&spec, Default::default());
        let mfcc_cfg = MfccConfig::default();
        let mfcc = mfcc(&mel_spec, mfcc_cfg);
        let onset_env = onset_strength(&mel_spec, None);

        // Tempo and beats
        let tempo = rsona::temporal::estimate_tempo(
            &onset_env.values(),
            audio.sample_rate(),
            frames.hop_size(),
        );
        let beats = beat_tracking_from_onset(
            &onset_env.values(),
            audio.sample_rate(),
            frames.hop_size(),
            tempo.period_frames,
            Default::default(),
        );
        let n_beats = beats.beat_frames.len();

        // Self-similarity and novelty
        let ssm_cfg = SelfSimilarityConfig::default();
        let ssm = self_similarity(&chroma.as_ndarray(), ssm_cfg);
        let novelty_cfg = NoveltyConfig::default();
        let _novelty = self_similarity::novelty_curve(&ssm, novelty_cfg);

        // Beat synchronization
        let sync_cfg = SyncConfig::default();
        let _synced_chroma = sync_to_beats(chroma.as_ndarray(), &beats.beat_frames, sync_cfg);

        let analysis_time = start_analysis.elapsed();

        println!("\nResults:");
        println!("  Frames: {}", n_frames);
        println!("  Tempo: {:.1} BPM", tempo.bpm);
        println!("  Beats: {}", n_beats);
        println!("  Chroma bins: {}", chroma.n_chroma());
        println!("  MFCC coefficients: {}", mfcc.n_mfcc());

        println!("\nTiming:");
        println!(
            "  Analysis time: {:.3} ms",
            analysis_time.as_secs_f64() * 1000.0
        );
        println!(
            "  Real-time factor: {:.2}x",
            duration_sec / analysis_time.as_secs_f32()
        );
    } else {
        println!("No audio file provided, testing synthetic audio");

        for &duration in &[10.0, 30.0, 60.0] {
            let audio = generate_test_audio(22050, duration);

            let start = Instant::now();

            let frame_cfg = FrameConfig::default();
            let frames = frame_audio(&audio, frame_cfg);

            let stft_cfg = StftConfig::default();
            let spec = stft(&frames, stft_cfg).unwrap();

            let mel_cfg = MelConfig::default();
            let mel_spec = mel_spectrogram(&spec, mel_cfg);

            let _chroma = chroma_stft(&spec, Default::default());
            let mfcc_cfg = MfccConfig::default();
            let _mfcc = mfcc(&mel_spec, mfcc_cfg);
            let onset_env = onset_strength(&mel_spec, None);

            let tempo = rsona::temporal::estimate_tempo(
                &onset_env.values(),
                audio.sample_rate(),
                frames.hop_size(),
            );
            let _beats = beat_tracking_from_onset(
                &onset_env.values(),
                audio.sample_rate(),
                frames.hop_size(),
                tempo.period_frames,
                Default::default(),
            );

            let analysis_time = start.elapsed();

            println!(
                "Duration: {:.1}s | Analysis: {:.3} ms | RTF: {:.2}x",
                duration,
                analysis_time.as_secs_f64() * 1000.0,
                duration / analysis_time.as_secs_f32()
            );
        }
    }
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║   Rsona Parallel Performance Benchmark                   ║");
    println!("║   Testing Rayon parallelization improvements             ║");
    println!("╚═══════════════════════════════════════════════════════════╝");

    let args: Vec<String> = std::env::args().collect();
    let audio_file = args.get(1).map(|s| s.as_str());

    // 1. Novelty curve with varying SSM sizes
    let ssm_sizes = vec![50, 100, 200, 500, 1000];
    benchmark_novelty_curve(&ssm_sizes);

    // 2. Beat loop finder with varying durations
    let loop_durations = vec![10.0, 20.0, 40.0];
    benchmark_beat_loop_finder(&loop_durations);

    // 3. Beat synchronization with varying durations
    let sync_durations = vec![10.0, 30.0, 60.0];
    benchmark_beat_sync(&sync_durations);

    // 4. Chroma extraction
    let chroma_durations = vec![10.0, 30.0, 60.0];
    benchmark_chroma_extraction(&chroma_durations);

    // 5. Full pipeline
    benchmark_full_pipeline(audio_file);

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║   Benchmark Complete                                      ║");
    println!("║                                                           ║");
    println!("║   Thresholds used:                                        ║");
    println!("║   - Novelty curve: n > 100 frames                         ║");
    println!("║   - Beat distances: n > 50 beats                          ║");
    println!("║   - Beat sync: n > 50 beats                               ║");
    println!("║   - Chroma filters: n >= 12 bins (typical)                ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
}
