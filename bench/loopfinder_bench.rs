use serde::Serialize;
use std::time::Instant;

use rsona::{
    audio,
    feature::{MfccConfig, OnsetConfig, mfcc, onset_strength},
    signal::{FrameConfig, frame},
    similarity::{FrameFeatures, LagEnergyConfig, SelfSimilarityConfig, self_similarity},
    spectrum::{MelConfig, StftConfig, mel_spectrogram, stft},
    structure::{SegmentationConfig, segment_intro_loop_outro_with_onsets},
};

#[derive(Serialize)]
struct LoopfinderBenchResult {
    audio_file: String,
    duration_seconds: f64,
    sample_rate: u32,
    n_samples: usize,
    parameters: Parameters,
    timing: Timing,
    best_result: BestLoop,
    segmentation: Segmentation,
    #[serde(skip_serializing_if = "Option::is_none")]
    confidence: Option<ConfidenceMetrics>,
}

#[derive(Serialize)]
struct Parameters {
    hop_size: usize,
    frame_size: usize,
    max_lag_frames: usize,
    min_loop_frames: usize,
    max_ssm_lag: usize,
}

#[derive(Serialize)]
struct Timing {
    load_time_ms: f64,
    frame_time_ms: f64,
    stft_time_ms: f64,
    mel_time_ms: f64,
    mfcc_time_ms: f64,
    similarity_time_ms: f64,
    onset_time_ms: f64,
    segmentation_time_ms: f64,
    total_time_ms: f64,
}

#[derive(Serialize)]
struct BestLoop {
    loop_begin_sample: usize,
    loop_end_sample: usize,
    loop_begin_seconds: f64,
    loop_end_seconds: f64,
    loop_duration_seconds: f64,
    loop_begin_frame: usize,
    loop_end_frame: usize,
    best_lag_frames: usize,
    best_phase_start: usize,
}

#[derive(Serialize)]
struct Segmentation {
    intro_seconds: f64,
    loop_seconds: f64,
    outro_seconds: f64,
    intro_frames: usize,
    loop_frames: usize,
    outro_frames: usize,
    num_loop_repeats: usize,
}

#[derive(Serialize)]
struct ConfidenceMetrics {
    loop_confidence: f32,
    tempo_bpm: Option<f32>,
    n_section_boundaries: usize,
    n_beat_frames: Option<usize>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <audio_path> [--detailed]", args[0]);
        std::process::exit(1);
    }

    let audio_path = &args[1];
    let detailed = args.iter().any(|arg| arg == "--detailed");

    let total_start = Instant::now();

    // --- Load Audio ---
    let t_load = Instant::now();
    let buffer = audio::load(audio_path).expect("Failed to load audio");
    let load_time = t_load.elapsed();

    let duration_seconds = buffer.samples.len() as f64 / buffer.sample_rate as f64;
    let n_samples = buffer.samples.len();

    if detailed {
        eprintln!("Loaded: {} samples @ {} Hz", n_samples, buffer.sample_rate);
        eprintln!("Duration: {:.2}s", duration_seconds);
    }

    // --- Frame ---
    let frame_cfg = FrameConfig {
        frame_size: 2048,
        hop_size: 512,
        ..Default::default()
    };

    // Store values before moving frame_cfg
    let hop_size = frame_cfg.hop_size;
    let frame_size = frame_cfg.frame_size;

    let t_frame = Instant::now();
    let frames = frame(&buffer, frame_cfg).expect("Failed to frame audio");
    let frame_time = t_frame.elapsed();

    let n_frames = frames.n_frames();

    if detailed {
        eprintln!("Frames: {}", n_frames);
    }

    // --- STFT ---
    let t_stft = Instant::now();
    let spec = stft(&frames, StftConfig::default()).expect("Failed to compute STFT");
    let stft_time = t_stft.elapsed();

    // --- Mel Spectrogram ---
    let t_mel = Instant::now();
    let mel = mel_spectrogram(&spec, MelConfig::default());
    let mel_time = t_mel.elapsed();

    // --- MFCC ---
    let t_mfcc = Instant::now();
    let mfcc_result = mfcc(&mel, MfccConfig::default());
    let mfcc_time = t_mfcc.elapsed();

    if detailed {
        eprintln!(
            "MFCC: {} frames, {} coefficients",
            mfcc_result.n_frames(),
            mfcc_result.n_mfcc()
        );
    }

    // --- Calculate max_lag for long loops ---
    // For 60+ second loops at hop=512, sr=44100, we need ~5200 frames
    // Use 75% of total frames as upper bound to allow for intro/outro
    let max_ssm_lag = (n_frames * 3 / 4).min(6000).max(800);
    // For lag energy, use slightly less to focus on more likely loop lengths
    let max_loop_lag = (n_frames * 2 / 3).min(5000).max(800);

    if detailed {
        eprintln!(
            "Max SSM lag: {} frames (~{:.1}s)",
            max_ssm_lag,
            max_ssm_lag as f64 * hop_size as f64 / frames.sample_rate() as f64
        );
        eprintln!(
            "Max loop lag: {} frames (~{:.1}s)",
            max_loop_lag,
            max_loop_lag as f64 * hop_size as f64 / frames.sample_rate() as f64
        );
    }

    // --- Self-Similarity Matrix ---
    let t_sim = Instant::now();
    let feats = FrameFeatures::new(
        mfcc_result.n_frames(),
        mfcc_result.n_mfcc(),
        mfcc_result.as_slice(),
    );

    let ssm = self_similarity(
        &feats,
        SelfSimilarityConfig {
            max_lag: Some(max_ssm_lag),
            ..Default::default()
        },
    );
    let similarity_time = t_sim.elapsed();

    if detailed {
        eprintln!("SSM: {} x {} frames", ssm.n_frames(), ssm.n_frames());
    }

    // --- Onset Strength ---
    let t_onset = Instant::now();
    let onset = onset_strength(&spec, OnsetConfig::default());
    let onset_time = t_onset.elapsed();

    if detailed {
        eprintln!("Onset envelope: {} frames", onset.values().len());
    }

    // --- Segmentation (Loop Finding) ---
    let t_seg = Instant::now();

    // Configure for longer loops with improved phase selection
    let seg_config = SegmentationConfig {
        lag: LagEnergyConfig {
            min_lag: 20,
            max_lag: max_loop_lag,
            ..Default::default()
        },
        phase: rsona::similarity::RepeatPhaseConfig {
            window: 50,       // Larger window for more stable phase selection
            smooth: Some(15), // More smoothing to avoid local minima
        },
        min_loop_frames: 50, // ~0.6 seconds minimum
        snap_to_section_boundaries: true,
        snap_to_beats: true,
        ..Default::default()
    };

    let seg_result =
        segment_intro_loop_outro_with_onsets(&frames, &ssm, onset.values(), seg_config)
            .expect("Failed to segment audio");
    let segmentation_time = t_seg.elapsed();

    let total_time = total_start.elapsed();

    // --- Build Result ---
    let best_loop = BestLoop {
        loop_begin_sample: seg_result.loop_seg.start_sample,
        loop_end_sample: seg_result.loop_seg.end_sample,
        loop_begin_seconds: seg_result.loop_seg.start_seconds,
        loop_end_seconds: seg_result.loop_seg.end_seconds,
        loop_duration_seconds: seg_result.loop_seg.end_seconds - seg_result.loop_seg.start_seconds,
        loop_begin_frame: seg_result.loop_seg.start_frame,
        loop_end_frame: seg_result.loop_seg.end_frame,
        best_lag_frames: seg_result.best_lag_frames,
        best_phase_start: seg_result.best_phase_start,
    };

    let segmentation = Segmentation {
        intro_seconds: seg_result.intro.end_seconds - seg_result.intro.start_seconds,
        loop_seconds: seg_result.loop_seg.end_seconds - seg_result.loop_seg.start_seconds,
        outro_seconds: seg_result.outro.end_seconds - seg_result.outro.start_seconds,
        intro_frames: seg_result.intro.end_frame - seg_result.intro.start_frame,
        loop_frames: seg_result.loop_seg.end_frame - seg_result.loop_seg.start_frame,
        outro_frames: seg_result.outro.end_frame - seg_result.outro.start_frame,
        num_loop_repeats: seg_result.num_loop_repeats,
    };

    let confidence = Some(ConfidenceMetrics {
        loop_confidence: seg_result.loop_confidence,
        tempo_bpm: seg_result.tempo_bpm,
        n_section_boundaries: seg_result.section_boundaries.len(),
        n_beat_frames: seg_result.beat_frames.as_ref().map(|b| b.len()),
    });

    let timing = Timing {
        load_time_ms: load_time.as_secs_f64() * 1000.0,
        frame_time_ms: frame_time.as_secs_f64() * 1000.0,
        stft_time_ms: stft_time.as_secs_f64() * 1000.0,
        mel_time_ms: mel_time.as_secs_f64() * 1000.0,
        mfcc_time_ms: mfcc_time.as_secs_f64() * 1000.0,
        similarity_time_ms: similarity_time.as_secs_f64() * 1000.0,
        onset_time_ms: onset_time.as_secs_f64() * 1000.0,
        segmentation_time_ms: segmentation_time.as_secs_f64() * 1000.0,
        total_time_ms: total_time.as_secs_f64() * 1000.0,
    };

    let parameters = Parameters {
        hop_size,
        frame_size,
        max_lag_frames: max_loop_lag,
        min_loop_frames: 50,
        max_ssm_lag,
    };

    let result = LoopfinderBenchResult {
        audio_file: audio_path.to_string(),
        duration_seconds,
        sample_rate: buffer.sample_rate,
        n_samples,
        parameters,
        timing,
        best_result: best_loop,
        segmentation,
        confidence,
    };

    // Output JSON
    let json = serde_json::to_string_pretty(&result).expect("Failed to serialize result");
    println!("{}", json);

    if detailed {
        eprintln!("\n=== Timing Breakdown ===");
        eprintln!("Load:         {:>8.2} ms", result.timing.load_time_ms);
        eprintln!("Frame:        {:>8.2} ms", result.timing.frame_time_ms);
        eprintln!("STFT:         {:>8.2} ms", result.timing.stft_time_ms);
        eprintln!("Mel:          {:>8.2} ms", result.timing.mel_time_ms);
        eprintln!("MFCC:         {:>8.2} ms", result.timing.mfcc_time_ms);
        eprintln!("Similarity:   {:>8.2} ms", result.timing.similarity_time_ms);
        eprintln!("Onset:        {:>8.2} ms", result.timing.onset_time_ms);
        eprintln!(
            "Segmentation: {:>8.2} ms",
            result.timing.segmentation_time_ms
        );
        eprintln!("---");
        eprintln!("Total:        {:>8.2} ms", result.timing.total_time_ms);
        eprintln!();
        eprintln!("=== Loop Result ===");
        eprintln!(
            "Intro:  {:.2}s → {:.2}s ({:.2}s)",
            seg_result.intro.start_seconds,
            seg_result.intro.end_seconds,
            result.segmentation.intro_seconds
        );
        eprintln!(
            "Loop:   {:.2}s → {:.2}s ({:.2}s)",
            result.best_result.loop_begin_seconds,
            result.best_result.loop_end_seconds,
            result.best_result.loop_duration_seconds
        );
        eprintln!(
            "Outro:  {:.2}s → {:.2}s ({:.2}s)",
            seg_result.outro.start_seconds,
            seg_result.outro.end_seconds,
            result.segmentation.outro_seconds
        );
        eprintln!();
        eprintln!(
            "Loop repeats {} time(s) (period: {:.2}s)",
            result.segmentation.num_loop_repeats,
            result.best_result.loop_duration_seconds / result.segmentation.num_loop_repeats as f64
        );
        eprintln!();
        if let Some(conf) = &result.confidence {
            eprintln!("Confidence: {:.2}", conf.loop_confidence);
            if let Some(bpm) = conf.tempo_bpm {
                eprintln!("Tempo:      {:.1} BPM", bpm);
            }
        }
    }
}
