//! rsona Benchmark Runner for librosa Comparison
//!
//! This binary runs rsona benchmarks following the shared benchmark specification.
//! All output is structured JSON matching the librosa runner schema.
//!
//! Usage:
//!     cargo run --release --bin rsona_runner -- --audio path/to/audio.wav --feature spectral_centroid --runs 10
//!     cargo run --release --bin rsona_runner -- --spec benchmark_spec.json --audio test.wav

use rsona::{
    audio,
    feature::{
        ChromaConfig, ChromaNorm, MfccConfig, OnsetConfig, chroma_stft, mfcc, onset_strength,
        spectral_bandwidth, spectral_centroid, spectral_flux, spectral_rolloff,
    },
    signal::{FrameConfig, frame},
    spectrum::{MelConfig, StftConfig, mel_spectrogram, stft},
    temporal::estimate_tempo,
};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Serialize, Deserialize)]
struct BenchmarkParams {
    sample_rate: Option<u32>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    n_mels: Option<usize>,
    n_mfcc: Option<usize>,
    n_chroma: Option<usize>,
    roll_percent: Option<f32>,
}

impl Default for BenchmarkParams {
    fn default() -> Self {
        Self {
            sample_rate: Some(22050),
            n_fft: Some(2048),
            hop_length: Some(512),
            n_mels: Some(128),
            n_mfcc: Some(13),
            n_chroma: Some(12),
            roll_percent: Some(0.85),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct ResultSummary {
    #[serde(skip_serializing_if = "Option::is_none")]
    mean: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    std: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    shape: Option<Vec<usize>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    value: Option<f32>,
}

#[derive(Debug, Serialize)]
struct BenchmarkResult {
    r#impl: String,
    feature: String,
    runtime_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    result_summary: Option<ResultSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct RuntimeStats {
    mean: f64,
    median: f64,
    std: f64,
    min: f64,
    max: f64,
}

#[derive(Debug, Serialize)]
struct AggregatedResult {
    r#impl: String,
    feature: String,
    runtime_ms: RuntimeStats,
    num_runs: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    result_summary: Option<ResultSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    breakdown: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct PipelineBreakdown {
    stft_ms: f64,
    mel_ms: f64,
    mfcc_ms: f64,
    centroid_ms: f64,
    flux_ms: f64,
    onset_ms: f64,
    tempo_ms: f64,
}

#[derive(Debug, Serialize)]
struct SingleFeatureOutput {
    benchmark_type: String,
    feature: String,
    audio: String,
    statistics: AggregatedResult,
    all_runs: Vec<BenchmarkResult>,
}

#[derive(Debug, Serialize)]
struct FullSuiteOutput {
    benchmark_type: String,
    audio: String,
    results: std::collections::HashMap<String, AggregatedResult>,
}

struct BenchmarkRunner {
    params: BenchmarkParams,
}

impl BenchmarkRunner {
    fn new(params: BenchmarkParams) -> Self {
        Self { params }
    }

    fn benchmark_spectral_centroid(
        &self,
        audio_path: &str,
    ) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        let buffer = audio::load(audio_path)?;

        let frame_cfg = FrameConfig {
            frame_size: self.params.n_fft.unwrap_or(2048),
            hop_size: self.params.hop_length.unwrap_or(512),
            center: true, // Match librosa's default behavior
            ..Default::default()
        };

        let frames = frame(&buffer, frame_cfg)?;

        let stft_cfg = StftConfig {
            n_fft: self.params.n_fft.unwrap_or(2048),
            ..Default::default()
        };

        let t_start = Instant::now();
        let spec = stft(&frames, stft_cfg)?;
        let centroid = spectral_centroid(&spec);
        let t_end = Instant::now();

        let values = centroid.values();
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
        let std = variance.sqrt();
        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        Ok(BenchmarkResult {
            r#impl: "rsona".to_string(),
            feature: "spectral_centroid".to_string(),
            runtime_ms: (t_end - t_start).as_secs_f64() * 1000.0,
            result_summary: Some(ResultSummary {
                mean: Some(mean),
                std: Some(std),
                min: Some(min),
                max: Some(max),
                shape: Some(vec![values.len()]),
                value: None,
            }),
            output: Some(serde_json::to_value(values)?),
        })
    }

    fn benchmark_spectral_bandwidth(
        &self,
        audio_path: &str,
    ) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        let buffer = audio::load(audio_path)?;

        let frame_cfg = FrameConfig {
            frame_size: self.params.n_fft.unwrap_or(2048),
            hop_size: self.params.hop_length.unwrap_or(512),
            center: true, // Match librosa's default behavior
            ..Default::default()
        };

        let frames = frame(&buffer, frame_cfg)?;

        let stft_cfg = StftConfig {
            n_fft: self.params.n_fft.unwrap_or(2048),
            ..Default::default()
        };

        let t_start = Instant::now();
        let spec = stft(&frames, stft_cfg)?;
        let bandwidth = spectral_bandwidth(&spec);
        let t_end = Instant::now();

        let values = bandwidth.values();
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
        let std = variance.sqrt();
        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        Ok(BenchmarkResult {
            r#impl: "rsona".to_string(),
            feature: "spectral_bandwidth".to_string(),
            runtime_ms: (t_end - t_start).as_secs_f64() * 1000.0,
            result_summary: Some(ResultSummary {
                mean: Some(mean),
                std: Some(std),
                min: Some(min),
                max: Some(max),
                shape: Some(vec![values.len()]),
                value: None,
            }),
            output: Some(serde_json::to_value(values)?),
        })
    }

    fn benchmark_spectral_rolloff(
        &self,
        audio_path: &str,
    ) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        let buffer = audio::load(audio_path)?;

        let frame_cfg = FrameConfig {
            frame_size: self.params.n_fft.unwrap_or(2048),
            hop_size: self.params.hop_length.unwrap_or(512),
            center: true, // Match librosa's default behavior
            ..Default::default()
        };

        let frames = frame(&buffer, frame_cfg)?;

        let stft_cfg = StftConfig {
            n_fft: self.params.n_fft.unwrap_or(2048),
            ..Default::default()
        };

        let t_start = Instant::now();
        let spec = stft(&frames, stft_cfg)?;
        let rolloff = spectral_rolloff(&spec, self.params.roll_percent.unwrap_or(0.85));
        let t_end = Instant::now();

        let values = rolloff.values();
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
        let std = variance.sqrt();
        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        Ok(BenchmarkResult {
            r#impl: "rsona".to_string(),
            feature: "spectral_rolloff".to_string(),
            runtime_ms: (t_end - t_start).as_secs_f64() * 1000.0,
            result_summary: Some(ResultSummary {
                mean: Some(mean),
                std: Some(std),
                min: Some(min),
                max: Some(max),
                shape: Some(vec![values.len()]),
                value: None,
            }),
            output: Some(serde_json::to_value(values)?),
        })
    }

    fn benchmark_spectral_flux(
        &self,
        audio_path: &str,
    ) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        let buffer = audio::load(audio_path)?;

        let frame_cfg = FrameConfig {
            frame_size: self.params.n_fft.unwrap_or(2048),
            hop_size: self.params.hop_length.unwrap_or(512),
            center: true, // Match librosa's default behavior
            ..Default::default()
        };

        let frames = frame(&buffer, frame_cfg)?;

        let stft_cfg = StftConfig {
            n_fft: self.params.n_fft.unwrap_or(2048),
            ..Default::default()
        };

        let onset_cfg = OnsetConfig {
            mel: MelConfig {
                n_mels: self.params.n_mels.unwrap_or(128),
                ..Default::default()
            },
            ..Default::default()
        };

        let t_start = Instant::now();
        let spec = stft(&frames, stft_cfg)?;
        let flux = onset_strength(&spec, onset_cfg);
        let t_end = Instant::now();

        let values = flux.values();
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
        let std = variance.sqrt();
        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        Ok(BenchmarkResult {
            r#impl: "rsona".to_string(),
            feature: "spectral_flux".to_string(),
            runtime_ms: (t_end - t_start).as_secs_f64() * 1000.0,
            result_summary: Some(ResultSummary {
                mean: Some(mean),
                std: Some(std),
                min: Some(min),
                max: Some(max),
                shape: Some(vec![values.len()]),
                value: None,
            }),
            output: Some(serde_json::to_value(values)?),
        })
    }

    fn benchmark_onset_strength(
        &self,
        audio_path: &str,
    ) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        let buffer = audio::load(audio_path)?;

        let frame_cfg = FrameConfig {
            frame_size: self.params.n_fft.unwrap_or(2048),
            hop_size: self.params.hop_length.unwrap_or(512),
            center: true, // Match librosa's default behavior
            ..Default::default()
        };

        let frames = frame(&buffer, frame_cfg)?;

        let stft_cfg = StftConfig {
            n_fft: self.params.n_fft.unwrap_or(2048),
            ..Default::default()
        };

        let onset_cfg = OnsetConfig {
            mel: MelConfig {
                n_mels: self.params.n_mels.unwrap_or(128),
                ..Default::default()
            },
            ..Default::default()
        };

        let t_start = Instant::now();
        let spec = stft(&frames, stft_cfg)?;
        let onset = onset_strength(&spec, onset_cfg);
        let t_end = Instant::now();

        let values = onset.values();
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
        let std = variance.sqrt();
        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        Ok(BenchmarkResult {
            r#impl: "rsona".to_string(),
            feature: "onset_strength".to_string(),
            runtime_ms: (t_end - t_start).as_secs_f64() * 1000.0,
            result_summary: Some(ResultSummary {
                mean: Some(mean),
                std: Some(std),
                min: Some(min),
                max: Some(max),
                shape: Some(vec![values.len()]),
                value: None,
            }),
            output: Some(serde_json::to_value(values)?),
        })
    }

    fn benchmark_tempo(
        &self,
        audio_path: &str,
    ) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        let buffer = audio::load(audio_path)?;

        let frame_cfg = FrameConfig {
            frame_size: self.params.n_fft.unwrap_or(2048),
            hop_size: self.params.hop_length.unwrap_or(512),
            center: true, // Match librosa's default behavior
            ..Default::default()
        };

        let frames = frame(&buffer, frame_cfg)?;

        let stft_cfg = StftConfig {
            n_fft: self.params.n_fft.unwrap_or(2048),
            ..Default::default()
        };

        let onset_cfg = OnsetConfig {
            mel: MelConfig {
                n_mels: self.params.n_mels.unwrap_or(128),
                ..Default::default()
            },
            ..Default::default()
        };

        let t_start = Instant::now();
        let spec = stft(&frames, stft_cfg)?;
        let onset = onset_strength(&spec, onset_cfg);
        let tempo = estimate_tempo(
            onset.values(),
            frames.sample_rate(),
            frames.hop_size(),
            Default::default(),
        );
        let t_end = Instant::now();

        Ok(BenchmarkResult {
            r#impl: "rsona".to_string(),
            feature: "tempo".to_string(),
            runtime_ms: (t_end - t_start).as_secs_f64() * 1000.0,
            result_summary: Some(ResultSummary {
                mean: None,
                std: None,
                min: None,
                max: None,
                shape: None,
                value: Some(tempo.bpm),
            }),
            output: Some(serde_json::to_value(tempo.bpm)?),
        })
    }

    fn benchmark_chroma_stft(
        &self,
        audio_path: &str,
    ) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        let buffer = audio::load(audio_path)?;

        let frame_cfg = FrameConfig {
            frame_size: self.params.n_fft.unwrap_or(2048),
            hop_size: self.params.hop_length.unwrap_or(512),
            center: true, // Match librosa's default behavior
            ..Default::default()
        };

        let frames = frame(&buffer, frame_cfg)?;

        let stft_cfg = StftConfig {
            n_fft: self.params.n_fft.unwrap_or(2048),
            ..Default::default()
        };

        let chroma_cfg = ChromaConfig {
            norm: ChromaNorm::Max,
            ..Default::default()
        };

        let t_start = Instant::now();
        let spec = stft(&frames, stft_cfg)?;
        let chroma = chroma_stft(&spec, chroma_cfg);
        let t_end = Instant::now();

        let values = chroma.as_slice();
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
        let std = variance.sqrt();
        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        Ok(BenchmarkResult {
            r#impl: "rsona".to_string(),
            feature: "chroma_stft".to_string(),
            runtime_ms: (t_end - t_start).as_secs_f64() * 1000.0,
            result_summary: Some(ResultSummary {
                mean: Some(mean),
                std: Some(std),
                min: Some(min),
                max: Some(max),
                shape: Some(vec![chroma.n_chroma(), chroma.n_frames()]),
                value: None,
            }),
            output: Some(serde_json::to_value(values)?),
        })
    }

    fn benchmark_mfcc(
        &self,
        audio_path: &str,
    ) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        let buffer = audio::load(audio_path)?;

        let frame_cfg = FrameConfig {
            frame_size: self.params.n_fft.unwrap_or(2048),
            hop_size: self.params.hop_length.unwrap_or(512),
            center: true, // Match librosa's default behavior
            ..Default::default()
        };

        let frames = frame(&buffer, frame_cfg)?;

        let stft_cfg = StftConfig {
            n_fft: self.params.n_fft.unwrap_or(2048),
            ..Default::default()
        };

        let mel_cfg = MelConfig {
            n_mels: self.params.n_mels.unwrap_or(128),
            ..Default::default()
        };

        let mfcc_cfg = MfccConfig {
            n_mfcc: self.params.n_mfcc.unwrap_or(13),
            ..Default::default()
        };

        let t_start = Instant::now();
        let spec = stft(&frames, stft_cfg)?;
        let mel = mel_spectrogram(&spec, mel_cfg);
        let mfcc_result = mfcc(&mel, mfcc_cfg);
        let t_end = Instant::now();

        let values = mfcc_result.as_slice();
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
        let std = variance.sqrt();
        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        Ok(BenchmarkResult {
            r#impl: "rsona".to_string(),
            feature: "mfcc".to_string(),
            runtime_ms: (t_end - t_start).as_secs_f64() * 1000.0,
            result_summary: Some(ResultSummary {
                mean: Some(mean),
                std: Some(std),
                min: Some(min),
                max: Some(max),
                shape: Some(vec![mfcc_result.n_frames(), mfcc_result.n_mfcc()]),
                value: None,
            }),
            output: Some(serde_json::to_value(values)?),
        })
    }

    fn benchmark_full_pipeline(
        &self,
        audio_path: &str,
    ) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        let buffer = audio::load(audio_path)?;

        let frame_cfg = FrameConfig {
            frame_size: self.params.n_fft.unwrap_or(2048),
            hop_size: self.params.hop_length.unwrap_or(512),
            center: true, // Match librosa's default behavior
            ..Default::default()
        };

        let frames = frame(&buffer, frame_cfg)?;

        let stft_cfg = StftConfig {
            n_fft: self.params.n_fft.unwrap_or(2048),
            ..Default::default()
        };

        let mel_cfg = MelConfig {
            n_mels: self.params.n_mels.unwrap_or(128),
            ..Default::default()
        };

        let mfcc_cfg = MfccConfig {
            n_mfcc: self.params.n_mfcc.unwrap_or(13),
            ..Default::default()
        };

        let onset_cfg = OnsetConfig {
            mel: mel_cfg.clone(),
            ..Default::default()
        };

        let mut breakdown = PipelineBreakdown {
            stft_ms: 0.0,
            mel_ms: 0.0,
            mfcc_ms: 0.0,
            centroid_ms: 0.0,
            flux_ms: 0.0,
            onset_ms: 0.0,
            tempo_ms: 0.0,
        };

        // STFT
        let t0 = Instant::now();
        let spec = stft(&frames, stft_cfg)?;
        breakdown.stft_ms = (Instant::now() - t0).as_secs_f64() * 1000.0;

        // Mel Spectrogram
        let t0 = Instant::now();
        let mel = mel_spectrogram(&spec, mel_cfg);
        breakdown.mel_ms = (Instant::now() - t0).as_secs_f64() * 1000.0;

        // MFCC
        let t0 = Instant::now();
        let mfcc_result = mfcc(&mel, mfcc_cfg);
        breakdown.mfcc_ms = (Instant::now() - t0).as_secs_f64() * 1000.0;

        // Spectral Centroid
        let t0 = Instant::now();
        let _centroid = spectral_centroid(&spec);
        breakdown.centroid_ms = (Instant::now() - t0).as_secs_f64() * 1000.0;

        // Spectral Flux
        let t0 = Instant::now();
        let _flux = spectral_flux(&spec);
        breakdown.flux_ms = (Instant::now() - t0).as_secs_f64() * 1000.0;

        // Onset Strength
        let t0 = Instant::now();
        let onset = onset_strength(&spec, onset_cfg);
        breakdown.onset_ms = (Instant::now() - t0).as_secs_f64() * 1000.0;

        // Tempo
        let t0 = Instant::now();
        let tempo = estimate_tempo(
            onset.values(),
            frames.sample_rate(),
            frames.hop_size(),
            Default::default(),
        );
        breakdown.tempo_ms = (Instant::now() - t0).as_secs_f64() * 1000.0;

        let total_ms = breakdown.stft_ms
            + breakdown.mel_ms
            + breakdown.mfcc_ms
            + breakdown.centroid_ms
            + breakdown.flux_ms
            + breakdown.onset_ms
            + breakdown.tempo_ms;

        Ok(BenchmarkResult {
            r#impl: "rsona".to_string(),
            feature: "full_pipeline".to_string(),
            runtime_ms: total_ms,
            result_summary: Some(ResultSummary {
                mean: None,
                std: None,
                min: None,
                max: None,
                shape: None,
                value: Some(tempo.bpm),
            }),
            output: Some(serde_json::json!({
                "breakdown": breakdown,
                "n_frames": mfcc_result.n_frames(),
                "tempo_bpm": tempo.bpm,
            })),
        })
    }

    fn run_feature_benchmark(
        &self,
        feature: &str,
        audio_path: &str,
        runs: usize,
        warmup: usize,
    ) -> Result<Vec<BenchmarkResult>, Box<dyn std::error::Error>> {
        // Warmup runs
        for _ in 0..warmup {
            let _ = match feature {
                "spectral_centroid" => self.benchmark_spectral_centroid(audio_path),
                "spectral_bandwidth" => self.benchmark_spectral_bandwidth(audio_path),
                "spectral_rolloff" => self.benchmark_spectral_rolloff(audio_path),
                "spectral_flux" => self.benchmark_spectral_flux(audio_path),
                "onset_strength" => self.benchmark_onset_strength(audio_path),
                "tempo" => self.benchmark_tempo(audio_path),
                "chroma_stft" => self.benchmark_chroma_stft(audio_path),
                "mfcc" => self.benchmark_mfcc(audio_path),
                "full_pipeline" => self.benchmark_full_pipeline(audio_path),
                _ => return Err(format!("Unknown feature: {}", feature).into()),
            };
        }

        // Actual benchmark runs
        let mut results = Vec::new();
        for _ in 0..runs {
            let result = match feature {
                "spectral_centroid" => self.benchmark_spectral_centroid(audio_path)?,
                "spectral_bandwidth" => self.benchmark_spectral_bandwidth(audio_path)?,
                "spectral_rolloff" => self.benchmark_spectral_rolloff(audio_path)?,
                "spectral_flux" => self.benchmark_spectral_flux(audio_path)?,
                "onset_strength" => self.benchmark_onset_strength(audio_path)?,
                "tempo" => self.benchmark_tempo(audio_path)?,
                "chroma_stft" => self.benchmark_chroma_stft(audio_path)?,
                "mfcc" => self.benchmark_mfcc(audio_path)?,
                "full_pipeline" => self.benchmark_full_pipeline(audio_path)?,
                _ => return Err(format!("Unknown feature: {}", feature).into()),
            };
            results.push(result);
        }

        Ok(results)
    }

    fn calculate_statistics(
        &self,
        results: &[BenchmarkResult],
    ) -> Result<AggregatedResult, Box<dyn std::error::Error>> {
        if results.is_empty() {
            return Err("No results to aggregate".into());
        }

        let runtimes: Vec<f64> = results.iter().map(|r| r.runtime_ms).collect();

        let mean = runtimes.iter().sum::<f64>() / runtimes.len() as f64;
        let variance =
            runtimes.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / runtimes.len() as f64;
        let std = variance.sqrt();

        let mut sorted = runtimes.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        let min = sorted[0];
        let max = sorted[sorted.len() - 1];

        Ok(AggregatedResult {
            r#impl: "rsona".to_string(),
            feature: results[0].feature.clone(),
            runtime_ms: RuntimeStats {
                mean,
                median,
                std,
                min,
                max,
            },
            num_runs: results.len(),
            result_summary: results[0].result_summary.clone(),
            output: results.last().unwrap().output.clone(),
            breakdown: None,
        })
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let mut audio_path: Option<String> = None;
    let mut feature: Option<String> = None;
    let mut runs: usize = 10;
    let mut warmup: usize = 2;
    let mut output_path: Option<String> = None;
    let mut params = BenchmarkParams::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--audio" => {
                audio_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--feature" => {
                feature = Some(args[i + 1].clone());
                i += 2;
            }
            "--runs" => {
                runs = args[i + 1].parse()?;
                i += 2;
            }
            "--warmup" => {
                warmup = args[i + 1].parse()?;
                i += 2;
            }
            "--output" => {
                output_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--n-fft" => {
                params.n_fft = Some(args[i + 1].parse()?);
                i += 2;
            }
            "--hop-length" => {
                params.hop_length = Some(args[i + 1].parse()?);
                i += 2;
            }
            "--n-mels" => {
                params.n_mels = Some(args[i + 1].parse()?);
                i += 2;
            }
            "--n-mfcc" => {
                params.n_mfcc = Some(args[i + 1].parse()?);
                i += 2;
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                i += 1;
            }
        }
    }

    let audio_path = audio_path.ok_or("--audio is required")?;

    let runner = BenchmarkRunner::new(params);

    if let Some(feat) = feature {
        // Single feature benchmark
        eprintln!("Running benchmark for feature: {}", feat);
        let results = runner.run_feature_benchmark(&feat, &audio_path, runs, warmup)?;
        let stats = runner.calculate_statistics(&results)?;

        let output_data = SingleFeatureOutput {
            benchmark_type: "single_feature".to_string(),
            feature: feat.clone(),
            audio: audio_path.clone(),
            statistics: stats,
            all_runs: results,
        };

        let output_json = serde_json::to_string_pretty(&output_data)?;

        if let Some(path) = output_path {
            std::fs::write(&path, output_json)?;
            eprintln!("Results written to {}", path);
        } else {
            println!("{}", output_json);
        }
    } else {
        // Run all supported features
        eprintln!("Running full benchmark suite...");

        let features = vec![
            "spectral_centroid",
            "spectral_bandwidth",
            "spectral_rolloff",
            "spectral_flux",
            "onset_strength",
            "tempo",
            "chroma_stft",
            "mfcc",
            "full_pipeline",
        ];

        let mut all_results = std::collections::HashMap::new();

        for feat in features {
            eprintln!("Running {}...", feat);
            match runner.run_feature_benchmark(feat, &audio_path, runs, warmup) {
                Ok(results) => match runner.calculate_statistics(&results) {
                    Ok(stats) => {
                        all_results.insert(feat.to_string(), stats);
                    }
                    Err(e) => {
                        eprintln!("Error calculating stats for {}: {}", feat, e);
                    }
                },
                Err(e) => {
                    eprintln!("Error running {}: {}", feat, e);
                }
            }
        }

        let output_data = FullSuiteOutput {
            benchmark_type: "full_suite".to_string(),
            audio: audio_path.clone(),
            results: all_results,
        };

        let output_json = serde_json::to_string_pretty(&output_data)?;

        if let Some(path) = output_path {
            std::fs::write(&path, output_json)?;
            eprintln!("Results written to {}", path);
        } else {
            println!("{}", output_json);
        }
    }

    Ok(())
}
