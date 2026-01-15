#!/usr/bin/env python3
"""
Librosa Benchmark Runner for rsona Comparison

This script runs librosa benchmarks following the shared benchmark specification.
All output is structured JSON - no markdown or prose.

Usage:
    python librosa_runner.py --spec benchmark_spec.json --audio path/to/audio.wav
    python librosa_runner.py --feature spectral_centroid --audio test.wav --runs 10
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import librosa.beat
import librosa.feature
import librosa.onset
import numpy as np
from scipy.stats import pearsonr

# Suppress librosa warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class LibrosaBenchmarkRunner:
    """Executes librosa benchmarks with structured output."""

    def __init__(self, spec_path: Optional[str] = None):
        """Initialize with optional benchmark spec."""
        self.spec = None
        if spec_path:
            with open(spec_path) as f:
                self.spec = json.load(f)

    def load_audio(
        self, audio_path: str, sr: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """Load audio file with librosa. Uses native sample rate if sr=None."""
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
        return y, sr

    def benchmark_spectral_centroid(
        self, y: np.ndarray, sr: int, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Benchmark spectral centroid calculation."""
        n_fft = params.get("n_fft", 2048)
        hop_length = params.get("hop_length", 512)

        t_start = time.perf_counter()
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        centroid = librosa.feature.spectral_centroid(
            S=S, sr=sr, n_fft=n_fft, hop_length=hop_length
        )
        t_end = time.perf_counter()

        return {
            "impl": "librosa",
            "feature": "spectral_centroid",
            "runtime_ms": (t_end - t_start) * 1000.0,
            "result_summary": {
                "mean": float(np.mean(centroid)),
                "std": float(np.std(centroid)),
                "min": float(np.min(centroid)),
                "max": float(np.max(centroid)),
                "shape": list(centroid.shape),
            },
            "output": centroid[0].tolist(),
        }

    def benchmark_spectral_bandwidth(
        self, y: np.ndarray, sr: int, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Benchmark spectral bandwidth calculation."""
        n_fft = params.get("n_fft", 2048)
        hop_length = params.get("hop_length", 512)

        t_start = time.perf_counter()
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        bandwidth = librosa.feature.spectral_bandwidth(
            S=S, sr=sr, n_fft=n_fft, hop_length=hop_length
        )
        t_end = time.perf_counter()

        return {
            "impl": "librosa",
            "feature": "spectral_bandwidth",
            "runtime_ms": (t_end - t_start) * 1000.0,
            "result_summary": {
                "mean": float(np.mean(bandwidth)),
                "std": float(np.std(bandwidth)),
                "min": float(np.min(bandwidth)),
                "max": float(np.max(bandwidth)),
                "shape": list(bandwidth.shape),
            },
            "output": bandwidth[0].tolist(),
        }

    def benchmark_spectral_rolloff(
        self, y: np.ndarray, sr: int, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Benchmark spectral rolloff calculation."""
        n_fft = params.get("n_fft", 2048)
        hop_length = params.get("hop_length", 512)
        roll_percent = params.get("roll_percent", 0.85)

        t_start = time.perf_counter()
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        rolloff = librosa.feature.spectral_rolloff(
            S=S, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=roll_percent
        )
        t_end = time.perf_counter()

        return {
            "impl": "librosa",
            "feature": "spectral_rolloff",
            "runtime_ms": (t_end - t_start) * 1000.0,
            "result_summary": {
                "mean": float(np.mean(rolloff)),
                "std": float(np.std(rolloff)),
                "min": float(np.min(rolloff)),
                "max": float(np.max(rolloff)),
                "shape": list(rolloff.shape),
            },
            "output": rolloff[0].tolist(),
        }

    def benchmark_spectral_flux(
        self, y: np.ndarray, sr: int, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Benchmark spectral flux calculation (onset strength from mel spectrogram)."""
        hop_length = params.get("hop_length", 512)

        t_start = time.perf_counter()
        flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        t_end = time.perf_counter()

        return {
            "impl": "librosa",
            "feature": "spectral_flux",
            "runtime_ms": (t_end - t_start) * 1000.0,
            "result_summary": {
                "mean": float(np.mean(flux)),
                "std": float(np.std(flux)),
                "min": float(np.min(flux)),
                "max": float(np.max(flux)),
                "shape": list(flux.shape),
            },
            "output": flux.tolist(),
        }

    def benchmark_onset_strength(
        self, y: np.ndarray, sr: int, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Benchmark onset strength calculation."""
        hop_length = params.get("hop_length", 512)

        t_start = time.perf_counter()
        onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        t_end = time.perf_counter()

        return {
            "impl": "librosa",
            "feature": "onset_strength",
            "runtime_ms": (t_end - t_start) * 1000.0,
            "result_summary": {
                "mean": float(np.mean(onset)),
                "std": float(np.std(onset)),
                "min": float(np.min(onset)),
                "max": float(np.max(onset)),
                "shape": list(onset.shape),
            },
            "output": onset.tolist(),
        }

    def benchmark_tempo(
        self, y: np.ndarray, sr: int, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Benchmark tempo estimation."""
        hop_length = params.get("hop_length", 512)

        t_start = time.perf_counter()
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        tempo = librosa.beat.tempo(
            onset_envelope=onset_env, sr=sr, hop_length=hop_length
        )[0]
        t_end = time.perf_counter()

        return {
            "impl": "librosa",
            "feature": "tempo",
            "runtime_ms": (t_end - t_start) * 1000.0,
            "result_summary": {
                "value": float(tempo),
            },
            "output": float(tempo),
        }

    def benchmark_chroma_stft(
        self, y: np.ndarray, sr: int, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Benchmark chroma STFT calculation."""
        n_fft = params.get("n_fft", 2048)
        hop_length = params.get("hop_length", 512)
        n_chroma = params.get("n_chroma", 12)

        t_start = time.perf_counter()
        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_chroma=n_chroma
        )
        t_end = time.perf_counter()

        return {
            "impl": "librosa",
            "feature": "chroma_stft",
            "runtime_ms": (t_end - t_start) * 1000.0,
            "result_summary": {
                "mean": float(np.mean(chroma)),
                "std": float(np.std(chroma)),
                "min": float(np.min(chroma)),
                "max": float(np.max(chroma)),
                "shape": list(chroma.shape),
            },
            "output": chroma.tolist(),
        }

    def benchmark_mfcc(
        self, y: np.ndarray, sr: int, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Benchmark MFCC calculation."""
        n_fft = params.get("n_fft", 2048)
        hop_length = params.get("hop_length", 512)
        n_mels = params.get("n_mels", 128)
        n_mfcc = params.get("n_mfcc", 13)

        t_start = time.perf_counter()
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc
        )
        t_end = time.perf_counter()

        return {
            "impl": "librosa",
            "feature": "mfcc",
            "runtime_ms": (t_end - t_start) * 1000.0,
            "result_summary": {
                "mean": float(np.mean(mfcc)),
                "std": float(np.std(mfcc)),
                "min": float(np.min(mfcc)),
                "max": float(np.max(mfcc)),
                "shape": list(mfcc.shape),
            },
            "output": mfcc.tolist(),
        }

    def benchmark_full_pipeline(
        self, y: np.ndarray, sr: int, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Benchmark complete feature extraction pipeline."""
        n_fft = params.get("n_fft", 2048)
        hop_length = params.get("hop_length", 512)
        n_mels = params.get("n_mels", 128)
        n_mfcc = params.get("n_mfcc", 13)

        breakdown = {}

        # STFT
        t0 = time.perf_counter()
        S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        t1 = time.perf_counter()
        breakdown["stft_ms"] = (t1 - t0) * 1000.0

        # Mel Spectrogram
        t0 = time.perf_counter()
        mel = librosa.feature.melspectrogram(
            S=np.abs(S) ** 2, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        t1 = time.perf_counter()
        breakdown["mel_ms"] = (t1 - t0) * 1000.0

        # MFCC
        t0 = time.perf_counter()
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=n_mfcc)
        t1 = time.perf_counter()
        breakdown["mfcc_ms"] = (t1 - t0) * 1000.0

        # Spectral Centroid
        t0 = time.perf_counter()
        centroid = librosa.feature.spectral_centroid(
            S=np.abs(S), sr=sr, n_fft=n_fft, hop_length=hop_length
        )
        t1 = time.perf_counter()
        breakdown["centroid_ms"] = (t1 - t0) * 1000.0

        # Spectral Flux (using onset strength)
        t0 = time.perf_counter()
        flux = librosa.onset.onset_strength(
            S=np.abs(S) ** 2, sr=sr, hop_length=hop_length
        )
        t1 = time.perf_counter()
        breakdown["flux_ms"] = (t1 - t0) * 1000.0

        # Onset Strength
        t0 = time.perf_counter()
        onset = librosa.onset.onset_strength(S=mel, sr=sr, hop_length=hop_length)
        t1 = time.perf_counter()
        breakdown["onset_ms"] = (t1 - t0) * 1000.0

        # Tempo
        t0 = time.perf_counter()
        tempo = librosa.beat.tempo(onset_envelope=onset, sr=sr, hop_length=hop_length)[
            0
        ]
        t1 = time.perf_counter()
        breakdown["tempo_ms"] = (t1 - t0) * 1000.0

        total_ms = sum(breakdown.values())

        return {
            "impl": "librosa",
            "feature": "full_pipeline",
            "runtime_ms": total_ms,
            "breakdown": breakdown,
            "result_summary": {
                "n_frames": int(mfcc.shape[1]),
                "tempo_bpm": float(tempo),
            },
        }

    def run_feature_benchmark(
        self,
        feature: str,
        audio_path: str,
        params: Dict[str, Any],
        runs: int = 10,
        warmup: int = 2,
    ) -> List[Dict[str, Any]]:
        """Run a single feature benchmark multiple times."""
        # Load audio once (use native sample rate for fair comparison with rsona)
        y, sr = self.load_audio(audio_path, sr=None)

        # Map feature name to benchmark method
        feature_map = {
            "spectral_centroid": self.benchmark_spectral_centroid,
            "spectral_bandwidth": self.benchmark_spectral_bandwidth,
            "spectral_rolloff": self.benchmark_spectral_rolloff,
            "spectral_flux": self.benchmark_spectral_flux,
            "onset_strength": self.benchmark_onset_strength,
            "tempo": self.benchmark_tempo,
            "chroma_stft": self.benchmark_chroma_stft,
            "mfcc": self.benchmark_mfcc,
            "full_pipeline": self.benchmark_full_pipeline,
        }

        if feature not in feature_map:
            raise ValueError(f"Unknown feature: {feature}")

        benchmark_func = feature_map[feature]

        # Warmup runs
        for _ in range(warmup):
            try:
                benchmark_func(y, sr, params)
            except Exception:
                pass

        # Actual benchmark runs
        results = []
        for _ in range(runs):
            try:
                result = benchmark_func(y, sr, params)
                results.append(result)
            except Exception as e:
                print(f"Error in run: {e}", file=sys.stderr)

        return results

    def calculate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate statistics from multiple runs."""
        if not results:
            return {}

        runtimes = [r["runtime_ms"] for r in results]

        stats = {
            "impl": "librosa",
            "feature": results[0]["feature"],
            "runtime_ms": {
                "mean": float(np.mean(runtimes)),
                "median": float(np.median(runtimes)),
                "std": float(np.std(runtimes)),
                "min": float(np.min(runtimes)),
                "max": float(np.max(runtimes)),
            },
            "num_runs": len(results),
        }

        # Include result summary from first run
        if "result_summary" in results[0]:
            stats["result_summary"] = results[0]["result_summary"]

        # Include output from last run for correctness checking
        if "output" in results[-1]:
            stats["output"] = results[-1]["output"]

        # Include breakdown if present
        if "breakdown" in results[0]:
            breakdown_keys = results[0]["breakdown"].keys()
            breakdown_stats = {}
            for key in breakdown_keys:
                values = [r["breakdown"][key] for r in results]
                breakdown_stats[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
            stats["breakdown"] = breakdown_stats

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Librosa benchmark runner for rsona comparison"
    )
    parser.add_argument(
        "--spec",
        type=str,
        help="Path to benchmark specification JSON",
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file",
    )
    parser.add_argument(
        "--feature",
        type=str,
        help="Single feature to benchmark (overrides spec)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of benchmark runs",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup runs",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file (default: stdout)",
    )
    parser.add_argument(
        "--params",
        type=str,
        help="JSON string of additional parameters",
    )

    args = parser.parse_args()

    runner = LibrosaBenchmarkRunner(args.spec)

    # Parse additional params
    params = {}
    if args.params:
        params = json.loads(args.params)

    # Set default params if not provided
    params.setdefault("sample_rate", 22050)
    params.setdefault("n_fft", 2048)
    params.setdefault("hop_length", 512)
    params.setdefault("n_mels", 128)
    params.setdefault("n_mfcc", 13)

    # Run benchmark
    if args.feature:
        # Single feature benchmark
        results = runner.run_feature_benchmark(
            args.feature, args.audio, params, args.runs, args.warmup
        )
        stats = runner.calculate_statistics(results)
        output_data = {
            "benchmark_type": "single_feature",
            "feature": args.feature,
            "audio": args.audio,
            "statistics": stats,
            "all_runs": results,
        }
    else:
        # Run all features from spec
        if not runner.spec:
            print("Error: --spec required when not using --feature", file=sys.stderr)
            sys.exit(1)

        all_results = {}
        isolated_tests = runner.spec["test_suites"][0]["tests"]

        for test in isolated_tests:
            feature = test["feature"]
            if test["status"] != "ready" or not test["librosa_available"]:
                continue

            print(f"Running {feature}...", file=sys.stderr)
            test_params = {**params, **test["params"]}

            results = runner.run_feature_benchmark(
                feature, args.audio, test_params, args.runs, args.warmup
            )
            stats = runner.calculate_statistics(results)
            all_results[feature] = stats

        output_data = {
            "benchmark_type": "full_suite",
            "audio": args.audio,
            "results": all_results,
        }

    # Output results
    output_json = json.dumps(output_data, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
