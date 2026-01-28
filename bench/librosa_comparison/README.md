# rsona vs librosa Benchmark Suite

A comprehensive, rigorous benchmarking system for comparing rsona (Rust) against librosa (Python) in audio analysis performance and data correctness.

This suite implements **feature-to-feature**, **apples-to-apples** comparisons using shared specifications, structured JSON output, and automated validation.

## Design Principles

- **No markdown prose**: Results are machine-readable artifacts (JSON, CSV, charts)
- **Shared specification**: Both implementations use identical parameters
- **Correctness-first**: Performance means nothing without data accuracy
- **Reusable code paths**: Benchmarks use production rsona code, not specialized fast paths
- **Coming Soon transparency**: Features without 1:1 parity are explicitly marked

## Quick Start

```bash
# Install Python dependencies
pip install librosa numpy scipy matplotlib

# Run complete benchmark suite
cd rsona/bench/librosa_comparison
chmod +x run_benchmarks.sh
./run_benchmarks.sh --audio /path/to/audio.wav

# View results
cat artifacts/performance_analysis.json | jq
open artifacts/runtime_comparison.png
```

## Architecture

```
librosa_comparison/
├── benchmark_spec.json          # Shared specification (params, thresholds)
├── librosa_runner.py            # Python benchmark executor
├── rsona_runner.rs              # Rust benchmark executor
├── validate_correctness.py      # Numerical validation tool
├── analyze_performance.py       # Performance analysis + artifact generator
├── run_benchmarks.sh            # Master orchestration script
└── artifacts/                   # Generated outputs (gitignored)
    ├── librosa_results.json     # Raw librosa benchmark data
    ├── rsona_results.json       # Raw rsona benchmark data
    ├── validation_results.json  # Correctness validation
    ├── performance_analysis.json# Performance comparison
    ├── performance_comparison.csv
    ├── runtime_comparison.png
    ├── speedup_comparison.png
    └── feature_parity_matrix.png
```

## Benchmark Specification

The `benchmark_spec.json` file defines:

- **Common parameters**: `n_fft`, `hop_length`, `sample_rate`, etc.
- **Feature tests**: Which features to benchmark and their status
- **Correctness thresholds**: MAE and correlation limits for validation
- **Flow tests**: End-to-end pipeline benchmarks
- **Scaling tests**: Performance vs audio duration

### Feature Status Values

- `ready`: 1:1 parity exists, benchmark implemented
- `coming_soon`: No true equivalence yet (marked in output)

## Running Benchmarks

### Full Suite

```bash
./run_benchmarks.sh \
  --audio test.wav \
  --runs 10 \
  --warmup 2 \
  --output-dir artifacts \
  --native-cpu
```

### Single Feature

```bash
./run_benchmarks.sh \
  --audio test.wav \
  --feature spectral_centroid \
  --runs 20
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--audio PATH` | Audio file to benchmark | First file in `bench/audio/` |
| `--runs N` | Number of benchmark iterations | 10 |
| `--warmup N` | Warmup iterations (discarded) | 2 |
| `--output-dir PATH` | Artifact output directory | `artifacts/` |
| `--feature NAME` | Benchmark single feature only | All features |
| `--native-cpu` | Build with `-C target-cpu=native` | Off |
| `--skip-build` | Skip Rust compilation | Off |
| `--skip-validation` | Skip correctness checks | Off |
| `--clean` | Remove previous artifacts first | Off |
| `--verbose` | Enable verbose logging | Off |

## Individual Tools

### librosa Runner

```bash
python3 librosa_runner.py \
  --audio test.wav \
  --feature spectral_centroid \
  --runs 10 \
  --output librosa_results.json
```

### rsona Runner

```bash
cargo run --release --bin rsona_runner -- \
  --audio test.wav \
  --feature spectral_centroid \
  --runs 10 \
  --output rsona_results.json
```

### Correctness Validator

```bash
python3 validate_correctness.py \
  --rsona rsona_results.json \
  --librosa librosa_results.json \
  --spec benchmark_spec.json \
  --output validation.json \
  --fail-on-error
```

### Performance Analyzer

```bash
python3 analyze_performance.py \
  --rsona rsona_results.json \
  --librosa librosa_results.json \
  --spec benchmark_spec.json \
  --output-dir artifacts \
  --format all
```

## Output Artifacts

### JSON Results

#### Benchmark Output Schema

```json
{
  "benchmark_type": "full_suite",
  "audio": "path/to/audio.wav",
  "results": {
    "spectral_centroid": {
      "impl": "rsona",
      "feature": "spectral_centroid",
      "runtime_ms": {
        "mean": 12.34,
        "median": 12.28,
        "std": 0.45,
        "min": 11.89,
        "max": 13.12
      },
      "num_runs": 10,
      "result_summary": {
        "mean": 1234.56,
        "std": 89.12,
        "shape": [128]
      },
      "output": [...]
    }
  }
}
```

#### Validation Output Schema

```json
{
  "validation_type": "full_suite",
  "passed": true,
  "features": {
    "spectral_centroid": {
      "feature": "spectral_centroid",
      "mae": 0.123,
      "rmse": 0.456,
      "correlation": 0.998,
      "correlation_p_value": 0.0,
      "thresholds": {
        "mae": 50.0,
        "correlation": 0.95
      },
      "passed": {
        "mae": true,
        "correlation": true,
        "overall": true
      }
    }
  },
  "summary": {
    "total_features": 8,
    "passed": 7,
    "failed": 1,
    "pass_rate": 0.875
  }
}
```

#### Performance Analysis Schema

```json
{
  "analysis_type": "full_suite",
  "features": {
    "spectral_centroid": {
      "rsona": {
        "mean_ms": 12.34,
        "std_ms": 0.45
      },
      "librosa": {
        "mean_ms": 45.67,
        "std_ms": 1.23
      },
      "speedup": {
        "mean": 3.70,
        "median": 3.68
      },
      "absolute_improvement_ms": 33.33
    }
  },
  "summary": {
    "speedup": {
      "mean": 5.23,
      "geometric_mean": 4.87,
      "max": 237.32
    },
    "fastest_features": [...],
    "slowest_features": [...]
  }
}
```

### CSV Export

The `performance_comparison.csv` file contains:

```csv
Feature,rsona_mean_ms,rsona_std_ms,librosa_mean_ms,librosa_std_ms,speedup_mean,speedup_median,improvement_ms
spectral_centroid,12.34,0.45,45.67,1.23,3.70,3.68,33.33
spectral_flux,8.92,0.31,28.45,0.89,3.19,3.17,19.53
...
```

Import into spreadsheet tools or use for regression tracking.

### Charts

All charts are PNG images at 150 DPI, optimized for README embedding and presentations.

#### Runtime Comparison
Bar chart showing side-by-side runtime comparison (rsona vs librosa) for each feature.

#### Speedup Comparison
Bar chart showing speedup factors. Bars colored green (>1x = rsona faster) or red (<1x = librosa faster).

#### Feature Parity Matrix
Visual matrix showing feature availability in both libraries and benchmark readiness.

## Feature Parity Matrix

| Feature | rsona | librosa | Benchmark | Status |
|---------|-------|---------|-----------|--------|
| RMS / Energy | ❌ | ✔ | 🚧 | Coming Soon |
| Spectral Centroid | ✔ | ✔ | ✅ | Ready |
| Spectral Bandwidth | ✔ | ✔ | ✅ | Ready |
| Spectral Rolloff | ✔ | ✔ | ✅ | Ready |
| Spectral Flux | ✔ | ✔ | ✅ | Ready |
| Onset Detection | ✔ | ✔ | ✅ | Ready |
| Tempo Estimation | ✔ | ✔ | ✅ | Ready |
| Chroma STFT | ✔ | ✔ | ✅ | Ready |
| MFCC | ✔ | ✔ | ✅ | Ready |
| Structural Segmentation | ✔ | ❌ | 🚧 | Coming Soon |
| Streaming Analysis | ✔ | ❌ | 🚧 | Coming Soon |
| SIMD Batch Processing | ✔ | ❌ | 🚧 | Coming Soon |

**Legend:**
- ✅ **Ready**: Full 1:1 parity, benchmarked
- 🚧 **Coming Soon**: No equivalent in counterpart library
- ❌ **Not Available**: Feature not implemented

## Correctness Validation

Performance comparisons are only valid if outputs are numerically similar.

### Validation Metrics

#### Mean Absolute Error (MAE)
```
MAE = mean(|rsona_output - librosa_output|)
```
Measures average absolute difference. Lower is better.

#### Pearson Correlation
```
r = correlation(rsona_output, librosa_output)
```
Measures linear relationship strength. Range: [-1, 1]. Target: > 0.90.

#### Relative Error
```
relative_error = mean(|rsona - librosa| / |librosa|) × 100%
```
Percentage-based error metric.

### Thresholds

Each feature has correctness thresholds defined in `benchmark_spec.json`:

```json
{
  "feature": "spectral_centroid",
  "correctness": {
    "mae_threshold": 50.0,
    "correlation_threshold": 0.95
  }
}
```

Benchmarks **fail** if thresholds are exceeded.

## Performance Metrics

### Speedup Factor
```
speedup = librosa_runtime / rsona_runtime
```
- `> 1.0`: rsona is faster
- `< 1.0`: librosa is faster
- `= 1.0`: Equivalent performance

### Real-Time Factor
```
RTF = processing_time / audio_duration
```
- `< 1.0`: Faster than real-time
- `> 1.0`: Slower than real-time

Example: RTF of 0.1 means processing 10 seconds of audio takes 1 second.

### Geometric Mean Speedup

For aggregate performance across features:
```
geometric_mean = exp(mean(log(speedup_i)))
```

More robust than arithmetic mean when dealing with ratios.

## End-to-End Flows

Beyond isolated features, the suite benchmarks complete pipelines:

### Full Feature Extraction
```
Load Audio → STFT → Mel Spec → MFCC → Centroid → Flux → Onset → Tempo
```
Measures total pipeline overhead and identifies bottlenecks.

### Batch Processing
Process N files sequentially to test scaling behavior.

### Scaling Tests
Benchmark with varying audio durations (30s, 2min, 10min) to detect algorithmic complexity issues.

## Interpreting Results

### Expected Outcomes

Based on design differences:

1. **STFT/FFT**: rsona should be 3-5x faster (RustFFT optimizations)
2. **Mel Spectrogram**: rsona should be 8-12x faster (sparse matrix ops)
3. **Tempo**: rsona should be 100-250x faster (efficient autocorrelation)
4. **Overall Pipeline**: rsona should be 2-3x faster

### Red Flags

- **Speedup < 1.0**: rsona is slower (investigate)
- **High standard deviation**: Inconsistent performance (run more iterations)
- **Correlation < 0.85**: Outputs don't match (algorithm difference)
- **MAE beyond threshold**: Numerical accuracy issue

### Common Variations

- **First run slower**: CPU frequency scaling, cache cold start
- **Different absolute times**: Hardware, system load
- **Consistent speedup ratios**: What matters for comparison

## Troubleshooting

### "Python dependencies missing"

```bash
pip install librosa numpy scipy matplotlib
# or
pip install -r ../requirements.txt
```

### "cargo build failed"

Ensure Rust is installed:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### "Validation failed"

Check the validation JSON for details:
```bash
cat artifacts/validation_results.json | jq '.features | to_entries[] | select(.value.passed.overall == false)'
```

Common causes:
- Different default parameters (check spec)
- Numerical precision differences (expected for some features)
- Algorithm divergence (may be intentional)

### "Speedup varies widely between runs"

- Close other applications
- Increase warmup iterations: `--warmup 5`
- Increase total runs: `--runs 20`
- Disable CPU frequency scaling (Linux):
  ```bash
  sudo cpupower frequency-set --governor performance
  ```

### "Charts not generated"

Ensure matplotlib backend works:
```bash
python3 -c "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; plt.figure(); plt.savefig('test.png')"
```

## Adding New Features

To benchmark a new feature:

1. **Add to spec** (`benchmark_spec.json`):
   ```json
   {
     "feature": "new_feature",
     "librosa_available": true,
     "rsona_available": true,
     "status": "ready",
     "params": {...},
     "correctness": {
       "mae_threshold": 0.1,
       "correlation_threshold": 0.90
     }
   }
   ```

2. **Implement in librosa runner** (`librosa_runner.py`):
   ```python
   def benchmark_new_feature(self, y, sr, params):
       t_start = time.perf_counter()
       result = librosa.new_feature(y, sr, **params)
       t_end = time.perf_counter()
       return {...}
   ```

3. **Implement in rsona runner** (`rsona_runner.rs`):
   ```rust
   fn benchmark_new_feature(&self, audio_path: &str) -> Result<BenchmarkResult> {
       let t_start = Instant::now();
       let result = new_feature(...);
       let t_end = Instant::now();
       Ok(BenchmarkResult {...})
   }
   ```

4. **Run and validate**:
   ```bash
   ./run_benchmarks.sh --feature new_feature
   ```

## CI Integration

Add to GitHub Actions workflow:

```yaml
- name: Run Benchmark Suite
  run: |
    cd rsona/bench/librosa_comparison
    ./run_benchmarks.sh --audio ../../audio/test.wav --runs 5
    
- name: Check Performance Regression
  run: |
    python3 check_regression.py \
      --baseline baseline_results.json \
      --current artifacts/rsona_results.json \
      --threshold 0.10  # Fail if >10% slower
```

## Reproducibility

For reproducible benchmarks:

1. **Same hardware**: CPU model affects absolute times
2. **Same software versions**: Pin Python/Rust toolchain versions
3. **Idle system**: Minimize background processes
4. **Consistent audio**: Use same test file
5. **Multiple runs**: Average over 10+ iterations
6. **Document environment**: Record CPU, OS, versions

## Performance Optimization Workflow

When rsona is slower than expected:

1. **Profile**: Use `cargo flamegraph` or `perf`
2. **Isolate**: Run single-feature benchmark
3. **Compare algorithms**: Check if implementations differ
4. **Validate correctness first**: Don't sacrifice accuracy for speed
5. **Optimize**: Focus on hot paths identified by profiler
6. **Re-benchmark**: Verify improvements
7. **Document**: Update benchmarks with new results

## Related Documentation

- [rsona Main README](../../../README.md) - Library overview
- [Existing Benchmarks](../README.md) - Previous benchmark setup
- [librosa Documentation](https://librosa.org/doc/latest/index.html)

## License

This benchmark suite is part of rsona and licensed under Apache 2.0.

Test audio files may have separate licenses - see `bench/audio/README.md`.

## Contributing

When contributing benchmark improvements:

1. Maintain output schema compatibility
2. Add tests for new features in all three tools
3. Update `benchmark_spec.json` with thresholds
4. Generate reference results for CI baselines
5. Document any new metrics or analysis methods

## Acknowledgments

This benchmark architecture is inspired by:
- [Criterion.rs](https://github.com/bheisler/criterion.rs) - Statistical benchmarking
- [hyperfine](https://github.com/sharkdp/hyperfine) - Command-line benchmarking
- Research practices in audio DSP evaluation