# rsona Benchmarks

This directory contains benchmarking tools to compare rsona's performance against librosa.

## Quick Start

```bash
# Run full pipeline comparison
./run_benchmark.sh

# Run feature-by-feature benchmark
python3 feature_benchmark.py

# Run with custom audio file
./run_benchmark.sh path/to/audio.mp3
```

## Benchmark Tools

### 1. `compare_bench.py` - Full Pipeline Comparison

Compares the complete audio analysis pipeline (STFT → Mel → MFCC → Onset → Tempo).

```bash
python3 compare_bench.py [audio_file] --iterations 5 --warmup 2
```

**Options:**
- `-n, --iterations N` - Number of benchmark iterations (default: 5)
- `-w, --warmup N` - Number of warmup iterations (default: 1)
- `-o, --output FORMAT` - Output format: markdown, json, csv (default: markdown)
- `-f, --file PATH` - Save output to file

### 2. `feature_benchmark.py` - Feature-by-Feature Analysis

Measures individual feature extraction performance to identify optimization opportunities.

```bash
python3 feature_benchmark.py [audio_file] --iterations 5
```

**Features Measured:**
- STFT (Short-Time Fourier Transform)
- Mel Spectrogram
- MFCC (Mel-Frequency Cepstral Coefficients)
- RMS (Root Mean Square Energy)
- Onset Strength
- Tempo Estimation
- Overall Pipeline

### 3. `run_benchmark.sh` - Automated Runner

Convenience script that checks dependencies, builds rsona, and runs comparisons.

```bash
# Default run
./run_benchmark.sh

# Custom iterations and output
./run_benchmark.sh -n 10 -o json -f results.json

# Skip rebuild for faster iteration
./run_benchmark.sh --skip-build
```

## Performance Results

### Feature-by-Feature Performance

Based on benchmarks with a 30-second audio file:

| Feature | librosa (ms) | rsona (ms) | Speedup |
|---------|--------------|------------|---------|
| **STFT** | 74.68 | 14.29 | **5.22x** ⚡ |
| **Mel Spectrogram** | 26.94 | 2.32 | **11.64x** 🚀 |
| **MFCC** | 7.70 | 2.65 | **2.91x** |
| **RMS** | 12.10 | 15.00 | 0.81x |
| **Onset Strength** | 1.42 | 2.17 | 0.65x |
| **Tempo Estimation** | 149.20 | 0.63 | **237.32x** ⚡🚀 |
| **Total Pipeline** | 290.04 | 134.96 | **2.15x** |

### Key Insights

#### 🚀 Massive Speedups

1. **Tempo Estimation: 237x faster**
   - Rust's efficient autocorrelation implementation
   - Low-level optimizations in peak detection
   - SIMD-friendly algorithms

2. **Mel Spectrogram: 11.64x faster**
   - Sparse mel filter banks (97% zeros)
   - Optimized matrix multiplication
   - Cache-friendly memory access patterns

3. **STFT: 5.22x faster**
   - RustFFT optimizations
   - Reduced allocations
   - Vectorized operations

#### ⚠️ Areas for Improvement

- **Onset Strength**: Currently 1.5x slower than librosa
  - Opportunity for optimization
  - May involve different algorithmic approach
  - Still fast in absolute terms (2.17ms)

### Overall Pipeline Performance

- **Full Pipeline Speedup**: 2.15x faster
- **Consistency**: Very low standard deviation (< 2ms)
- **Accuracy**: Perfect tempo estimation (< 0.001% difference)
- **Frame Parity**: Exact frame count match with librosa

## Understanding the Results

### Why Different Speedup Numbers?

You might see different speedup figures referenced:

1. **Full Pipeline (2.15x)**: Includes audio loading, which is similar speed in both implementations
2. **Core Features (5-12x)**: Individual feature extraction speedups (STFT, Mel, MFCC)
3. **Tempo (237x)**: Specialized algorithm with extreme optimization
4. **End-to-End (~1.9x)**: When audio loading dominates (large files)

### Warmup Iterations

The benchmarks include warmup iterations because:
- First run includes disk I/O caching
- JIT compilation in Python
- Memory allocator warmup
- CPU frequency scaling

Without warmup, the first iteration can be 2-4x slower, skewing results.

### Statistical Measures

- **Mean**: Average performance across all iterations
- **Median**: Middle value (robust to outliers)
- **Std Dev**: Consistency of measurements (lower is better)
- **Speedup**: Ratio of librosa time / rsona time

## Requirements

### System Requirements
- Rust 1.70+ with cargo
- Python 3.8+
- Audio file in MP3, WAV, or FLAC format

### Python Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- librosa >= 0.10.0
- numpy >= 1.20.0
- serde_json (for parsing rsona output)

### Building rsona

```bash
# Standard build
cargo build --release

# With native CPU optimizations (recommended)
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

## CI/CD Integration

The benchmark suite includes GitHub Actions workflow for regression detection:

- Runs on every PR and push to main
- Compares current vs baseline performance
- Fails if performance regresses > 10%
- Generates performance reports

See `.github/workflows/benchmark.yml` for configuration.

## Benchmark Data

### Test Audio

The default test audio is located in `bench/audio/`:
- **File**: Alex-Productions - Future Bass Technology _ Shades.mp3
- **Duration**: ~30 seconds
- **Sample Rate**: 44.1 kHz
- **License**: CC0 (Public Domain)

You can add your own audio files for testing:

```bash
# Test with your audio
python3 compare_bench.py path/to/your/audio.wav
```

### Result Files

Benchmark results are saved as:
- `*_REPORT.md` - Generated markdown reports (gitignored)
- `*.json` - JSON output files (gitignored)
- `*.csv` - CSV export files (gitignored)
- CI generates `curr.json` and `base.json` for regression testing

## Directory Structure

```
bench/
├── README.md                 # This file
├── QUICKSTART.md            # 30-second getting started guide
├── RESULTS_SUMMARY.md       # Latest benchmark results
├── .gitignore               # Ignore generated reports
│
├── compare_bench.py         # Full pipeline comparison
├── feature_benchmark.py     # Feature-by-feature analysis
├── rsona_bench.rs           # Rust benchmark binary
├── run_benchmark.sh         # Automated runner script
├── requirements.txt         # Python dependencies
│
└── audio/                   # Test audio files
    ├── *.mp3
    └── *.wav
```

## Troubleshooting

### "librosa not found"

```bash
pip install librosa numpy
# or
pip install -r requirements.txt
```

### "cargo build failed"

Ensure you have Rust installed:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Inconsistent Results

- Ensure no other heavy processes are running
- Use `--warmup 2` or higher
- Run more iterations with `-n 10`
- Close other applications during benchmarking

### Audio File Not Found

Place audio files in `bench/audio/` directory or specify full path:
```bash
./run_benchmark.sh /full/path/to/audio.mp3
```

## Advanced Usage

### Generate CSV for Analysis

```bash
python3 compare_bench.py --iterations 10 --output csv --file results.csv
```

Import into spreadsheet tools for detailed analysis.

### JSON Output for Automation

```bash
python3 feature_benchmark.py --output json --file results.json
```

Parse with tools like `jq`:
```bash
cat results.json | jq '.stft.speedup'
```

### Continuous Monitoring

Set up periodic benchmarks:
```bash
# Run daily benchmark and save results
./run_benchmark.sh -o json -f "benchmark_$(date +%Y%m%d).json"
```

## Contributing

When adding new benchmarks:

1. Follow the existing pattern in `compare_bench.py`
2. Include warmup iterations
3. Use statistical measures (mean, median, stdev)
4. Document expected performance characteristics
5. Add CI integration if appropriate

## Related Documentation

- [Performance Guide](../PERFORMANCE.md) - Optimization techniques
- [Feature Parity](../PARITY.md) - Feature comparison with librosa
- [FAQ](../FAQ.md) - Common questions about performance

## License

Benchmark scripts are MIT licensed. Test audio files have their own licenses (see audio/README.md).