# Quick Start: rsona vs librosa Benchmarks

Get up and running with the benchmark suite in 5 minutes.

## Prerequisites

- Python 3.8+
- Rust/Cargo (install from https://rustup.rs)
- An audio file (WAV, MP3, or FLAC)

## Installation

```bash
# 1. Install Python dependencies
pip install librosa numpy scipy matplotlib

# 2. Navigate to benchmark directory
cd rsona/bench/librosa_comparison

# 3. Make script executable
chmod +x run_benchmarks.sh
```

## Run Benchmarks

### Option 1: Use Default Audio

If you have audio files in `bench/audio/`, the script will auto-detect:

```bash
./run_benchmarks.sh
```

### Option 2: Specify Audio File

```bash
./run_benchmarks.sh --audio /path/to/your/audio.wav
```

### Option 3: Quick Test (Fewer Runs)

```bash
./run_benchmarks.sh --audio test.wav --runs 5 --warmup 1
```

### Option 4: Maximum Performance (Native CPU)

```bash
./run_benchmarks.sh --audio test.wav --native-cpu
```

## View Results

After completion, check the `artifacts/` directory:

```bash
# View performance summary (requires jq)
cat artifacts/performance_analysis.json | jq '.summary'

# View as table
cat artifacts/performance_comparison.csv

# Open charts (macOS)
open artifacts/runtime_comparison.png
open artifacts/speedup_comparison.png

# Open charts (Linux)
xdg-open artifacts/runtime_comparison.png

# Open charts (Windows)
start artifacts/runtime_comparison.png
```

## Common Commands

### Benchmark Single Feature

```bash
./run_benchmarks.sh --feature spectral_centroid --audio test.wav
```

### Clean Previous Results

```bash
./run_benchmarks.sh --audio test.wav --clean
```

### Skip Rust Rebuild (Faster Iteration)

```bash
./run_benchmarks.sh --audio test.wav --skip-build
```

### Verbose Output (Debugging)

```bash
./run_benchmarks.sh --audio test.wav --verbose
```

## Understanding Output

### Terminal Summary

The script displays:
- ✅ Which steps completed successfully
- 📊 Key performance metrics (speedup, improvement %)
- 🏆 Fastest and slowest features
- ✓ Correctness validation status

### Generated Files

| File | Description |
|------|-------------|
| `librosa_results.json` | Raw librosa benchmark data |
| `rsona_results.json` | Raw rsona benchmark data |
| `validation_results.json` | Correctness check results |
| `performance_analysis.json` | Full performance comparison |
| `performance_comparison.csv` | Spreadsheet-friendly export |
| `runtime_comparison.png` | Bar chart: side-by-side runtimes |
| `speedup_comparison.png` | Bar chart: speedup factors |
| `feature_parity_matrix.png` | Visual feature availability |

## Quick Checks

### Did It Work?

```bash
# Check if artifacts were created
ls -lh artifacts/

# Verify JSON is valid
jq . artifacts/rsona_results.json > /dev/null && echo "✓ Valid JSON"

# Count successful features
jq '.results | length' artifacts/performance_analysis.json
```

### View Speedup Summary

```bash
jq -r '.features | to_entries[] | "\(.key): \(.value.speedup.mean)x faster"' \
  artifacts/performance_analysis.json
```

### Check Validation Status

```bash
jq '.passed' artifacts/validation_results.json
```

## Troubleshooting

### "Python dependencies missing"

```bash
pip install librosa numpy scipy matplotlib
```

### "cargo not found"

Install Rust:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### "No audio file found"

Place an audio file in `bench/audio/` or specify with `--audio`:
```bash
./run_benchmarks.sh --audio ~/Music/test.wav
```

### "Permission denied"

```bash
chmod +x run_benchmarks.sh
```

### Compilation Errors

Clean and rebuild:
```bash
cd ../../..  # Back to rsona root
cargo clean
cargo build --release --bin rsona_runner
cd bench/librosa_comparison
./run_benchmarks.sh --audio test.wav
```

## Common Workflows

### Daily Development

```bash
# Quick check after code changes
./run_benchmarks.sh --runs 3 --skip-validation
```

### Pre-Commit Validation

```bash
# Full benchmark with correctness checks
./run_benchmarks.sh --runs 10 --native-cpu
```

### Performance Investigation

```bash
# Benchmark specific feature with more runs
./run_benchmarks.sh --feature spectral_centroid --runs 20 --verbose
```

### Generate Marketing Charts

```bash
# Clean run with all visualization
./run_benchmarks.sh --clean --runs 15 --native-cpu
```

## Next Steps

- **Read Full Docs**: See [README.md](README.md) for comprehensive guide
- **Add Features**: Learn how to benchmark new features
- **CI Integration**: Set up automated benchmarking
- **Custom Analysis**: Extend `analyze_performance.py`

## Help

```bash
# Show all options
./run_benchmarks.sh --help

# Check versions
python3 --version
cargo --version
pip list | grep librosa
```

## Expected Results

For a 30-second audio file on modern hardware:

- **rsona typically 2-5x faster** than librosa
- **Tempo estimation: 100-250x faster**
- **Total runtime: ~200-400ms** for full suite
- **Correctness: >90% features pass** validation

## Tips

1. **Close other apps** for consistent benchmarks
2. **Use same audio file** for fair comparison
3. **Run multiple times** to account for variance
4. **Check validation first** before trusting speedup numbers
5. **Native CPU build** gives best performance

## Quick Example Session

```bash
# Complete workflow
cd rsona/bench/librosa_comparison
./run_benchmarks.sh --audio ~/test.wav --native-cpu --runs 10

# Check results
cat artifacts/performance_analysis.json | jq '.summary'
open artifacts/speedup_comparison.png

# Share results
cp artifacts/*.png ~/Desktop/
cp artifacts/performance_comparison.csv ~/Desktop/
```

That's it! You're ready to benchmark rsona vs librosa.