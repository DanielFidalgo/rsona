# Feature Parity Validation Guide

## Overview

Feature parity validation ensures that rsona produces **numerically accurate results** compared to the reference implementation (librosa). This goes beyond performance benchmarking by validating the **correctness** of computed audio features.

## Why Feature Parity Matters

When optimizing audio processing code, there are three critical dimensions:

1. **Performance** - How fast is it? (measured by `compare_bench.py`)
2. **Correctness** - Are the results accurate? (measured by `feature_parity.py`) ⭐
3. **Usability** - Is the API easy to use? (manual testing)

**Feature parity validation focuses on correctness.** A 100x speedup is meaningless if the results are wrong.

### Real-World Impact

❌ **Without parity validation:**
- Silent numerical errors can go undetected
- Algorithm changes might introduce subtle bugs
- Users lose confidence in results

✅ **With parity validation:**
- Mathematical correctness is continuously verified
- Regressions are caught immediately in CI
- Users can trust rsona's output for research and production

## Quick Start

### Generate Feature Parity Report

```bash
# Use test audio (recommended for first run)
python3 bench/feature_parity.py test_audio.wav

# Use your own audio
python3 bench/feature_parity.py path/to/your/music.mp3

# With custom tolerance (default is 1%)
python3 bench/feature_parity.py audio.wav --tolerance 0.005  # 0.5%

# Save report to file
python3 bench/feature_parity.py audio.wav --output PARITY_REPORT.md
```

### Read the Report

The script generates a markdown report showing:
- ✓/✗ Pass/fail status for each feature
- Mean and maximum errors (as percentages)
- Correlation coefficients
- Shape validation
- Statistical summaries

## What Gets Validated

Feature parity validation compares **actual computed values** between rsona and librosa:

### 1. STFT Magnitude Spectrum
- **What:** Short-Time Fourier Transform magnitude values
- **Shape:** `(frequency_bins, frames)` - typically `(1025, 1292)` for 30s audio
- **Tolerance:** Mean error < 1%, correlation > 0.999
- **Why it matters:** Foundation for all spectral features

### 2. Mel Spectrogram
- **What:** Perceptually-weighted frequency representation
- **Shape:** `(n_mels, frames)` - typically `(128, 1292)`
- **Tolerance:** Mean error < 1%
- **Why it matters:** Used for MFCCs and music analysis

### 3. MFCC (Mel-Frequency Cepstral Coefficients)
- **What:** Compact representation of spectral envelope
- **Shape:** `(n_mfcc, frames)` - typically `(13, 1292)`
- **Tolerance:** Mean error < 1%
- **Why it matters:** Critical for audio classification and similarity

### 4. RMS Energy
- **What:** Root Mean Square energy per frame
- **Shape:** `(frames,)` - typically `(1292,)`
- **Tolerance:** Mean error < 1%
- **Why it matters:** Indicates audio loudness over time

### 5. Onset Strength Envelope
- **What:** Likelihood of note onsets over time
- **Shape:** `(frames,)` - typically `(1292,)`
- **Tolerance:** Mean error < 1%
- **Why it matters:** Used for tempo and beat tracking

### 6. Tempo Estimation
- **What:** Estimated beats per minute (BPM)
- **Shape:** Single scalar value
- **Tolerance:** Difference < 5% (relaxed due to algorithm sensitivity)
- **Why it matters:** Core rhythm analysis metric

## How It Works

### Architecture

```
┌─────────────┐
│  Audio File │
└──────┬──────┘
       │
       ├────────────────────┬─────────────────────┐
       │                    │                     │
       ▼                    ▼                     ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   librosa   │      │    rsona    │      │  Validator  │
│   Pipeline  │      │  (--features)│      │   Engine    │
└──────┬──────┘      └──────┬──────┘      └─────────────┘
       │                    │                     ▲
       │ JSON output        │ JSON output         │
       │                    │                     │
       └────────────────────┴─────────────────────┘
                            │
                            ▼
                     ┌─────────────┐
                     │  Comparison │
                     │   & Report  │
                     └─────────────┘
```

### Step-by-Step Process

#### 1. librosa Pipeline (`run_librosa_pipeline`)
```python
y, sr = librosa.load(audio_file)
S = librosa.stft(y)
mel = librosa.feature.melspectrogram(S=S)
mfcc = librosa.feature.mfcc(S=mel)
rms = librosa.feature.rms(S=S)
onset = librosa.onset.onset_strength(S=mel)
tempo = librosa.beat.tempo(onset_envelope=onset)
```

#### 2. rsona Pipeline (`run_rsona_pipeline`)
```bash
cargo run --release --bin rsona_bench -- audio.wav --features
```

Outputs JSON:
```json
{
  "audio_info": {
    "sample_rate": 44100,
    "n_samples": 1323000
  },
  "features": {
    "stft_magnitude": [[...], [...], ...],
    "mel_spectrogram": [[...], [...], ...],
    "mfcc": [[...], [...], ...],
    "rms": [...],
    "onset_envelope": [...],
    "tempo_bpm": 114.84
  }
}
```

#### 3. Validation (`ParityValidator.validate_feature`)
```python
# For each feature pair:
diff = np.abs(rsona_array - librosa_array)
rel_error = diff / (np.abs(librosa_array) + 1e-10)
mean_error_pct = np.mean(rel_error) * 100
max_error_pct = np.max(rel_error) * 100
correlation = np.corrcoef(rsona_flat, librosa_flat)[0, 1]

pass_test = (mean_error_pct <= tolerance * 100) and (correlation > 0.99)
```

#### 4. Report Generation
- Markdown table with per-feature results
- Statistical summaries
- Pass/fail indicators
- Recommendations for failures

## Understanding Results

### Reading the Report

```markdown
## Feature Validation Results

| Feature | Status | Mean Error | Max Error | Correlation | Shape |
|---------|--------|------------|-----------|-------------|-------|
| STFT Magnitude | ✓ PASS | 0.23% | 1.45% | 0.9998 | (1025, 1292) |
| Mel Spectrogram | ✓ PASS | 0.31% | 2.03% | 0.9997 | (128, 1292) |
| MFCC | ✓ PASS | 0.45% | 3.21% | 0.9995 | (13, 1292) |
| RMS | ✓ PASS | 0.12% | 0.89% | 0.9999 | (1292,) |
| Onset Envelope | ✓ PASS | 0.67% | 4.11% | 0.9992 | (1292,) |
| Tempo | ✓ PASS | 4.44% | - | - | scalar |

## Summary
✅ Overall Status: PASS (6/6 features)
```

### Interpreting Metrics

**Mean Error (%):**
- < 1%: Excellent match ✓
- 1-5%: Good, minor differences
- 5-10%: Concerning, investigate
- > 10%: Likely bug ✗

**Max Error (%):**
- Peak difference in any single element
- Can be higher than mean (outliers)
- Check if it's localized or widespread

**Correlation:**
- > 0.999: Nearly identical ✓
- 0.99-0.999: Very similar
- 0.95-0.99: Similar structure
- < 0.95: Different patterns ✗

**Shape:**
- Must match exactly
- Different shapes = incompatible data

### Common Failure Patterns

#### 1. Shape Mismatch
```
Error: Shape mismatch - rsona: (1025, 1292), librosa: (1025, 1293)
```
**Cause:** Different frame calculation or hop size
**Fix:** Verify `FrameConfig` parameters match librosa defaults

#### 2. High Mean Error (> 5%)
```
Mean Error: 12.3% (FAIL)
```
**Cause:** Algorithmic difference or numerical instability
**Fix:** Review algorithm implementation, check for precision issues

#### 3. Low Correlation (< 0.95)
```
Correlation: 0.87 (FAIL)
```
**Cause:** Fundamentally different computation
**Fix:** Verify the algorithm matches librosa's approach

#### 4. Tempo Difference (> 10%)
```
rsona: 127.5 BPM, librosa: 115.2 BPM (diff: 10.7%)
```
**Cause:** Tempo estimation is inherently sensitive
**Fix:** Check onset detection parameters, test on more audio

## CI/CD Integration

### GitHub Actions Workflow

Feature parity runs automatically in CI:

```yaml
- name: Run feature parity validation
  run: |
    python3 bench/feature_parity.py test_audio.wav \
      --tolerance 0.01 \
      --output bench/FEATURE_PARITY.md
```

### Workflow Behavior

✅ **On Success:**
- Report uploaded as artifact
- Appears in PR comment
- Shown in workflow summary

⚠️ **On Failure:**
- Error report generated
- Workflow continues (non-blocking)
- Team notified for investigation

### Accessing Reports

1. **PR Comments:** Automatically posted to pull requests
2. **Workflow Artifacts:** Download from Actions tab (90-day retention)
3. **Workflow Summary:** View directly in GitHub UI

## Local Development

### Running Validation During Development

```bash
# Quick validation during feature development
python3 bench/feature_parity.py test_audio.wav

# Stricter tolerance for release validation
python3 bench/feature_parity.py test_audio.wav --tolerance 0.005

# Test with multiple files
for audio in audio/*.wav; do
  python3 bench/feature_parity.py "$audio" --output "parity_$(basename $audio).md"
done
```

### Debugging Failures

#### Step 1: Isolate the Feature
```bash
# Test just rsona output
cargo run --release --bin rsona_bench -- test_audio.wav --features > rsona_out.json

# Check JSON structure
jq '.features.stft_magnitude | length' rsona_out.json
```

#### Step 2: Compare Specific Values
```python
import json
import numpy as np

# Load outputs
with open('rsona_out.json') as f:
    rsona = json.load(f)

# Check specific regions
stft = np.array(rsona['features']['stft_magnitude'])
print(f"STFT stats: min={stft.min()}, max={stft.max()}, mean={stft.mean()}")
```

#### Step 3: Visualize Differences
```python
import matplotlib.pyplot as plt

diff = np.abs(rsona_stft - librosa_stft)
plt.imshow(np.log1p(diff), aspect='auto')
plt.colorbar()
plt.title('Absolute Difference (log scale)')
plt.savefig('stft_diff.png')
```

## Advanced Topics

### Custom Tolerance Levels

Different features may need different tolerances:

```python
# In feature_parity.py, you can customize per-feature:
validator.tolerance = 0.01  # Default 1%

# For tempo (more lenient)
validator.validate_feature("tempo", ..., tolerance_override=0.10)  # 10%

# For STFT (stricter)
validator.validate_feature("stft", ..., tolerance_override=0.005)  # 0.5%
```

### Adding New Features

To validate a new feature:

1. **Add extraction to rsona_bench:**
```rust
// In rsona_bench.rs
let new_feature = compute_new_feature(&data);

// In Features struct
#[derive(Serialize)]
struct Features {
    // ... existing features
    new_feature: Vec<f32>,
}
```

2. **Add extraction to librosa pipeline:**
```python
# In feature_parity.py
def run_librosa_pipeline(audio_path):
    # ... existing code
    new_feature = librosa.feature.new_feature(y=y, sr=sr)
    return {
        # ... existing features
        "new_feature": new_feature,
    }
```

3. **Add validation:**
```python
validator.validate_feature(
    "new_feature",
    rsona_data["new_feature"],
    librosa_data["new_feature"],
    "Description of new feature"
)
```

### Benchmarking + Parity Together

Complete validation workflow:

```bash
# 1. Performance benchmark
python3 bench/compare_bench.py audio.wav --output markdown

# 2. Feature parity validation
python3 bench/feature_parity.py audio.wav --output PARITY.md

# 3. Combined report
echo "## Performance" > FULL_REPORT.md
cat BENCHMARK_RESULTS.md >> FULL_REPORT.md
echo -e "\n## Accuracy" >> FULL_REPORT.md
cat PARITY.md >> FULL_REPORT.md
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'librosa'"
```bash
pip install librosa soundfile numpy
# or
pip install -r bench/requirements.txt
```

### "cargo: command not found"
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### "AttributeError: No librosa.feature attribute rhythm"
**Fixed!** This was due to an API change in librosa 0.10.x. The script now uses `librosa.beat.tempo()` instead.

### "JSON parsing error"
```bash
# Verify rsona_bench output is valid JSON
cargo run --release --bin rsona_bench -- test_audio.wav --features | jq .

# If it fails, check stderr
cargo run --release --bin rsona_bench -- test_audio.wav --features 2> error.log
cat error.log
```

### "Shape mismatch errors"
Ensure both implementations use the same parameters:
- Frame size: 2048
- Hop size: 512
- FFT size: 2048
- Mel bins: 128
- MFCC coeffs: 13

Check `rsona_bench.rs` and `feature_parity.py` use matching configs.

## Best Practices

### 1. Run Parity Before Optimization
- Validate correctness **first**
- Then optimize for speed
- Re-validate after optimization

### 2. Test with Diverse Audio
- Different genres (classical, electronic, speech)
- Different sample rates (22050, 44100, 48000)
- Different durations (short clips, long files)

### 3. Document Known Differences
Some differences are acceptable:
- Floating-point precision (< 0.1%)
- Different RNG seeds
- Edge effects at boundaries

### 4. Use Version Control
```bash
# Baseline before changes
python3 bench/feature_parity.py audio.wav --output parity_before.md

# After changes
python3 bench/feature_parity.py audio.wav --output parity_after.md

# Compare
diff parity_before.md parity_after.md
```

## Summary

Feature parity validation is **essential for maintaining trust** in rsona's numerical accuracy. It ensures that:

- ✅ Speed optimizations don't sacrifice correctness
- ✅ Algorithm changes maintain compatibility
- ✅ Users can rely on rsona for research and production
- ✅ Regressions are caught immediately

**Key Commands:**
```bash
# Generate test audio
scripts/download_test_audio.sh

# Run validation
python3 bench/feature_parity.py test_audio.wav

# View report
cat FEATURE_PARITY.md
```

**Next Steps:**
- Read the generated report
- Check all features pass
- Investigate any failures
- Integrate into your workflow

For performance benchmarking, see [README.md](README.md) and [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md).