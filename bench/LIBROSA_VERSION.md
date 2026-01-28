# librosa Version Requirement

## Required Version

**librosa >= 0.10.0** is required for the rsona benchmark suite.

## Why This Version?

### API Changes in librosa 0.10.0

librosa 0.10.0 introduced significant API reorganization:

**Old API (< 0.10.0):**
```python
tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
```

**New API (>= 0.10.0):**
```python
tempo = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr)
```

The tempo estimation function was moved from `librosa.beat` to `librosa.feature.rhythm` as part of a broader reorganization to improve API consistency.

### Why We Don't Support Old Versions

1. **Simplicity** - Single API path is easier to maintain
2. **Future-proofing** - Old API will be removed in librosa 1.0
3. **Consistency** - All users use the same API
4. **CI/CD** - One version to test against

## Installation

### Fresh Install

```bash
# Install all benchmark dependencies
pip install -r bench/requirements.txt
```

This installs:
- `librosa>=0.10.0`
- `numpy>=1.24.0`
- `soundfile>=0.12.0`

### Upgrade Existing Installation

```bash
# Upgrade librosa to latest version
pip install --upgrade 'librosa>=0.10.0'
```

### Verify Installation

```bash
python3 -c "import librosa; print(f'librosa version: {librosa.__version__}')"
```

Expected output:
```
librosa version: 0.10.1  (or higher)
```

## Troubleshooting

### Error: "No librosa.feature attribute rhythm"

**Symptom:**
```
AttributeError: module 'librosa.feature' has no attribute 'rhythm'
```

**Cause:** You have librosa < 0.10.0 installed.

**Solution:**
```bash
pip install --upgrade 'librosa>=0.10.0'
```

### Error: "librosa version X.X.X is too old"

**Symptom:**
```
ERROR: librosa version 0.9.2 is too old.
This script requires librosa >= 0.10.0
```

**Cause:** The script detected an old version at startup.

**Solution:**
```bash
pip install --upgrade 'librosa>=0.10.0'
```

### Check Current Version

```bash
# Method 1: Python
python3 -c "import librosa; print(librosa.__version__)"

# Method 2: pip
pip show librosa | grep Version
```

### Clean Reinstall

If you have persistent issues:

```bash
# Uninstall old version
pip uninstall librosa

# Install fresh
pip install 'librosa>=0.10.0'
```

## CI/CD Configuration

The GitHub Actions workflow automatically installs the correct version:

```yaml
- name: Install Python dependencies
  run: |
    python -m pip install --upgrade pip
    pip install -r bench/requirements.txt
```

This ensures all CI runs use librosa >= 0.10.0.

## Migration Guide

If you have code using the old API:

### Before (librosa < 0.10.0)
```python
import librosa

# Tempo estimation
tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length)[0]
```

### After (librosa >= 0.10.0)
```python
import librosa

# Tempo estimation
tempo = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length)[0]
```

**Note:** The old API still works in 0.10.x but shows deprecation warnings. It will be removed in librosa 1.0.

## Version Check in Scripts

The `feature_parity.py` script includes an automatic version check:

```python
import librosa
from packaging import version

librosa_version = version.parse(librosa.__version__)
required_version = version.parse("0.10.0")

if librosa_version < required_version:
    print(f"ERROR: librosa version {librosa.__version__} is too old.")
    print(f"This script requires librosa >= 0.10.0")
    sys.exit(1)
```

This prevents cryptic errors by failing fast with a clear message.

## Why 0.10.0 Specifically?

- **Released:** 2023-11-16
- **Stable:** Well-tested in production
- **Modern:** Includes performance improvements
- **Supported:** Active maintenance and updates
- **Breaking changes:** Minimal from 0.10.x to future versions

## Dependencies

librosa 0.10.0+ requires:

- Python >= 3.8
- numpy >= 1.20.0
- scipy >= 1.2.0
- scikit-learn >= 0.20.0
- joblib >= 0.14
- decorator >= 4.3.0
- soundfile >= 0.12.0 (for audio loading)

All automatically installed via `pip install librosa`.

## Summary

✅ **Always use librosa >= 0.10.0** for rsona benchmarks  
✅ **Install via:** `pip install -r bench/requirements.txt`  
✅ **Verify with:** `python3 -c "import librosa; print(librosa.__version__)"`  
✅ **CI enforces:** Version requirement automatically  

This ensures consistent behavior across all environments and avoids API compatibility issues.
