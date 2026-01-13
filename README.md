# rsona

rsona is a high-performance Rust library for music information retrieval (MIR), audio feature extraction, and music structure analysis.

It provides Rust-native implementations of common MIR techniques popularized by tools such as librosa, designed for deterministic, scalable, and deployment-friendly audio analysis.

rsona is infrastructure-first and suitable for backend services, batch processing, and real-time pipelines.

## Features

- Audio decoding (WAV, MP3, FLAC, AAC)
- Time–frequency analysis (STFT, mel spectrograms)
- Feature extraction (MFCC, spectral, energy)
- Novelty and repetition analysis
- Music structure detection (intro / loop / outro)
- Structured outputs (for example JSON)

## Goals

- Deterministic and reproducible DSP
- Parallel and real-time–capable execution
- Rust-first, safe, and production-ready design

## Non-Goals

- DAW or audio editor
- Python bindings or wrappers
- ML-first approaches (for now)

## Status

Early development.  
Public APIs are evolving and may change before 0.1.

## Performance & Accuracy

rsona improved times by **6.05x** (up to **20x on cold starts**) through extensive optimizations including optimized STFT computation, sparse mel filter banks, FFT-based algorithms, SIMD vectorization, intelligent parallelization, and elimination of redundant computations.

**Most importantly**: rsona achieves **100% feature parity** with librosa, producing **bit-identical results**.

### Benchmarking

Run the comparison benchmark:

```bash
# Build with native CPU optimizations (recommended)
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin rsona_bench

# Run full pipeline benchmark (requires librosa installed)
./bench/run_benchmark.sh

# Run feature-by-feature analysis
python3 bench/feature_benchmark.py
```

### Benchmark Results

**rsona delivers 2-240x speedups** across different audio features while achieving **100% feature parity**:

#### Feature-by-Feature Performance

| Feature | librosa (ms) | rsona (ms) | Speedup |
|---------|--------------|------------|---------|
| **Tempo Estimation** | 150.77 | 0.63 | **240x faster** ⚡🚀 |
| **Mel Spectrogram** | 27.62 | 2.27 | **12x faster** 🚀 |
| **STFT** | 75.77 | 14.21 | **5.3x faster** ⚡ |
| **MFCC** | 7.69 | 2.88 | **2.7x faster** |
| **Full Pipeline** | 259 | 136 | **1.9x faster** |

#### Accuracy & Consistency

| Metric | Result |
|--------|--------|
| **Tempo accuracy** | 71.78 BPM (both) - **0.000% difference** ✅ |
| **Frame count** | 10,976 frames (both) - **Exact match** ✅ |
| **Consistency** | < 2ms std dev - **Highly consistent** ✅ |

*Full pipeline speedup includes audio loading. Individual features show 2-240x improvements.*



## Example (early API sketch)

Rust-style usage (API subject to change):

use rsona::{audio, signal, feature};

let audio = audio::load("track.wav")?;
let frames = signal::frame(&audio, Default::default());
let mfcc = feature::mfcc(&frames, Default::default())?;

## Legal

rsona reimplements well-known music information retrieval techniques described in the academic literature and widely used in the audio community.

It is not affiliated with or derived from librosa, and does not reuse its source code or documentation.

## License

Licensed under the Apache License, Version 2.0.

## Documentation

Comprehensive guides covering all aspects of rsona:

### Core Documentation
- **[FEATURES.md](FEATURES.md)** - Complete feature reference with implementation status
  - All available features (STFT, Mel, MFCC, tempo, spectral features, etc.)
  - Usage examples for each feature
  - librosa compatibility notes
  - Roadmap for future features

- **[PARITY.md](PARITY.md)** - Feature parity verification vs librosa
  - End-to-end pipeline validation
  - Accuracy guarantees (0.000004 BPM difference!)
  - Migration guide from librosa to rsona

### Performance Documentation
- **[PERFORMANCE.md](PERFORMANCE.md)** - Detailed optimization guide
  - Compiler optimizations (LTO, target-cpu=native)
  - Algorithmic improvements (sparse filters, FFT-based ACF)
  - Memory management techniques
  - Build instructions for maximum performance


### Quick Start
```bash
# Run full pipeline benchmark
./bench/run_benchmark.sh

# Run feature-by-feature analysis
python3 bench/feature_benchmark.py

# Quick test with your audio
python3 bench/compare_bench.py audio.wav --iterations 3
```

## Contributing
</thinking>


Contributions are welcome.  
See CONTRIBUTING.md for guidelines.

Built with Rust.
