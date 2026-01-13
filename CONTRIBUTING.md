# Contributing to rsona

Thank you for your interest in contributing to rsona.

rsona is a Rust-native library for music information retrieval (MIR) and audio structure analysis. We welcome contributions that improve correctness, performance, documentation, and usability.

## Core Principles

### Clean-room implementation

rsona is an independent Rust implementation of common music information retrieval (MIR) techniques.

Contributors may study existing libraries (such as librosa) and academic references to understand algorithms and expected behavior. However, implementations must be written independently and must not involve direct copying or mechanical translation of source code from other projects.

The goal is to reimplement ideas and algorithms, not reuse code or documentation text.

Comparing outputs against other libraries for validation or benchmarking is permitted and encouraged.

### Determinism

Given the same input, rsona must produce the same output.

Avoid:
- Randomized behavior
- Non-deterministic ordering
- Thread-order–dependent results

### Layered design

Higher-level features must be derived from lower-level primitives.

For example:
- Structure detection builds on features
- Features build on spectra
- Spectra build on framed signals

Avoid shortcuts or UI-specific logic.

## What to contribute

Good contributions include:
- DSP primitives (STFT, filters, transforms)
- Feature extraction (MFCC, spectral descriptors)
- Structure analysis (novelty, repetition, segmentation)
- Performance improvements
- Tests and benchmarks
- Documentation and examples

## What not to contribute (for now)

Please avoid:
- Machine learning models
- Python bindings
- DAW plugin formats
- Breaking API changes without discussion

These may be considered later as optional layers.

## Code style

- Follow idiomatic Rust
- Prefer explicit configuration over implicit defaults
- Use clear, documented error types
- Add tests for new functionality

## Licensing

By contributing to rsona, you agree that your contributions are licensed under the Apache License, Version 2.0.

## Questions

If you are unsure about an approach or the legality of an implementation, open an issue before submitting a pull request.

Thank you for helping build rsona.
