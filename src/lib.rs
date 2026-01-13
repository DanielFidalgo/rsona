//! rsona — Music Information Retrieval (MIR) in Rust
//!
//! rsona is a Rust-native library for audio feature extraction and music
//! structure analysis. It provides deterministic, scalable primitives for
//! building MIR pipelines, inspired by common workflows popularized by tools
//! such as librosa.
//!
//! ## Design
//!
//! rsona is layered explicitly:
//!
//! - `audio`   — decoding and time-domain buffers
//! - `signal`  — framing and windowing
//! - `spectrum`— STFT and spectral representations
//! - `feature` — MFCC and other descriptors
//!
//! Higher-level analysis is always built on top of lower-level primitives.
//!
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]
#![warn(unused_imports)]
/// Load audio file.
pub mod audio;
pub mod feature;
pub mod signal;
pub mod similarity;
pub mod spectrum;
pub mod structure;
pub mod temporal;
