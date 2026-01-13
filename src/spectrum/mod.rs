//! Frequency-domain analysis: STFT and spectral representations.

mod db;
mod mel;
mod stft;

pub use db::{DbConfig, amplitude_to_db, power_to_db};
pub use mel::{MelConfig, MelFilterBank, MelScale, MelSpectrogram, mel_spectrogram};
pub use stft::{Spectrogram, SpectrumError, StftConfig, stft};
