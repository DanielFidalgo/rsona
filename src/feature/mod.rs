//! Spectral and cepstral feature extraction.

/// Macro to generate time-series feature result types.
///
/// This macro eliminates boilerplate for feature types that store
/// a single f32 value per frame (centroid, bandwidth, rolloff, etc.).
///
/// # Example
///
/// ```ignore
/// time_series_feature!(SpectralCentroid);
/// ```
///
/// Expands to:
///
/// ```ignore
/// #[derive(Debug, Clone)]
/// pub struct SpectralCentroid {
///     n_frames: usize,
///     values: Vec<f32>,
/// }
///
/// impl SpectralCentroid {
///     pub fn new(values: Vec<f32>) -> Self {
///         let n_frames = values.len();
///         Self { n_frames, values }
///     }
///
///     pub fn n_frames(&self) -> usize {
///         self.n_frames
///     }
///
///     pub fn values(&self) -> &[f32] {
///         &self.values
///     }
/// }
/// ```
#[macro_export]
macro_rules! time_series_feature {
    ($name:ident) => {
        /// Feature values.
        #[derive(Debug, Clone)]
        pub struct $name {
            /// Frame count.
            n_frames: usize,
            /// Feature values.
            values: Vec<f32>,
        }

        impl $name {
            /// Create a new instance from values.
            #[inline]
            pub fn new(values: Vec<f32>) -> Self {
                let n_frames = values.len();
                Self { n_frames, values }
            }

            /// Frame count.
            #[inline]
            pub fn n_frames(&self) -> usize {
                self.n_frames
            }

            /// Feature values.
            #[inline]
            pub fn values(&self) -> &[f32] {
                &self.values
            }
        }
    };
}

mod bandwidth;
mod centroid;
pub mod chroma;
mod flux;
mod mfcc;
mod onset;
mod rms;
mod rolloff;

pub use bandwidth::{SpectralBandwidth, spectral_bandwidth};
pub use centroid::{SpectralCentroid, spectral_centroid};
pub use chroma::{
    A4_HZ, ChromaConfig, ChromaFilterBank, ChromaNorm, Chromagram, N_CHROMA,
    build_chroma_filterbank, chroma_stft,
};
pub use flux::{SpectralFlux, spectral_flux};
pub use mfcc::{DctNorm, MfccConfig, MfccResult, mfcc};
pub use onset::{OnsetConfig, OnsetEnvelope, onset_strength, onset_strength_from_mel};
pub use rms::{Rms, rms, rms_from_spectrogram};
pub use rolloff::{SpectralRolloff, spectral_rolloff};
