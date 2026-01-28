use thiserror::Error;

/// Errors that can occur while loading audio.
#[derive(Debug, Error)]
pub enum AudioError {
    /// Unsupported audio format.
    #[error("unsupported audio format")]
    UnsupportedFormat,

    /// Failed to decode audio.
    #[error("failed to decode audio")]
    DecodeError,

    /// IO error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}
