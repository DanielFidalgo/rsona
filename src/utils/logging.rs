//! Logging utilities for rsona.
//!
//! This module provides simple logging initialization for rsona binaries and examples.
//!
//! # Usage
//!
//! In your binary or example:
//!
//! ```no_run
//! use rsona::utils::logging;
//!
//! fn main() {
//!     // Initialize with default settings (WARN level)
//!     logging::init();
//!
//!     // Or initialize with a specific level
//!     logging::init_with_level(logging::Level::Debug);
//! }
//! ```
//!
//! # Environment Variables
//!
//! You can control logging via the `RUST_LOG` environment variable:
//!
//! ```bash
//! RUST_LOG=debug cargo run --example mfcc -- audio.wav
//! RUST_LOG=rsona=trace cargo run --example mfcc -- audio.wav
//! ```
//!
//! # Log Levels
//!
//! - `TRACE`: Very verbose, shows all internal operations
//! - `DEBUG`: Detailed information for debugging
//! - `INFO`: General information about operations
//! - `WARN`: Warnings (default)
//! - `ERROR`: Errors only

use tracing_subscriber::{EnvFilter, fmt};

/// Initialize logging with default settings.
///
/// This sets up logging with WARN level by default, unless overridden
/// by the `RUST_LOG` environment variable.
///
/// # Example
///
/// ```no_run
/// use rsona::utils::logging;
///
/// fn main() {
///     logging::init();
///     tracing::info!("This won't show with default settings");
///     tracing::warn!("This will show");
/// }
/// ```
pub fn init() {
    init_with_level(tracing::Level::WARN);
}

/// Initialize logging with a specific level.
///
/// The `RUST_LOG` environment variable will override this setting if present.
///
/// # Example
///
/// ```no_run
/// use rsona::utils::logging;
/// use tracing::Level;
///
/// fn main() {
///     logging::init_with_level(Level::DEBUG);
///     tracing::debug!("This will show");
/// }
/// ```
pub fn init_with_level(level: tracing::Level) {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(level.to_string().to_lowercase()));

    fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_thread_ids(false)
        .with_thread_names(false)
        .with_file(false)
        .with_line_number(false)
        .compact()
        .init();
}

/// Initialize logging for verbose/detailed output.
///
/// This is useful for examples and debugging, showing INFO level messages.
///
/// # Example
///
/// ```no_run
/// use rsona::utils::logging;
///
/// fn main() {
///     logging::init_verbose();
///     tracing::info!("This will show");
///     tracing::debug!("This won't show unless RUST_LOG=debug");
/// }
/// ```
pub fn init_verbose() {
    init_with_level(tracing::Level::INFO);
}

/// Initialize logging for debugging.
///
/// Shows DEBUG level messages and above.
///
/// # Example
///
/// ```no_run
/// use rsona::utils::logging;
///
/// fn main() {
///     logging::init_debug();
///     tracing::debug!("This will show");
///     tracing::trace!("This won't show unless RUST_LOG=trace");
/// }
/// ```
pub fn init_debug() {
    init_with_level(tracing::Level::DEBUG);
}

/// Re-export tracing Level for convenience
pub use tracing::Level;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_does_not_panic() {
        // Just ensure initialization doesn't panic
        // We can't test actual logging output easily in unit tests
        init();
    }
}
