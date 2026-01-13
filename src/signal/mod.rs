//! Time-domain signal processing utilities: framing, windowing, and envelopes.
//!
//! This module operates purely in the time domain. It does not perform FFTs or
//! frequency-domain transforms. Those belong in `spectrum`.

mod frame;
mod window;

pub use frame::{
    frame, ChannelMode, FrameConfig, Frames, Padding, SignalError,
};
pub use window::{Window, WindowSpec};
