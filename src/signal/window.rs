//! Window functions for framing.

/// Supported window types.
///
/// These are commonly used for STFT framing and time-domain feature extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Window {
    /// Rectangular window (no tapering).
    Rectangular,
    /// Hann window.
    Hann,
    /// Hamming window.
    Hamming,
}

/// A precomputed window (length + coefficients).
///
/// Keeping this separate avoids recomputing the window coefficients per frame.
#[derive(Debug, Clone)]
pub struct WindowSpec {
    /// Window type.
    pub kind: Window,
    /// Window length.
    pub len: usize,
    /// Window coefficients.
    pub coeffs: Vec<f32>,
}

impl WindowSpec {
    /// Build a window of the given `kind` and `len`.
    pub fn new(kind: Window, len: usize) -> Self {
        assert!(len > 0, "window length must be > 0");
        let coeffs = match kind {
            Window::Rectangular => vec![1.0; len],
            Window::Hann => hann(len),
            Window::Hamming => hamming(len),
        };
        Self { kind, len, coeffs }
    }
}

fn hann(len: usize) -> Vec<f32> {
    // Hann: w[n] = 0.5 - 0.5*cos(2πn/(N-1))
    // N=1 edge: define as [1.0]
    if len == 1 {
        return vec![1.0];
    }
    let n_minus_1 = (len - 1) as f32;
    let two_pi = std::f32::consts::TAU; // 2π
    (0..len)
        .map(|n| {
            let angle = two_pi * (n as f32) / n_minus_1;
            0.5 - 0.5 * angle.cos()
        })
        .collect()
}

fn hamming(len: usize) -> Vec<f32> {
    // Hamming: w[n] = 0.54 - 0.46*cos(2πn/(N-1))
    // N=1 edge: define as [1.0]
    if len == 1 {
        return vec![1.0];
    }
    let n_minus_1 = (len - 1) as f32;
    let two_pi = std::f32::consts::TAU;
    (0..len)
        .map(|n| {
            let angle = two_pi * (n as f32) / n_minus_1;
            0.54 - 0.46 * angle.cos()
        })
        .collect()
}
