//! Export chroma filters for comparison with reference implementation.
//!
//! This utility exports the chroma filter bank weights to JSON format
//! so they can be compared with librosa's reference implementation.
//!
//! Usage:
//!   cargo run --example export_chroma_filters > filters.json

use rsona::feature::chroma::build_chroma_filterbank;
use serde_json::json;

fn main() {
    // Parameters matching the comparison script
    let sr = 22050;
    let n_fft = 2048;
    let n_bins = (n_fft / 2) + 1;
    let n_chroma = 12;
    let tuning = 0.0;
    let fmin = 32.7; // C1
    let fmax = sr as f32 / 2.0; // Nyquist
    let ctroct = 5.0;
    let octwidth = 2.0;

    eprintln!("Building rsona chroma filter bank...");
    eprintln!("  Sample rate: {} Hz", sr);
    eprintln!("  FFT size: {}", n_fft);
    eprintln!("  Number of bins: {}", n_bins);
    eprintln!("  Number of chroma: {}", n_chroma);
    eprintln!("  Frequency range: {:.1} Hz - {:.1} Hz", fmin, fmax);

    // Build the filter bank
    let filter_bank = build_chroma_filterbank(
        sr, n_fft, n_bins, n_chroma, tuning, fmin, fmax, ctroct, octwidth,
    );

    // Convert to dense representation
    let dense_filters = filter_bank.to_dense();

    eprintln!("\nFilter statistics:");
    let note_names = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    for (i, filter) in dense_filters.iter().enumerate() {
        let peak: f32 = filter.iter().copied().fold(0.0f32, |a, b| a.max(b));
        let nonzero_count = filter.iter().filter(|&&w| w > 0.001).count();
        eprintln!(
            "  Chroma {:2} ({}): peak = {:.6}, nonzero bins = {}",
            i, note_names[i], peak, nonzero_count
        );
    }

    // Calculate overall statistics
    let all_peaks: Vec<f32> = dense_filters
        .iter()
        .map(|filter| filter.iter().copied().fold(0.0f32, |a, b| a.max(b)))
        .collect();

    let mean_peak: f32 = all_peaks.iter().sum::<f32>() / all_peaks.len() as f32;
    let min_peak = all_peaks
        .iter()
        .copied()
        .fold(f32::INFINITY, |a, b| a.min(b));
    let max_peak = all_peaks.iter().copied().fold(0.0f32, |a, b| a.max(b));

    eprintln!("\nOverall peak statistics:");
    eprintln!("  Mean: {:.6}", mean_peak);
    eprintln!("  Min:  {:.6}", min_peak);
    eprintln!("  Max:  {:.6}", max_peak);

    // Export to JSON
    let output = json!({
        "parameters": {
            "sample_rate": sr,
            "n_fft": n_fft,
            "n_bins": n_bins,
            "n_chroma": n_chroma,
            "tuning": tuning,
            "fmin": fmin,
            "fmax": fmax,
            "ctroct": ctroct,
            "octwidth": octwidth,
        },
        "filters": dense_filters,
        "peaks": all_peaks,
        "statistics": {
            "mean_peak": mean_peak,
            "min_peak": min_peak,
            "max_peak": max_peak,
        },
    });

    println!("{}", serde_json::to_string_pretty(&output).unwrap());

    eprintln!("\n✓ Filters exported to stdout");
    eprintln!("  Save with: cargo run --example export_chroma_filters > rsona_filters.json");
}
