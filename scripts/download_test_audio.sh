#!/usr/bin/env bash
#
# download_test_audio.sh - Generate test audio for rsona benchmarking
#
# Usage: ./scripts/download_test_audio.sh [output_file]
#
# This script generates consistent test audio using sox.
# Same audio generation used by CI/CD for reproducible benchmarks.

set -euo pipefail

OUTPUT="${1:-test_audio.wav}"

echo "Generating test audio for benchmarking..."
echo ""

# Check if sox is installed
if ! command -v sox &> /dev/null; then
    echo "Error: sox is required"
    echo ""
    echo "Install with:"
    echo "  macOS:  brew install sox"
    echo "  Ubuntu: sudo apt-get install sox"
    echo "  Windows: Download from https://sourceforge.net/projects/sox/"
    echo ""
    exit 1
fi

# Generate 15-second test audio with simulated beats
# - Two-tone harmony: 220 Hz (A3) + 330 Hz (E4)
# - Tremolo effect: 2 Hz creates ~120 BPM beat pattern
# - Perfect for testing tempo detection and beat tracking
# - Fade in/out to avoid clicks
sox -n -r 44100 -c 2 "$OUTPUT" \
    synth 15 sine 220 sine 330 \
    fade 0.1 15 0.1 tremolo 2 50 gain -3

if [ -f "$OUTPUT" ]; then
    echo ""
    echo "✓ Test audio generated successfully!"
    echo ""
    ls -lh "$OUTPUT"
    echo ""
    echo "Audio properties:"
    echo "  Duration: 15 seconds"
    echo "  Sample rate: 44100 Hz"
    echo "  Channels: Stereo"
    echo "  Format: WAV (PCM)"
    echo "  Simulated tempo: ~120 BPM"
    echo "  Method: Generated (reproducible)"
    echo ""
    echo "This is the same audio used by CI/CD workflows."
    echo ""
    echo "You can now run:"
    echo "  cargo run --release --bin rsona_bench -- $OUTPUT"
    echo "  cargo run --example mfcc -- $OUTPUT"
    echo "  cargo run --example tempo_beats -- $OUTPUT"
    echo ""
    echo "Optional: For testing with real music, see bench/TEST_AUDIO.md"
else
    echo "Error: Failed to create $OUTPUT"
    exit 1
fi
