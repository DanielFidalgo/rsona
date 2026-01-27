#!/usr/bin/env bash
# Test script for feature parity validation
# Usage: ./test_parity.sh [audio_file]

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
AUDIO_FILE="${1:-test_audio.wav}"

echo "======================================"
echo "Feature Parity Validation Test"
echo "======================================"
echo ""

# Check Python dependencies
echo "Checking Python dependencies..."
if ! python3 -c "import librosa, numpy, soundfile" 2>/dev/null; then
    echo "❌ Missing Python dependencies"
    echo ""
    echo "Install with:"
    echo "  pip install librosa numpy soundfile"
    echo ""
    exit 1
fi
echo "✓ Python dependencies OK"
echo ""

# Check Rust/Cargo
echo "Checking Rust toolchain..."
if ! command -v cargo &> /dev/null; then
    echo "❌ Cargo not found"
    echo ""
    echo "Install Rust from: https://rustup.rs/"
    echo ""
    exit 1
fi
echo "✓ Rust toolchain OK"
echo ""

# Check/generate test audio
echo "Checking test audio..."
if [ ! -f "$PROJECT_ROOT/$AUDIO_FILE" ] && [ "$AUDIO_FILE" = "test_audio.wav" ]; then
    echo "Test audio not found, generating..."
    if [ -f "$PROJECT_ROOT/scripts/download_test_audio.sh" ]; then
        cd "$PROJECT_ROOT"
        bash scripts/download_test_audio.sh
        cd "$SCRIPT_DIR"
    else
        echo "⚠️  No test audio found. Using sox to generate..."
        sox -n -r 44100 -c 1 "$PROJECT_ROOT/test_audio.wav" synth 30 sine 440 tremolo 2
    fi
fi

if [ ! -f "$PROJECT_ROOT/$AUDIO_FILE" ]; then
    echo "❌ Audio file not found: $AUDIO_FILE"
    exit 1
fi
echo "✓ Audio file found: $AUDIO_FILE"
echo ""

# Build rsona_bench
echo "Building rsona_bench..."
cd "$PROJECT_ROOT"
if cargo build --release --bin rsona_bench --quiet; then
    echo "✓ Build successful"
else
    echo "❌ Build failed"
    exit 1
fi
cd "$SCRIPT_DIR"
echo ""

# Test rsona_bench --features
echo "Testing rsona_bench --features output..."
RSONA_OUTPUT=$(cargo run --release --bin rsona_bench --quiet -- "$PROJECT_ROOT/$AUDIO_FILE" --features 2>&1)
if [ $? -ne 0 ]; then
    echo "❌ rsona_bench --features failed"
    echo "$RSONA_OUTPUT"
    exit 1
fi

# Validate JSON output
if echo "$RSONA_OUTPUT" | jq . > /dev/null 2>&1; then
    echo "✓ Valid JSON output"
else
    echo "❌ Invalid JSON output from rsona_bench"
    echo "Output:"
    echo "$RSONA_OUTPUT"
    exit 1
fi
echo ""

# Run feature parity validation
echo "======================================"
echo "Running Feature Parity Validation"
echo "======================================"
echo ""

cd "$PROJECT_ROOT"
if python3 bench/feature_parity.py "$AUDIO_FILE" --tolerance 0.01 --output bench/FEATURE_PARITY_TEST.md; then
    echo ""
    echo "======================================"
    echo "✅ Feature Parity Validation PASSED"
    echo "======================================"
    echo ""

    # Show summary
    if [ -f "bench/FEATURE_PARITY_TEST.md" ]; then
        echo "Report saved to: bench/FEATURE_PARITY_TEST.md"
        echo ""
        echo "Summary:"
        head -40 bench/FEATURE_PARITY_TEST.md
        echo ""
        echo "View full report: cat bench/FEATURE_PARITY_TEST.md"
    fi

    exit 0
else
    echo ""
    echo "======================================"
    echo "❌ Feature Parity Validation FAILED"
    echo "======================================"
    echo ""
    echo "Check the errors above for details."
    echo ""
    echo "Common issues:"
    echo "  - librosa version mismatch (need >= 0.10.0)"
    echo "  - Audio file format not supported"
    echo "  - Numerical tolerance too strict"
    echo ""
    echo "Try running manually for more details:"
    echo "  python3 bench/feature_parity.py $AUDIO_FILE --tolerance 0.01"
    echo ""
    exit 1
fi
