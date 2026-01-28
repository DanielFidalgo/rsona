#!/usr/bin/env bash
#
# download_test_audio.sh - Download test audio for rsona benchmarking
#
# Downloads Creative Commons licensed music from Freesound for testing.
# Falls back to local files or synthetic generation if download fails.
#
# Usage: ./scripts/download_test_audio.sh [output_file]
#
# Environment variables:
#   FREESOUND_API_KEY - Your Freesound API key (get from https://freesound.org/apiv2/apply)
#

set -euo pipefail

OUTPUT="${1:-test_audio.wav}"

# Freesound track details
FREESOUND_ID="527676"
FREESOUND_URL="https://freesound.org/people/Andrewkn/sounds/${FREESOUND_ID}/"
TRACK_NAME="Race by Andrew kn (Andrewkn)"
TRACK_LICENSE="CC-BY 4.0"
EXPECTED_DURATION="2:33"

echo "============================================"
echo "rsona Test Audio Download"
echo "============================================"
echo ""

# Function to download from Freesound
download_from_freesound() {
    if [ -z "${FREESOUND_API_KEY:-}" ]; then
        echo "⚠️  No Freesound API key found"
        echo ""
        echo "To download from Freesound:"
        echo "  1. Get free API key: https://freesound.org/apiv2/apply"
        echo "  2. Set environment variable:"
        echo "     export FREESOUND_API_KEY='your-key-here'"
        echo ""
        return 1
    fi

    echo "📥 Downloading from Freesound..."
    echo "Track: ${TRACK_NAME}"
    echo "URL: ${FREESOUND_URL}"
    echo "License: ${TRACK_LICENSE}"
    echo ""

    # Freesound download endpoint requires authentication
    # Note: This downloads the original uploaded file (usually WAV format)
    local download_url="https://freesound.org/apiv2/sounds/${FREESOUND_ID}/download/?token=${FREESOUND_API_KEY}"

    # Download the audio file with authentication
    if curl -L -f -o "${OUTPUT}" "${download_url}"; then
        echo ""
        echo "✓ Downloaded from Freesound successfully!"
        return 0
    else
        echo "❌ Failed to download from Freesound"
        return 1
    fi
}

# Function to copy from bench/audio if exists
copy_from_bench_audio() {
    # Check for any audio files in bench/audio
    if [ -d "bench/audio" ]; then
        local audio_file
        audio_file=$(find bench/audio -name "*.wav" -o -name "*.mp3" | head -1)

        if [ -n "${audio_file}" ] && [ -f "${audio_file}" ]; then
            echo "📁 Found existing audio in bench/audio/"
            echo "Copying: ${audio_file}"
            echo ""

            if cp "${audio_file}" "${OUTPUT}"; then
                echo "✓ Copied existing audio file successfully!"
                return 0
            fi
        fi
    fi

    return 1
}

# Function to generate synthetic audio as fallback
generate_synthetic_audio() {
    echo "🔧 Generating synthetic test audio with sox..."
    echo ""

    # Check if sox is installed
    if ! command -v sox &> /dev/null; then
        echo "❌ Error: sox is required for audio generation"
        echo ""
        echo "Install with:"
        echo "  macOS:  brew install sox"
        echo "  Ubuntu: sudo apt-get install sox libsox-fmt-all"
        echo "  Windows: Download from https://sourceforge.net/projects/sox/"
        echo ""
        return 1
    fi

    # Generate 15-second test audio with simulated beats
    # - Two-tone harmony: 220 Hz (A3) + 330 Hz (E4)
    # - Tremolo effect: 2 Hz creates ~120 BPM beat pattern
    # - Fade in/out to avoid clicks
    sox -n -r 44100 -c 2 "${OUTPUT}" \
        synth 15 sine 220 sine 330 \
        fade 0.1 15 0.1 tremolo 2 50 gain -3

    if [ -f "${OUTPUT}" ]; then
        echo ""
        echo "✓ Generated synthetic test audio!"
        echo ""
        echo "⚠️  Note: This is synthetic audio for basic testing."
        echo "   For better results, use real music from Freesound."
        return 0
    else
        echo "❌ Failed to generate test audio"
        return 1
    fi
}

# Try download methods in order of preference
echo "Attempting to acquire test audio..."
echo ""

# Method 1: Download from Freesound (preferred)
if download_from_freesound; then
    METHOD="Downloaded from Freesound"
# Method 2: Copy from bench/audio if exists
elif copy_from_bench_audio; then
    METHOD="Copied from bench/audio"
# Method 3: Generate synthetic audio (fallback)
elif generate_synthetic_audio; then
    METHOD="Generated synthetic audio"
else
    echo ""
    echo "❌ All methods failed. Could not acquire test audio."
    echo ""
    echo "Manual options:"
    echo "  1. Download manually from: ${FREESOUND_URL}"
    echo "  2. Place any audio file in bench/audio/"
    echo "  3. Install sox and run this script again"
    echo ""
    exit 1
fi

# Verify file was created
if [ ! -f "${OUTPUT}" ]; then
    echo "❌ Error: ${OUTPUT} was not created"
    exit 1
fi

# Show file info
echo ""
echo "============================================"
echo "✓ Test Audio Ready"
echo "============================================"
echo ""
ls -lh "${OUTPUT}"
echo ""
echo "Method: ${METHOD}"

# Try to show audio properties with sox if available
if command -v soxi &> /dev/null; then
    echo ""
    echo "Audio properties:"
    soxi "${OUTPUT}" | grep -E "(Channels|Sample Rate|Duration|Precision)"
fi

echo ""
echo "You can now run benchmarks:"
echo "  cargo run --release --bin rsona_bench -- ${OUTPUT}"
echo "  python3 bench/feature_parity.py ${OUTPUT}"
echo "  cargo run --example tempo_beats -- ${OUTPUT}"
echo ""

if [ "${METHOD}" = "Generated synthetic audio" ]; then
    echo "============================================"
    echo "💡 Tip: Get Real Music"
    echo "============================================"
    echo ""
    echo "For testing with real music, set up Freesound:"
    echo ""
    echo "  1. Create free account: https://freesound.org/home/register/"
    echo "  2. Get API key: https://freesound.org/apiv2/apply"
    echo "  3. Export key:"
    echo "     export FREESOUND_API_KEY='your-key-here'"
    echo "  4. Run this script again"
    echo ""
    echo "Track: ${TRACK_NAME}"
    echo "License: ${TRACK_LICENSE}"
    echo "URL: ${FREESOUND_URL}"
    echo ""
fi

echo "Attribution: See ATTRIBUTION.md for licensing details"
echo ""
