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

# Function to download from Freesound using OAuth 2.0 client credentials flow
download_from_freesound() {
    if [ -z "${FREESOUND_API_KEY:-}" ]; then
        echo "⚠️  No Freesound API key found"
        echo ""
        echo "To download from Freesound:"
        echo "  1. Get free API key: https://freesound.org/apiv2/apply"
        echo "  2. Set environment variables:"
        echo "     export FREESOUND_API_KEY='your-api-key'"
        echo "     export FREESOUND_CLIENT_ID='your-client-id'"
        echo ""
        return 1
    fi

    if [ -z "${FREESOUND_CLIENT_ID:-}" ]; then
        echo "⚠️  No Freesound client ID found"
        echo ""
        echo "To download from Freesound:"
        echo "  1. Get free API key and client ID: https://freesound.org/apiv2/apply"
        echo "  2. Set environment variables:"
        echo "     export FREESOUND_API_KEY='your-api-key'"
        echo "     export FREESOUND_CLIENT_ID='your-client-id'"
        echo ""
        return 1
    fi

    echo "📥 Downloading from Freesound..."
    echo "Track: ${TRACK_NAME}"
    echo "URL: ${FREESOUND_URL}"
    echo "License: ${TRACK_LICENSE}"
    echo ""

    # OAuth 2.0 Client Credentials Flow
    # Step 1: Obtain access token
    echo "Step 1: Obtaining OAuth 2.0 access token..."
    echo "Client ID: ${FREESOUND_CLIENT_ID:0:8}... (truncated)"

    local token_url="https://freesound.org/apiv2/oauth2/access_token/"
    local token_response
    local access_token

    # Request access token using client credentials grant
    token_response=$(curl -s -f -X POST "${token_url}" \
        -d "grant_type=client_credentials" \
        -d "client_id=${FREESOUND_CLIENT_ID}" \
        -d "client_secret=${FREESOUND_API_KEY}" 2>&1)

    if [ $? -ne 0 ]; then
        echo "❌ Failed to obtain access token from Freesound OAuth endpoint"
        echo "Response: ${token_response}"
        return 1
    fi

    # Parse access token from JSON response
    if command -v jq &> /dev/null; then
        access_token=$(echo "${token_response}" | jq -r '.access_token')
    elif command -v python3 &> /dev/null; then
        access_token=$(echo "${token_response}" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('access_token', ''))" 2>/dev/null)
    else
        # Fallback: simple grep and cut (less reliable)
        access_token=$(echo "${token_response}" | grep -o '"access_token":"[^"]*"' | cut -d'"' -f4)
    fi

    if [ -z "${access_token}" ] || [ "${access_token}" = "null" ] || [ "${access_token}" = "None" ]; then
        echo "❌ Failed to extract access token from response"
        echo "Response: ${token_response}"
        return 1
    fi

    echo "✓ Access token obtained"
    echo ""

    # Step 2: Download audio file using access token
    echo "Step 2: Downloading audio file..."
    local download_url="https://freesound.org/apiv2/sounds/${FREESOUND_ID}/download/"

    # Use Bearer token authentication (OAuth 2.0 standard)
    if curl -L -f -o "${OUTPUT}" \
        -H "Authorization: Bearer ${access_token}" \
        "${download_url}"; then
        echo ""
        echo "✓ Downloaded from Freesound successfully!"

        # Verify it's an audio file
        if command -v file &> /dev/null; then
            local file_type=$(file -b "${OUTPUT}")
            echo "File type: ${file_type}"
        fi

        return 0
    else
        echo "❌ Failed to download from Freesound"
        echo ""
        echo "Possible issues:"
        echo "  - Invalid access token"
        echo "  - Sound not available for download"
        echo "  - Network connectivity issues"
        echo ""
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
