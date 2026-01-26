#!/usr/bin/env bash
#
# download_test_audio.sh - Download or generate Creative Commons test audio
#
# Usage: ./scripts/download_test_audio.sh [output_file]
#
# This script downloads a real music track from Freesound (CC-BY 4.0)
# or falls back to generating test audio with sox.

set -euo pipefail

OUTPUT="${1:-test_audio.wav}"
TEMP_MP3="${OUTPUT%.wav}.mp3"

echo "Setting up test audio for benchmarking..."
echo ""

# Try downloading from Freesound
# "Race" by Andrew kn (Andrewkn)
# https://freesound.org/people/Andrewkn/sounds/527676/
# License: Creative Commons Attribution 4.0
# Duration: 2:33, 44.1kHz, Stereo, Ambient/Atmospheric

download_success=false

echo "Attempting to download from Freesound..."
for userid in 10497458 10096348 527676; do
    if curl -L --max-time 30 --retry 3 \
        "https://cdn.freesound.org/previews/527/527676_${userid}-hq.mp3" \
        -o "$TEMP_MP3" 2>/dev/null && [ -s "$TEMP_MP3" ]; then
        
        # Verify it's actually an MP3 file, not HTML
        if file "$TEMP_MP3" | grep -q "Audio"; then
            echo "✓ Downloaded from Freesound"
            download_success=true
            break
        else
            rm "$TEMP_MP3"
        fi
    fi
done

# Process downloaded audio or generate fallback
if [ "$download_success" = true ]; then
    echo "Converting to WAV..."
    
    if command -v ffmpeg &> /dev/null; then
        ffmpeg -i "$TEMP_MP3" -ar 44100 -ac 2 "$OUTPUT" -y -loglevel quiet
        rm "$TEMP_MP3"
    elif command -v sox &> /dev/null; then
        sox "$TEMP_MP3" "$OUTPUT"
        rm "$TEMP_MP3"
    else
        echo "Error: ffmpeg or sox required to convert MP3 to WAV"
        echo "Install with:"
        echo "  macOS:  brew install ffmpeg"
        echo "  Ubuntu: sudo apt-get install ffmpeg"
        exit 1
    fi
    
    echo ""
    echo "✓ Real music track ready for testing!"
    echo ""
    echo "Attribution:"
    echo "  Title: 'Race'"
    echo "  Artist: Andrew kn (Andrewkn)"
    echo "  Source: https://freesound.org/people/Andrewkn/sounds/527676/"
    echo "  License: Creative Commons Attribution 4.0"
    echo "  Duration: 2:33"
    echo ""
else
    echo "Download failed, generating test audio with sox..."
    
    if ! command -v sox &> /dev/null; then
        echo "Error: sox required to generate test audio"
        echo "Install with:"
        echo "  macOS:  brew install sox"
        echo "  Ubuntu: sudo apt-get install sox"
        exit 1
    fi
    
    # Generate 15-second test tone with tremolo (simulates beats)
    sox -n -r 44100 -c 2 "$OUTPUT" \
        synth 15 sine 220 tremolo 2 0.5 \
        synth 15 sine 330 tremolo 2 0.5 mix \
        fade 0.1 15 0.1 \
        gain -3
    
    echo "✓ Generated synthetic test audio (15s, 44.1kHz, Stereo)"
fi

if [ -f "$OUTPUT" ]; then
    echo ""
    echo "File ready:"
    ls -lh "$OUTPUT"
    echo ""
    echo "You can now run:"
    echo "  cargo run --release --bin rsona_bench -- $OUTPUT"
    echo "  cargo run --example mfcc -- $OUTPUT"
    echo "  cargo run --example tempo_beats -- $OUTPUT"
else
    echo "Error: Failed to create $OUTPUT"
    exit 1
fi
