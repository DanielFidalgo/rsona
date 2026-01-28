#!/usr/bin/env bash
#
# Compare loop detection results between Similarity and MusicalStructure modes
#
# This script demonstrates the difference between:
# - Similarity mode: Finds tightest similarity match (best for game loops)
# - MusicalStructure mode: Prefers musical phrase boundaries (best for DJ/composition)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default audio file
DEFAULT_AUDIO="$SCRIPT_DIR/audio/Alex-Productions - Future Bass Technology _ Shades.wav"
AUDIO_FILE="${1:-$DEFAULT_AUDIO}"

if [ ! -f "$AUDIO_FILE" ]; then
    echo -e "${RED}Error: Audio file not found: $AUDIO_FILE${NC}"
    echo "Usage: $0 [audio_file.wav]"
    exit 1
fi

echo -e "${CYAN}======================================${NC}"
echo -e "${CYAN}Loop Detection Mode Comparison${NC}"
echo -e "${CYAN}======================================${NC}"
echo ""
echo -e "${BLUE}Audio File:${NC} $(basename "$AUDIO_FILE")"
echo ""

# Build the binary if needed
if [ -n "$CARGO_TARGET_DIR" ]; then
    BINARY="$CARGO_TARGET_DIR/release/loopfinder_bench_beats"
else
    BINARY="$PROJECT_ROOT/target/release/loopfinder_bench_beats"
fi

if [ ! -f "$BINARY" ]; then
    echo -e "${YELLOW}Building loopfinder_bench_beats...${NC}"
    cd "$PROJECT_ROOT"
    cargo build --release --bin loopfinder_bench_beats

    # Update binary path after build
    if [ -n "$CARGO_TARGET_DIR" ]; then
        BINARY="$CARGO_TARGET_DIR/release/loopfinder_bench_beats"
    else
        BINARY="$PROJECT_ROOT/target/release/loopfinder_bench_beats"
    fi
    echo ""
fi

# Run Similarity mode
echo -e "${BLUE}Running Similarity mode (tightest match)...${NC}"
SIMILARITY_JSON=$(mktemp)
"$BINARY" "$AUDIO_FILE" --strategy=similarity > "$SIMILARITY_JSON" 2>/dev/null

# Run MusicalStructure mode
echo -e "${BLUE}Running MusicalStructure mode (phrase-aligned)...${NC}"
MUSICAL_JSON=$(mktemp)
"$BINARY" "$AUDIO_FILE" --strategy=musical > "$MUSICAL_JSON" 2>/dev/null

echo ""
echo -e "${CYAN}======================================${NC}"
echo -e "${CYAN}Results Comparison${NC}"
echo -e "${CYAN}======================================${NC}"
echo ""

# Extract key metrics using jq
if ! command -v jq &> /dev/null; then
    echo -e "${YELLOW}Warning: jq not found. Showing raw JSON.${NC}"
    echo ""
    echo -e "${BLUE}=== Similarity Mode ===${NC}"
    cat "$SIMILARITY_JSON"
    echo ""
    echo -e "${BLUE}=== MusicalStructure Mode ===${NC}"
    cat "$MUSICAL_JSON"
else
    # Parse results
    SIM_START=$(jq -r '.best_result.loop_begin_seconds' "$SIMILARITY_JSON")
    SIM_END=$(jq -r '.best_result.loop_end_seconds' "$SIMILARITY_JSON")
    SIM_DURATION=$(jq -r '.best_result.loop_duration_seconds' "$SIMILARITY_JSON")
    SIM_SCORE=$(jq -r '.best_result.score' "$SIMILARITY_JSON")
    SIM_TIME=$(jq -r '.timing.total_time_ms' "$SIMILARITY_JSON")

    MUS_START=$(jq -r '.best_result.loop_begin_seconds' "$MUSICAL_JSON")
    MUS_END=$(jq -r '.best_result.loop_end_seconds' "$MUSICAL_JSON")
    MUS_DURATION=$(jq -r '.best_result.loop_duration_seconds' "$MUSICAL_JSON")
    MUS_SCORE=$(jq -r '.best_result.score' "$MUSICAL_JSON")
    MUS_TIME=$(jq -r '.timing.total_time_ms' "$MUSICAL_JSON")

    TOTAL_DURATION=$(jq -r '.duration_seconds' "$SIMILARITY_JSON")
    SAMPLE_RATE=$(jq -r '.sample_rate' "$SIMILARITY_JSON")

    # Calculate differences
    START_DIFF=$(echo "$MUS_START - $SIM_START" | bc -l 2>/dev/null || echo "N/A")
    END_DIFF=$(echo "$MUS_END - $SIM_END" | bc -l 2>/dev/null || echo "N/A")
    DURATION_DIFF=$(echo "$MUS_DURATION - $SIM_DURATION" | bc -l 2>/dev/null || echo "N/A")

    # Format output
    printf "${GREEN}%-30s${NC} ${YELLOW}%-20s${NC} ${BLUE}%-20s${NC}\n" "Metric" "Similarity" "MusicalStructure"
    printf "%-30s %-20s %-20s\n" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" "━━━━━━━━━━━━━━━━━━━━" "━━━━━━━━━━━━━━━━━━━━"

    printf "%-30s ${YELLOW}%.3f${NC}s            ${BLUE}%.3f${NC}s\n" "Loop Start Time:" "$SIM_START" "$MUS_START"
    printf "%-30s ${YELLOW}%.3f${NC}s            ${BLUE}%.3f${NC}s\n" "Loop End Time:" "$SIM_END" "$MUS_END"
    printf "%-30s ${YELLOW}%.3f${NC}s            ${BLUE}%.3f${NC}s\n" "Loop Duration:" "$SIM_DURATION" "$MUS_DURATION"
    printf "%-30s ${YELLOW}%.4f${NC}              ${BLUE}%.4f${NC}\n" "Similarity Score:" "$SIM_SCORE" "$MUS_SCORE"
    printf "%-30s ${YELLOW}%.2f${NC}ms             ${BLUE}%.2f${NC}ms\n" "Processing Time:" "$SIM_TIME" "$MUS_TIME"

    echo ""
    echo -e "${CYAN}======================================${NC}"
    echo -e "${CYAN}Differences${NC}"
    echo -e "${CYAN}======================================${NC}"
    echo ""

    if [ "$START_DIFF" != "N/A" ] && [ "$END_DIFF" != "N/A" ] && [ "$DURATION_DIFF" != "N/A" ]; then
        printf "%-30s %.3fs\n" "Start Time Difference:" "$START_DIFF"
        printf "%-30s %.3fs\n" "End Time Difference:" "$END_DIFF"
        printf "%-30s %.3fs\n" "Duration Difference:" "$DURATION_DIFF"

        # Calculate as bars if tempo is available
        TEMPO=$(jq -r '.parameters.tempo_bpm // "null"' "$MUSICAL_JSON")
        if [ "$TEMPO" != "null" ] && [ "$TEMPO" != "0" ]; then
            BEATS_PER_SEC=$(echo "scale=4; $TEMPO / 60.0" | bc)
            SIM_BEATS=$(echo "scale=2; $SIM_DURATION * $BEATS_PER_SEC" | bc)
            MUS_BEATS=$(echo "scale=2; $MUS_DURATION * $BEATS_PER_SEC" | bc)
            SIM_BARS=$(echo "scale=2; $SIM_BEATS / 4.0" | bc)
            MUS_BARS=$(echo "scale=2; $MUS_BEATS / 4.0" | bc)

            echo ""
            printf "%-30s %.1f bars (%.1f beats)\n" "Similarity Loop Length:" "$SIM_BARS" "$SIM_BEATS"
            printf "%-30s %.1f bars (%.1f beats)\n" "MusicalStructure Length:" "$MUS_BARS" "$MUS_BEATS"
        fi
    fi

    echo ""
    echo -e "${CYAN}======================================${NC}"
    echo -e "${CYAN}Interpretation${NC}"
    echo -e "${CYAN}======================================${NC}"
    echo ""

    echo -e "${YELLOW}Similarity Mode:${NC}"
    echo "  ✓ Optimizes for tightest similarity match"
    echo "  ✓ Best for seamless game background loops"
    echo "  ✓ May find shorter loops"
    echo ""

    echo -e "${BLUE}MusicalStructure Mode:${NC}"
    echo "  ✓ Prefers power-of-2 bar counts (8, 16, 32, 64)"
    echo "  ✓ Best for DJ mixing and composition"
    echo "  ✓ Considers musical phrase boundaries"
    echo "  ✓ May sacrifice some similarity for structure"
    echo ""

    # Recommendation
    if [ "$START_DIFF" != "N/A" ]; then
        DIFF_ABS=$(echo "$DURATION_DIFF" | tr -d '-')
        LARGE_DIFF=$(echo "$DIFF_ABS > 10" | bc -l)

        if [ "$LARGE_DIFF" -eq 1 ]; then
            echo -e "${GREEN}Analysis:${NC}"
            echo "  The modes found significantly different loops!"
            echo "  Choose based on your use case:"
            echo "    • Game loops → Use Similarity mode"
            echo "    • DJ sets → Use MusicalStructure mode"
        else
            echo -e "${GREEN}Analysis:${NC}"
            echo "  The modes found similar loops."
            echo "  Either mode should work well for this track."
        fi
    fi
fi

# Cleanup
rm -f "$SIMILARITY_JSON" "$MUSICAL_JSON"

echo ""
echo -e "${CYAN}======================================${NC}"
echo "Full JSON output saved to: /tmp/similarity.json and /tmp/musical.json"
echo "Re-run with: $0 \"$AUDIO_FILE\""
echo -e "${CYAN}======================================${NC}"
