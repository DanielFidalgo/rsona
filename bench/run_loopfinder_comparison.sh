#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
AUDIO_FILE=""
ITERATIONS=1
WIN_SECONDS=8.0
MIN_LOOP_SECONDS=10.0
OUTPUT_FILE=""
SKIP_BUILD=0
SAVE_JSON=0

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Loop Finder Comparison Benchmark${NC}"
    echo -e "${BLUE}  librosa_loopfinder vs rsona${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

print_usage() {
    cat << EOF
Usage: $0 [audio_file] [options]

Compare librosa_loopfinder vs rsona loop finding performance.

Arguments:
    audio_file              Path to audio file (default: bench/audio/*.wav)

Options:
    -i, --iterations N      Number of iterations for timing (default: 1)
    -w, --win-seconds N     Feature window length in seconds (default: 8.0)
    -m, --min-loop-seconds N Minimum loop duration in seconds (default: 10.0)
    -o, --output FILE       Save report to markdown file
    --json                  Also save JSON comparison data
    --skip-build            Skip rebuilding Rust binary
    -h, --help              Show this help message

Examples:
    # Run with default audio file
    $0

    # Run with custom audio file
    $0 path/to/audio.mp3

    # Run with multiple iterations and save report
    $0 -i 5 -o comparison_report.md

    # Quick run without rebuilding
    $0 --skip-build
EOF
}

check_python_deps() {
    echo -e "${YELLOW}Checking Python dependencies...${NC}"

    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: python3 not found${NC}"
        echo "Please install Python 3.8 or later"
        exit 1
    fi

    # Check for required Python packages
    python3 -c "import librosa" 2>/dev/null || {
        echo -e "${RED}Error: librosa not found${NC}"
        echo "Install with: pip install librosa"
        exit 1
    }

    python3 -c "import numpy" 2>/dev/null || {
        echo -e "${RED}Error: numpy not found${NC}"
        echo "Install with: pip install numpy"
        exit 1
    }

    python3 -c "import sklearn" 2>/dev/null || {
        echo -e "${RED}Error: scikit-learn not found${NC}"
        echo "Install with: pip install scikit-learn"
        exit 1
    }

    python3 -c "from librosa_loopfinder import find_loop_points" 2>/dev/null || {
        echo -e "${RED}Error: librosa_loopfinder not found${NC}"
        echo "Install with: pip install git+https://github.com/Kexanone/librosa_loopfinder.git"
        exit 1
    }

    echo -e "${GREEN}✓ All Python dependencies found${NC}"
}

build_rust() {
    if [ "$SKIP_BUILD" -eq 1 ]; then
        echo -e "${YELLOW}Skipping Rust build${NC}"
        return
    fi

    echo -e "${YELLOW}Building Rust binary...${NC}"
    cd "$PROJECT_ROOT"

    if [ -n "$RUSTFLAGS" ]; then
        echo "Using RUSTFLAGS: $RUSTFLAGS"
        cargo build --release --bin loopfinder_bench
    else
        echo "Building with target-cpu=native for best performance"
        RUSTFLAGS="-C target-cpu=native" cargo build --release --bin loopfinder_bench
    fi

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Rust binary built successfully${NC}"
    else
        echo -e "${RED}Error: Failed to build Rust binary${NC}"
        exit 1
    fi
}

find_default_audio() {
    # Look for audio files in bench/audio/
    local audio_dir="$SCRIPT_DIR/audio"

    if [ -f "$audio_dir/edm_loop.wav" ]; then
        echo "$audio_dir/edm_loop.wav"
    elif [ -f "$audio_dir/Alex-Productions - Future Bass Technology _ Shades.wav" ]; then
        echo "$audio_dir/Alex-Productions - Future Bass Technology _ Shades.wav"
    else
        # Find any .wav or .mp3 file
        local first_audio=$(find "$audio_dir" -type f \( -name "*.wav" -o -name "*.mp3" \) | head -1)
        if [ -n "$first_audio" ]; then
            echo "$first_audio"
        else
            echo ""
        fi
    fi
}

run_comparison() {
    local audio_path="$1"

    if [ ! -f "$audio_path" ]; then
        echo -e "${RED}Error: Audio file not found: $audio_path${NC}"
        exit 1
    fi

    echo -e "${YELLOW}Running comparison on: $(basename "$audio_path")${NC}"
    echo ""

    # Build command
    local cmd="python3 $SCRIPT_DIR/compare_loopfinder.py"
    cmd="$cmd \"$audio_path\""
    cmd="$cmd -i $ITERATIONS"
    cmd="$cmd -w $WIN_SECONDS"
    cmd="$cmd -m $MIN_LOOP_SECONDS"

    if [ -n "$OUTPUT_FILE" ]; then
        cmd="$cmd -o \"$OUTPUT_FILE\""
    fi

    if [ "$SAVE_JSON" -eq 1 ]; then
        cmd="$cmd --json"
    fi

    # Run comparison
    eval $cmd

    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ Comparison completed successfully${NC}"

        if [ -n "$OUTPUT_FILE" ]; then
            echo -e "${GREEN}  Report saved to: $OUTPUT_FILE${NC}"
        fi

        if [ "$SAVE_JSON" -eq 1 ]; then
            local json_file="${OUTPUT_FILE%.md}.json"
            if [ -z "$OUTPUT_FILE" ]; then
                json_file="comparison.json"
            fi
            echo -e "${GREEN}  JSON data saved to: $json_file${NC}"
        fi
    else
        echo -e "${RED}Error: Comparison failed${NC}"
        exit 1
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_usage
            exit 0
            ;;
        -i|--iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        -w|--win-seconds)
            WIN_SECONDS="$2"
            shift 2
            ;;
        -m|--min-loop-seconds)
            MIN_LOOP_SECONDS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --json)
            SAVE_JSON=1
            shift
            ;;
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        -*)
            echo -e "${RED}Error: Unknown option: $1${NC}"
            echo ""
            print_usage
            exit 1
            ;;
        *)
            AUDIO_FILE="$1"
            shift
            ;;
    esac
done

# Main execution
print_header

# Find audio file if not specified
if [ -z "$AUDIO_FILE" ]; then
    AUDIO_FILE=$(find_default_audio)
    if [ -z "$AUDIO_FILE" ]; then
        echo -e "${RED}Error: No audio file found in bench/audio/${NC}"
        echo "Please specify an audio file or add one to bench/audio/"
        echo ""
        print_usage
        exit 1
    fi
    echo -e "${BLUE}Using default audio file: $(basename "$AUDIO_FILE")${NC}"
    echo ""
fi

# Check dependencies
check_python_deps
echo ""

# Build Rust binary
build_rust
echo ""

# Run comparison
run_comparison "$AUDIO_FILE"
