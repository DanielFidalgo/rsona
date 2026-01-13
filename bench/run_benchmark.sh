#!/bin/bash
# Benchmark runner script for rsona vs librosa comparison

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
ITERATIONS=5
OUTPUT_FORMAT="markdown"
OUTPUT_FILE=""
AUDIO_FILE=""

# Print colored message
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check dependencies
check_dependencies() {
    print_info "Checking dependencies..."

    # Check Rust
    if ! command_exists cargo; then
        print_error "Cargo not found. Please install Rust: https://rustup.rs/"
        exit 1
    fi
    print_info "✓ Cargo found: $(cargo --version | head -n1)"

    # Check Python
    if command_exists python3; then
        PYTHON_CMD="python3"
    elif command_exists python; then
        PYTHON_CMD="python"
    else
        print_error "Python not found. Please install Python 3.8+"
        exit 1
    fi
    print_info "✓ Python found: $($PYTHON_CMD --version)"

    # Check librosa
    if ! $PYTHON_CMD -c "import librosa" 2>/dev/null; then
        print_warn "librosa not found. Installing dependencies..."
        $PYTHON_CMD -m pip install -r "$SCRIPT_DIR/requirements.txt" || {
            print_error "Failed to install Python dependencies"
            print_info "Try manually: pip install librosa numpy"
            exit 1
        }
    fi
    print_info "✓ librosa found"
}

# Build rsona benchmark
build_rsona() {
    print_info "Building rsona benchmark in release mode..."
    cd "$PROJECT_DIR"
    cargo build --release --bin rsona_bench || {
        print_error "Failed to build rsona benchmark"
        exit 1
    }
    print_info "✓ Build complete"
}

# Find audio file
find_audio() {
    if [ -n "$AUDIO_FILE" ]; then
        if [ ! -f "$AUDIO_FILE" ]; then
            print_error "Audio file not found: $AUDIO_FILE"
            exit 1
        fi
        echo "$AUDIO_FILE"
    else
        # Find first MP3 in audio directory
        local audio_dir="$SCRIPT_DIR/audio"
        local audio=$(find "$audio_dir" -name "*.mp3" -o -name "*.wav" -o -name "*.flac" | head -n1)
        if [ -z "$audio" ]; then
            print_error "No audio files found in $audio_dir"
            print_info "Please add an audio file to the bench/audio/ directory"
            exit 1
        fi
        echo "$audio"
    fi
}

# Run benchmark
run_benchmark() {
    local audio=$(find_audio)
    print_info "Using audio file: $(basename "$audio")"
    print_info "Running $ITERATIONS iterations..."
    print_info ""

    cd "$PROJECT_DIR"

    local cmd="$PYTHON_CMD bench/compare_bench.py"
    cmd="$cmd \"$audio\""
    cmd="$cmd --iterations $ITERATIONS"
    cmd="$cmd --output $OUTPUT_FORMAT"

    if [ -n "$OUTPUT_FILE" ]; then
        cmd="$cmd --file \"$OUTPUT_FILE\""
    fi

    eval $cmd || {
        print_error "Benchmark failed"
        exit 1
    }

    print_info ""
    if [ -n "$OUTPUT_FILE" ]; then
        print_info "Results saved to: $OUTPUT_FILE"
    fi
}

# Usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] [AUDIO_FILE]

Run benchmark comparison between rsona and librosa.

OPTIONS:
    -n, --iterations N    Number of iterations (default: 5)
    -o, --output FORMAT   Output format: markdown, json, csv (default: markdown)
    -f, --file PATH       Save output to file
    -h, --help            Show this help message
    --skip-build          Skip building rsona (use existing binary)
    --skip-deps           Skip dependency checks

EXAMPLES:
    # Run with default settings
    $0

    # Run 10 iterations with specific audio file
    $0 -n 10 path/to/audio.mp3

    # Generate JSON report
    $0 -o json -f results.json

    # Quick run (skip checks and build)
    $0 --skip-deps --skip-build

EOF
}

# Parse arguments
SKIP_BUILD=false
SKIP_DEPS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        -f|--file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        -*)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            AUDIO_FILE="$1"
            shift
            ;;
    esac
done

# Main execution
main() {
    print_info "rsona vs librosa Benchmark Runner"
    print_info "=================================="
    print_info ""

    if [ "$SKIP_DEPS" = false ]; then
        check_dependencies
        print_info ""
    fi

    if [ "$SKIP_BUILD" = false ]; then
        build_rsona
        print_info ""
    fi

    run_benchmark

    print_info ""
    print_info "Benchmark complete! 🎉"
}

main
