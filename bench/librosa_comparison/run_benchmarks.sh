#!/usr/bin/env bash
# rsona vs librosa Benchmark Suite Runner
#
# This script orchestrates the complete benchmarking pipeline:
# 1. Dependency checks
# 2. Rust compilation
# 3. librosa benchmark execution
# 4. rsona benchmark execution
# 5. Correctness validation
# 6. Performance analysis
# 7. Artifact generation
#
# Usage:
#   ./run_benchmarks.sh [options]
#
# Options:
#   --audio PATH          Path to audio file (default: bench/audio/test.wav)
#   --runs N              Number of benchmark runs (default: 10)
#   --warmup N            Number of warmup runs (default: 2)
#   --output-dir PATH     Output directory for artifacts (default: artifacts)
#   --skip-build          Skip Rust compilation
#   --skip-validation     Skip correctness validation
#   --feature NAME        Run single feature only
#   --native-cpu          Build with native CPU optimizations
#   --clean               Clean previous artifacts before running
#   --verbose             Enable verbose output
#   --help                Show this help message

set -e  # Exit on error

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCH_DIR="$SCRIPT_DIR"

# Default configuration
AUDIO_FILE=""
RUNS=10
WARMUP=2
OUTPUT_DIR="$BENCH_DIR/artifacts"
SKIP_BUILD=false
SKIP_VALIDATION=false
FEATURE=""
NATIVE_CPU=false
CLEAN=false
VERBOSE=false
SPEC_FILE="$BENCH_DIR/benchmark_spec.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

log_step() {
    echo ""
    echo -e "${GREEN}==>${NC} $*"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --audio)
            AUDIO_FILE="$2"
            shift 2
            ;;
        --runs)
            RUNS="$2"
            shift 2
            ;;
        --warmup)
            WARMUP="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --feature)
            FEATURE="$2"
            shift 2
            ;;
        --native-cpu)
            NATIVE_CPU=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            head -n 25 "$0" | grep "^#" | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Find default audio file if not specified
if [[ -z "$AUDIO_FILE" ]]; then
    # Look for audio files in bench/audio/
    AUDIO_DIR="$PROJECT_ROOT/bench/audio"
    if [[ -d "$AUDIO_DIR" ]]; then
        # Find first audio file
        AUDIO_FILE=$(find "$AUDIO_DIR" -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" \) | head -n 1)
    fi

    if [[ -z "$AUDIO_FILE" ]]; then
        log_error "No audio file specified and none found in bench/audio/"
        log_error "Please provide an audio file with --audio PATH"
        exit 1
    fi

    log_info "Using audio file: $AUDIO_FILE"
fi

# Verify audio file exists
if [[ ! -f "$AUDIO_FILE" ]]; then
    log_error "Audio file not found: $AUDIO_FILE"
    exit 1
fi

# Clean previous artifacts if requested
if [[ "$CLEAN" == "true" ]]; then
    log_step "Cleaning previous artifacts"
    if [[ -d "$OUTPUT_DIR" ]]; then
        rm -rf "$OUTPUT_DIR"
        log_success "Cleaned $OUTPUT_DIR"
    fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set verbose mode
if [[ "$VERBOSE" == "true" ]]; then
    set -x
fi

log_step "rsona vs librosa Benchmark Suite"
log_info "Configuration:"
log_info "  Audio file: $AUDIO_FILE"
log_info "  Runs: $RUNS (warmup: $WARMUP)"
log_info "  Output dir: $OUTPUT_DIR"
log_info "  Feature: ${FEATURE:-all}"

# Check dependencies
log_step "Checking dependencies"

# Check Python
if ! command -v python3 &> /dev/null; then
    log_error "python3 is not installed"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
log_success "Python: $PYTHON_VERSION"

# Check Rust/Cargo
if ! command -v cargo &> /dev/null; then
    log_error "cargo is not installed"
    log_error "Install Rust from https://rustup.rs/"
    exit 1
fi
RUST_VERSION=$(cargo --version | awk '{print $2}')
log_success "Rust: $RUST_VERSION"

# Check Python dependencies
log_info "Checking Python dependencies..."
MISSING_DEPS=()

python3 -c "import librosa" 2>/dev/null || MISSING_DEPS+=("librosa")
python3 -c "import numpy" 2>/dev/null || MISSING_DEPS+=("numpy")
python3 -c "import scipy" 2>/dev/null || MISSING_DEPS+=("scipy")
python3 -c "import matplotlib" 2>/dev/null || MISSING_DEPS+=("matplotlib")

if [[ ${#MISSING_DEPS[@]} -gt 0 ]]; then
    log_error "Missing Python dependencies: ${MISSING_DEPS[*]}"
    log_error "Install with: pip install ${MISSING_DEPS[*]}"
    exit 1
fi

log_success "All Python dependencies installed"

# Build Rust benchmark runner
if [[ "$SKIP_BUILD" != "true" ]]; then
    log_step "Building rsona benchmark runner"

    cd "$PROJECT_ROOT"

    if [[ "$NATIVE_CPU" == "true" ]]; then
        log_info "Building with native CPU optimizations..."
        RUSTFLAGS="-C target-cpu=native" cargo build --release --bin rsona_runner
    else
        log_info "Building in release mode..."
        cargo build --release --bin rsona_runner
    fi

    log_success "Built rsona_runner"
else
    log_info "Skipping Rust build (--skip-build)"
fi

# Define output files
LIBROSA_OUTPUT="$OUTPUT_DIR/librosa_results.json"
RSONA_OUTPUT="$OUTPUT_DIR/rsona_results.json"
VALIDATION_OUTPUT="$OUTPUT_DIR/validation_results.json"
ANALYSIS_OUTPUT="$OUTPUT_DIR/performance_analysis.json"

# Run librosa benchmark
log_step "Running librosa benchmark"

LIBROSA_CMD=(
    python3 "$BENCH_DIR/librosa_runner.py"
    --audio "$AUDIO_FILE"
    --runs "$RUNS"
    --warmup "$WARMUP"
    --output "$LIBROSA_OUTPUT"
)

if [[ -n "$FEATURE" ]]; then
    LIBROSA_CMD+=(--feature "$FEATURE")
else
    LIBROSA_CMD+=(--spec "$SPEC_FILE")
fi

log_info "Command: ${LIBROSA_CMD[*]}"
"${LIBROSA_CMD[@]}"

if [[ -f "$LIBROSA_OUTPUT" ]]; then
    log_success "librosa results saved to $LIBROSA_OUTPUT"
else
    log_error "librosa benchmark failed to produce output"
    exit 1
fi

# Run rsona benchmark
log_step "Running rsona benchmark"

# Use cargo run to handle non-standard target directories
RSONA_CMD=(
    cargo run --release --bin rsona_runner --
    --audio "$AUDIO_FILE"
    --runs "$RUNS"
    --warmup "$WARMUP"
    --output "$RSONA_OUTPUT"
)

if [[ -n "$FEATURE" ]]; then
    RSONA_CMD+=(--feature "$FEATURE")
fi

log_info "Command: ${RSONA_CMD[*]}"
cd "$PROJECT_ROOT"
"${RSONA_CMD[@]}"
cd "$BENCH_DIR"

if [[ -f "$RSONA_OUTPUT" ]]; then
    log_success "rsona results saved to $RSONA_OUTPUT"
else
    log_error "rsona benchmark failed to produce output"
    exit 1
fi

# Run correctness validation
if [[ "$SKIP_VALIDATION" != "true" ]]; then
    log_step "Validating correctness"

    VALIDATION_CMD=(
        python3 "$BENCH_DIR/validate_correctness.py"
        --rsona "$RSONA_OUTPUT"
        --librosa "$LIBROSA_OUTPUT"
        --spec "$SPEC_FILE"
        --output "$VALIDATION_OUTPUT"
    )

    log_info "Command: ${VALIDATION_CMD[*]}"

    if "${VALIDATION_CMD[@]}"; then
        log_success "Validation results saved to $VALIDATION_OUTPUT"

        # Display validation summary
        VALIDATION_PASSED=$(python3 -c "import json; print(json.load(open('$VALIDATION_OUTPUT'))['passed'])")
        if [[ "$VALIDATION_PASSED" == "True" ]]; then
            log_success "✓ All correctness checks PASSED"
        else
            log_warn "⚠ Some correctness checks FAILED"
            log_warn "See $VALIDATION_OUTPUT for details"
        fi
    else
        log_warn "Validation encountered errors (continuing anyway)"
    fi
else
    log_info "Skipping correctness validation (--skip-validation)"
fi

# Run performance analysis
log_step "Analyzing performance"

ANALYSIS_CMD=(
    python3 "$BENCH_DIR/analyze_performance.py"
    --rsona "$RSONA_OUTPUT"
    --librosa "$LIBROSA_OUTPUT"
    --spec "$SPEC_FILE"
    --output-dir "$OUTPUT_DIR"
    --format all
)

log_info "Command: ${ANALYSIS_CMD[*]}"
"${ANALYSIS_CMD[@]}"

log_success "Analysis complete - artifacts generated in $OUTPUT_DIR"

# Display summary
log_step "Benchmark Summary"

# Extract and display key metrics
if command -v jq &> /dev/null; then
    echo ""
    log_info "Performance Summary:"

    if [[ -f "$OUTPUT_DIR/performance_analysis.json" ]]; then
        echo ""
        jq -r '.summary | "  Total Features: \(.total_features)\n  Mean Speedup: \(.speedup.mean | tonumber | . * 100 | round / 100)x\n  Geometric Mean Speedup: \(.speedup.geometric_mean | tonumber | . * 100 | round / 100)x\n  Total Time Improvement: \(.total_time.improvement_pct | tonumber | . * 100 | round / 100)%"' "$OUTPUT_DIR/performance_analysis.json"

        echo ""
        log_info "Fastest Features (biggest speedup):"
        jq -r '.summary.fastest_features[] | "  - \(.feature): \(.speedup | . * 100 | round / 100)x faster (\(.librosa_ms | . * 100 | round / 100)ms → \(.rsona_ms | . * 100 | round / 100)ms)"' "$OUTPUT_DIR/performance_analysis.json"

        echo ""
        log_info "Slowest Features (smallest speedup):"
        jq -r '.summary.slowest_features[] | "  - \(.feature): \(.speedup | . * 100 | round / 100)x (\(.librosa_ms | . * 100 | round / 100)ms → \(.rsona_ms | . * 100 | round / 100)ms)"' "$OUTPUT_DIR/performance_analysis.json"
    fi

    if [[ -f "$VALIDATION_OUTPUT" ]] && [[ "$SKIP_VALIDATION" != "true" ]]; then
        echo ""
        log_info "Correctness Summary:"
        jq -r '.summary | "  Total Features: \(.total_features)\n  Passed: \(.passed)\n  Failed: \(.failed)\n  Pass Rate: \(.pass_rate * 100 | round)%"' "$VALIDATION_OUTPUT"
    fi
else
    log_warn "jq not installed - skipping summary display"
    log_info "Install jq to see formatted summary: https://stedolan.github.io/jq/"
fi

echo ""
log_success "Benchmark suite completed successfully!"
echo ""
log_info "Generated artifacts:"
log_info "  - Librosa results: $LIBROSA_OUTPUT"
log_info "  - rsona results: $RSONA_OUTPUT"
if [[ "$SKIP_VALIDATION" != "true" ]]; then
    log_info "  - Validation: $VALIDATION_OUTPUT"
fi
log_info "  - Performance analysis: $OUTPUT_DIR/performance_analysis.json"
log_info "  - CSV export: $OUTPUT_DIR/performance_comparison.csv"
log_info "  - Runtime chart: $OUTPUT_DIR/runtime_comparison.png"
log_info "  - Speedup chart: $OUTPUT_DIR/speedup_comparison.png"
log_info "  - Parity matrix: $OUTPUT_DIR/feature_parity_matrix.png"

echo ""
log_info "To view results:"
log_info "  cat $OUTPUT_DIR/performance_analysis.json | jq"
log_info "  open $OUTPUT_DIR/runtime_comparison.png"
echo ""
