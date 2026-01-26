# Test Audio Files for Benchmarking

This document describes where to find Creative Commons audio files for testing and benchmarking rsona.

## Why No Committed Audio?

We don't commit audio files to the repository because:
- **Repository size**: Audio files are large and bloat git history
- **Licensing clarity**: External CC sources are clearly licensed
- **Flexibility**: Easy to test with different audio samples
- **CI/CD**: Downloaded on-demand and cached

## CI/CD Workflow

The GitHub Actions workflow (`benchmark.yml`) automatically:
1. Downloads "Race" by Andrew kn from Freesound (CC-BY 4.0)
2. Falls back to generated audio if download fails
3. Caches the audio for subsequent runs
4. Uses the same audio file for baseline and current comparisons

This ensures **reproducible benchmarks** with consistent, real-world music.

### Test Track Details

**Current Test Audio:**
- **Track:** "Race" by Andrew kn (Andrewkn)
- **Source:** https://freesound.org/people/Andrewkn/sounds/527676/
- **License:** Creative Commons Attribution 4.0 (CC-BY 4.0)
- **Duration:** 2:33 (153 seconds)
- **Format:** 44.1 kHz, 16-bit, Stereo WAV
- **Type:** Ambient/Atmospheric/Soundscape
- **Why this track:** Real music content with progressive sound, suitable for testing tempo detection, structure analysis, and spectral features

See [ATTRIBUTION.md](../ATTRIBUTION.md) for full attribution details.
</text>

<old_text line=22>
## Primary Approach: Generate with Sox

For **consistent, reproducible benchmarks**, we generate test audio with sox:

```bash
./scripts/download_test_audio.sh
```

This creates a 15-second audio file with:
- Simulated beats (~120 BPM) for tempo detection
- Two-tone harmony (220 Hz + 330 Hz)
- Stereo, 44.1 kHz, 16-bit PCM WAV
- Consistent across all runs

### Why Generate Instead of Download?

✅ **Reproducible** - Same audio every time  
✅ **No network issues** - Works offline  
✅ **Fast** - Instant generation vs slow downloads  
✅ **No licensing concerns** - Generated content  
✅ **Consistent benchmarks** - Fair comparisons

## Primary Approach: Generate with Sox

For **consistent, reproducible benchmarks**, we generate test audio with sox:

```bash
./scripts/download_test_audio.sh
```

This creates a 15-second audio file with:
- Simulated beats (~120 BPM) for tempo detection
- Two-tone harmony (220 Hz + 330 Hz)
- Stereo, 44.1 kHz, 16-bit PCM WAV
- Consistent across all runs

### Why Generate Instead of Download?

✅ **Reproducible** - Same audio every time  
✅ **No network issues** - Works offline  
✅ **Fast** - Instant generation vs slow downloads  
✅ **No licensing concerns** - Generated content  
✅ **Consistent benchmarks** - Fair comparisons

## Optional: Real-World Audio Sources

For testing with actual music (not benchmarking), you can use:

### 1. Freesound (CC0 / CC-BY)
- **URL**: https://freesound.org/
- **License**: Various CC licenses, many CC0
- **Best for**: Real-world audio, music loops, sound effects

### 2. Internet Archive Audio
- **URL**: https://archive.org/details/audio
- **License**: Public domain and CC licenses
- **Best for**: Music, spoken word, historical recordings

### 3. Free Music Archive (FMA)
- **URL**: https://freemusicarchive.org/
- **License**: Various CC licenses
- **Best for**: Full music tracks

## Recommended Test Audio Characteristics

For benchmarking rsona, use audio with:

### Duration
- **5-30 seconds**: Good for quick tests
- **1-3 minutes**: Standard benchmarks
- **3+ minutes**: Stress testing

### Format
- **WAV**: Uncompressed, best for consistency
- **FLAC**: Lossless, good alternative
- **MP3**: Common format, tests decoder

### Content
- **Music with clear beats**: Tests tempo detection
- **Loops**: Tests structure analysis
- **Varied sections**: Tests segmentation
- **Single frequency tones**: Tests accuracy (generated)

## Local Testing

### Generate test audio (recommended):

```bash
# Generate with the provided script (recommended)
./scripts/download_test_audio.sh

# Or generate manually with sox:
sox -n -r 44100 -c 2 test_audio.wav \
  synth 15 sine 220 tremolo 2 0.5 \
  synth 15 sine 330 tremolo 2 0.5 mix \
  fade 0.1 15 0.1 \
  gain -3
```

### Or use your own audio:

```bash
# Use any audio file you have
cp ~/Music/your-song.wav test_audio.wav
cp ~/Music/your-song.mp3 test_audio.mp3
```
</text>

<old_text line=102>
### Run benchmarks:

```bash
# Run with test audio
cargo run --release --bin rsona_bench -- test_audio/test.wav

# Run examples
cargo run --example mfcc -- test_audio/test.wav
cargo run --example tempo_beats -- test_audio/test.wav
```

### Run benchmarks:

```bash
# Run with test audio
cargo run --release --bin rsona_bench -- test_audio/test.wav

# Run examples
cargo run --example mfcc -- test_audio/test.wav
cargo run --example tempo_beats -- test_audio/test.wav
```

## Advanced: Generating Custom Test Tones

Use `sox` to generate specific test patterns:

```bash
# Install sox first:
# macOS: brew install sox
# Ubuntu: sudo apt-get install sox
# Windows: https://sourceforge.net/projects/sox/

# Pure tone (440 Hz, 10 seconds)
sox -n -r 44100 -c 2 pure_tone.wav synth 10 sine 440

# Frequency sweep (100 Hz to 10 kHz, 5 seconds)
sox -n -r 44100 -c 1 sweep.wav synth 5 sine 100-10000

# Click track (120 BPM, 10 seconds) 
sox -n -r 44100 -c 2 clicks.wav synth 10 pulse 2 gain -6

# White noise (5 seconds)
sox -n -r 44100 -c 2 noise.wav synth 5 whitenoise

# Musical pattern with beats (like CI uses)
sox -n -r 44100 -c 2 musical.wav \
  synth 15 sine 220 tremolo 2 0.5 \
  synth 15 sine 330 tremolo 2 0.5 mix \
  fade 0.1 15 0.1 gain -3
```

## Example: Good Test Files

### For Tempo/Beat Detection
- Music with clear beats (EDM, hip-hop, drums)
- 120-140 BPM range
- 10-30 seconds duration

### For Structure Analysis (Intro/Loop/Outro)
- Video game music with clear loop points
- Songs with distinct intro/verse/chorus sections
- 1-2 minutes duration

### For MFCC/Spectral Features
- Speech samples (for MFCC specifically)
- Music with varied instrumentation
- 5-15 seconds duration

## CI/CD Audio Download

The CI workflow downloads real music from Freesound:

**Primary (Downloaded):**
```bash
# Download "Race" by Andrew kn from Freesound
curl "https://cdn.freesound.org/previews/527/527676_*-hq.mp3" -o test_audio.mp3
ffmpeg -i test_audio.mp3 -ar 44100 -ac 2 test_audio.wav
```

**Fallback (Generated):**
```bash
sox -n -r 44100 -c 2 test_audio.wav \
  synth 15 sine 220 tremolo 2 0.5 \
  synth 15 sine 330 tremolo 2 0.5 mix \
  fade 0.1 15 0.1 gain -3
```

This ensures:
- **Real-world testing** with actual music
- **Reproducible benchmarks** - same track cached across runs
- **Fallback reliability** - generates audio if download fails
- **Proper attribution** - See ATTRIBUTION.md

## Adding to .gitignore

Test audio files should NOT be committed. Add to `.gitignore`:

```
# Test audio files
test_audio/
*.wav
*.mp3
*.flac
*.ogg
bench/audio/
```

(Note: Keep any audio in `examples/` if they're tiny reference files < 100KB)

## License Compliance

When using Creative Commons audio:

1. **CC0 (Public Domain)**: No attribution required, best for CI/CD
2. **CC-BY**: Attribution required, note in docs/comments
3. **CC-BY-SA**: Attribution + share-alike, compatible with Apache 2.0
4. **CC-BY-NC**: Non-commercial only, avoid for open source projects

Always verify license before use!

## Updating CI/CD Audio Generation

To change the audio generated in CI:

1. Update the sox command in `benchmark.yml`
2. Update the cache key to regenerate:

```yaml
key: test-audio-generated-v2  # Increment version
```

3. Test locally first:

```bash
./scripts/download_test_audio.sh
cargo run --release --bin rsona_bench -- test_audio.wav
```

## Troubleshooting

### "sox: command not found"
Install sox:
- **macOS**: `brew install sox`
- **Ubuntu**: `sudo apt-get install sox`
- **Windows**: Download from https://sourceforge.net/projects/sox/

### "No such file or directory"
- Run `./scripts/download_test_audio.sh` first
- Or generate manually with sox command above

### "Unsupported audio format"
- Ensure file is valid WAV format
- Check with: `file test_audio.wav`
- Should show: "RIFF (little-endian) data, WAVE audio"

## Resources

- [Creative Commons Search](https://search.creativecommons.org/)
- [Awesome Audio Sources](https://github.com/ad-si/awesome-audio-visualization)
- [SoX Documentation](http://sox.sourceforge.net/sox.html)
- [Freesound API](https://freesound.org/docs/api/)

## Contributing

If you find good CC test audio sources, please:
1. Verify the license
2. Add to this document
3. Include direct download link if possible
4. Note what it's good for testing