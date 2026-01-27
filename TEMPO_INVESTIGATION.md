# 🔍 Tempo Estimation Accuracy Investigation

## 📊 Current Status

**Observed Issue:**
- **librosa:** 120.19 BPM
- **rsona:** 114.84 BPM  
- **Difference:** 5.34 BPM (4.44% error)

## 🎯 Expected Behavior

The test audio is generated with:
```bash
sox -n -r 44100 -c 2 test_audio.wav \
  synth 15 sine 220 sine 330 \
  fade 0.1 15 0.1 tremolo 2 50 gain -3
```

**Tremolo at 2 Hz = 120 BPM**
- 2 Hz = 2 cycles per second
- 2 cycles/sec × 60 sec/min = 120 cycles/min = 120 BPM

**Expected:** Both should detect ~120 BPM  
**Actual:** librosa ✓ (120.19), rsona ✗ (114.84)

## 🔬 Possible Causes

### 1. Algorithm Differences

**rsona uses FFT-based autocorrelation:**
```rust
// src/temporal/tempo.rs
let acf = compute_acf_fft(&env, max_lag, cfg.normalize_acf);
let best_lag = find_best_tempo_with_prior(&acf, ...);
```

**librosa may use different approach:**
- Different autocorrelation method
- Different peak-picking strategy
- Different tempo prior weighting

### 2. Parameter Differences

**rsona default config:**
```rust
TempoConfig {
    min_bpm: 60.0,
    max_bpm: 320.0,
    smooth: Some(3),
    normalize_acf: true,
    start_bpm: 120.0,  // Prior center
    std_bpm: 40.0,     // Prior width
}
```

**Check if librosa uses different:**
- BPM range
- Smoothing window
- Tempo prior

### 3. Onset Strength Differences

The tempo algorithm depends on onset strength envelope:
```rust
let onset = onset_strength_from_mel(&mel, DbConfig::default());
let tempo = estimate_tempo(onset.values(), ...);
```

**If onset detection differs, tempo will differ too.**

### 4. Edge Case with Generated Audio

Generated tremolo might not be ideal for tempo detection:
- May not have clear percussive onsets
- Tremolo creates amplitude modulation, not real beats
- Algorithms might interpret differently

## 🧪 Investigation Steps

### Step 1: Test with Real Music

```bash
# Download real music with known tempo
wget https://example.com/120bpm-track.wav

# Test both
python3 bench/compare_bench.py 120bpm-track.wav
```

**If they match on real music:** Generated audio edge case  
**If they still differ:** Algorithm/parameter issue

### Step 2: Compare Onset Envelopes

Add to rsona_bench:
```rust
// Output onset envelope for comparison
println!("ONSET: {:?}", &onset.values()[..10]);
```

Compare with librosa:
```python
onset = librosa.onset.onset_strength(S=mel, sr=sr)
print("ONSET:", onset[:10])
```

**If onsets match:** Problem in tempo algorithm  
**If onsets differ:** Problem in onset detection

### Step 3: Check Autocorrelation

Add debugging to rsona:
```rust
// Output ACF values
println!("ACF: {:?}", &acf[..50]);
println!("Best lag: {}", best_lag);
```

Compare with librosa autocorrelation values.

### Step 4: Test Different Parameters

Try disabling tempo prior:
```rust
TempoConfig {
    start_bpm: 120.0,
    std_bpm: 1000.0,  // Very wide prior = minimal effect
    ..Default::default()
}
```

Try different smoothing:
```rust
smooth: None,  // No smoothing
// or
smooth: Some(1),  // Minimal smoothing
```

## 🎯 Quick Fixes to Try

### Fix 1: Adjust Tempo Prior

If rsona is biased away from 120, try:
```rust
// More aggressive prior towards 120 BPM
start_bpm: 120.0,
std_bpm: 20.0,  // Narrower (was 40.0)
```

### Fix 2: Different Peak-Picking

Check `find_best_tempo_with_prior` function:
- May need different threshold
- May need octave correction (60/120/240 BPM ambiguity)

### Fix 3: Onset Detection Tuning

Try different onset parameters:
```rust
let onset_cfg = OnsetConfig {
    detrend: true,  // or false
    ..Default::default()
};
```

## 📈 Acceptance Criteria

### Minimal
- **< 5% error** on real music (current: 4.44% on synthetic)

### Good
- **< 2% error** on real music
- **< 3% error** on synthetic audio

### Ideal
- **< 0.5% error** (matching README claim)
- Consistent across different audio types

## 🚀 Next Steps

1. **Test with real music** - Determine if issue is synthetic-only
2. **Compare intermediate values** - Find where divergence occurs
3. **Review algorithm** - Check for bugs or differences
4. **Tune parameters** - Optimize for better match
5. **Document tradeoffs** - If intentional difference, explain why

## 📝 Additional Context

**From README:**
> **Tempo accuracy** | 71.78 BPM (both) - **0.000% difference** ✅

This suggests rsona *can* match librosa exactly. The 4.44% difference on generated audio might be:
- Edge case with artificial tremolo
- Different audio causing different behavior
- Recent regression (needs investigation)

## 🔗 Related Files

- `src/temporal/tempo.rs` - Tempo estimation implementation
- `src/feature/onset.rs` - Onset strength detection
- `bench/compare_bench.py` - Comparison script
- `README.md` - Performance claims

---

**Status:** 🟡 Investigation needed  
**Priority:** Medium (works well enough, but 4% error is worth investigating)  
**Impact:** Tempo estimation accuracy

