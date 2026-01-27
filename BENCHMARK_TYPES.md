# 📊 Benchmark Types: Complete Overview

## 🎯 Two Complementary Benchmarks

The workflow now runs **TWO different benchmarks** in the same CI run:

### 1️⃣ Version Comparison (rsona vs rsona)
**Purpose:** Track performance changes between versions

**Compares:** 
- Baseline version (latest tag) vs Current code

**Measures:**
- Performance speedups/regressions
- Feature-by-feature timing
- Statistical consistency

**Output:** `BENCHMARK_RESULTS.md`

**Example:**
```markdown
## Summary
Overall Performance: 🚀 **FASTER**
- Speedup: 1.91x
- Improvement: +47.5%

| Feature | Baseline | Current | Speedup |
|---------|----------|---------|---------|
| Tempo   | 150ms    | 0.6ms   | 240x    |
```

---

### 2️⃣ Library Comparison (rsona vs librosa)
**Purpose:** Validate accuracy and compare against reference implementation

**Compares:**
- rsona (Rust) vs librosa (Python)

**Measures:**
- Absolute performance (both implementations)
- Speedup factor (how much faster is rsona)
- Accuracy (tempo, frame counts match)

**Output:** `LIBROSA_COMPARISON.md`

**Example:**
```markdown
## Benchmark Results

### Timing Comparison
- librosa: 127.45 ms
- rsona: 17.20 ms
- **Speedup: 7.41x faster** 🚀

### Accuracy
- Tempo: rsona 120.0 BPM vs librosa 120.0 BPM ✓
- Frames: 2581 (both) ✓
```

---

## 🔄 How They Work Together

```
Single CI Run
├─ Version Comparison (rsona vs rsona)
│  ├─ Detects: Regressions in your code
│  ├─ Tracks: Performance over time
│  └─ Output: BENCHMARK_RESULTS.md
│
└─ Library Comparison (rsona vs librosa)
   ├─ Validates: Correctness against reference
   ├─ Shows: Real-world speedup
   └─ Output: LIBROSA_COMPARISON.md
```

## 📊 When Each Is Useful

### Version Comparison
**When:**
- Making code changes
- Optimizing algorithms
- Reviewing PRs

**Answers:**
- "Did my change make it faster/slower?"
- "Which feature regressed?"
- "Is performance consistent?"

### Library Comparison
**When:**
- Validating correctness
- Marketing/documentation
- Onboarding new users

**Answers:**
- "Is rsona faster than librosa?"
- "Are the results accurate?"
- "By how much is it faster?"

## 📁 Artifacts Generated

Both reports are uploaded as artifacts:

```
benchmark-reports.zip
├─ BENCHMARK_RESULTS.md     (rsona vs rsona)
└─ LIBROSA_COMPARISON.md    (rsona vs librosa)
```

Retention: **90 days**

## 🎯 Summary in GitHub Actions

The Actions summary page shows both:

```
## Benchmark Complete 🎉

### Version Comparison
# rsona Benchmark Report
[Shows baseline vs current comparison]

---

### Librosa Comparison
# rsona vs librosa Benchmark
[Shows rsona vs librosa comparison]
```

## ⚙️ Configuration

Both benchmarks use the same settings:
- **Audio:** `test_audio.wav` (generated with sox)
- **Iterations:** 5
- **Warmup:** 2
- **Output:** Markdown format

## 🚀 Benefits of Both

**Version Comparison:**
✅ Catch regressions early
✅ Track optimization progress
✅ Compare different approaches

**Library Comparison:**
✅ Validate correctness
✅ Show real-world performance
✅ Compare against established reference

## 💡 Example Use Cases

### Scenario 1: Optimizing STFT
```
Version Comparison:
- Before: STFT 75ms
- After: STFT 15ms
- Result: 5x faster! ✅

Library Comparison:
- rsona STFT: 15ms
- librosa STFT: 75ms
- Result: Still 5x faster than librosa ✅
```

### Scenario 2: Bug Fix
```
Version Comparison:
- Before: Tempo 120 BPM
- After: Tempo 120 BPM
- Performance: Same
- Correctness: ✅ Now passes validation

Library Comparison:
- rsona: 120 BPM
- librosa: 120 BPM
- Result: Accuracy matches! ✅
```

## 🔧 Extending Further

Future enhancements could add:
- [ ] Comparison with other libraries (essentia, aubio)
- [ ] Bit-exact accuracy validation
- [ ] Feature-by-feature accuracy metrics
- [ ] Historical trend charts

## ✅ Summary

**You now have:**
1. 📈 **Version tracking** - Is your code getting faster?
2. 🎯 **Reference validation** - Is rsona as accurate as librosa?
3. 🚀 **Real-world metrics** - How much faster is rsona?
4. ✅ **Correctness checks** - Are outputs valid?

**All in one CI run!** 🎉

---

## Next CI Run Will Show

```
✓ Version Comparison: [baseline vs current]
✓ Library Comparison: [rsona vs librosa]
✓ Two reports uploaded
✓ Both visible in Actions summary
```

**Complete benchmark coverage!** 🚀
