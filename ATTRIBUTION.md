# Audio Attribution

This file documents the Creative Commons licensed audio used in rsona's testing and benchmarking.

## Test Audio for CI/CD Benchmarks

### "Race" by Andrew kn

**Artist:** Andrew kn (Andrewkn)  
**Source:** [Freesound.org](https://freesound.org/people/Andrewkn/sounds/527676/)  
**License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)  
**Duration:** 2:33  
**Format:** WAV, 44.1 kHz, 16-bit, Stereo  
**Type:** Ambient/Atmospheric/Soundscape

**Description:** A progressive ambient track with light and beautiful pad sounds. Perfect for testing music information retrieval features including tempo detection, structure analysis, and spectral features.

**Usage in rsona:**
- Downloaded automatically in CI/CD workflows (`.github/workflows/benchmark.yml`)
- Can be downloaded locally with `./scripts/download_test_audio.sh`
- Used for consistent, reproducible performance benchmarks
- Not committed to repository (downloaded on-demand and cached)

**License Requirements (CC BY 4.0):**
- ✅ Attribution provided (this file)
- ✅ Link to original work provided above
- ✅ Link to license provided above
- ✅ Free to share and adapt for any purpose

**Attribution Text:**
```
"Race" by Andrew kn (Andrewkn)
https://freesound.org/people/Andrewkn/sounds/527676/
Licensed under Creative Commons Attribution 4.0
```

## Why This Audio?

We chose this track for benchmarking because:
1. **Real music content** - More realistic than synthetic tones
2. **Clear structure** - Good for testing segmentation algorithms
3. **Appropriate length** - 2:33 is perfect for benchmarks (not too long, not too short)
4. **High quality** - 44.1 kHz, stereo, good for MIR testing
5. **Proper licensing** - CC-BY 4.0 is compatible with open source
6. **Atmospheric/ambient** - Tests algorithms on non-percussive music

## Fallback Audio

If the Freesound download fails, the workflow generates synthetic test audio using sox:
- 15-second stereo test tone
- Two frequencies (220 Hz + 330 Hz) with tremolo
- Simulates musical beats for tempo detection testing
- No attribution needed (generated content)

## License Compliance

This project (rsona) is licensed under Apache 2.0. The CC-BY 4.0 audio is:
- ✅ **Compatible** - CC-BY allows commercial use
- ✅ **Properly attributed** - See above
- ✅ **Not bundled** - Downloaded separately, not in git
- ✅ **Clearly documented** - This file serves as attribution

## Additional Resources

- [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
- [Freesound Terms of Use](https://freesound.org/help/tos_web/)
- [Artist's Freesound Profile](https://freesound.org/people/Andrewkn/)

## Contributing

If you use rsona with different test audio, please:
1. Ensure proper licensing (CC0, CC-BY, or similar)
2. Add attribution to this file
3. Update scripts to download/generate the audio
4. Verify compatibility with Apache 2.0

## Questions?

For questions about audio licensing or attribution, please open an issue on the rsona repository.

---

*Thank you to Andrew kn and the Freesound community for providing Creative Commons audio that makes open source projects like rsona possible!*