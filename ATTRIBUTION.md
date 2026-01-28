# Audio Attribution

This file documents Creative Commons licensed audio that can be used for rsona's testing and benchmarking.

## CI/CD Test Audio

**The CI/CD workflow uses generated audio (sox)** - no external downloads or attribution required.

## Optional: Real Music Track for Manual Testing

For users who want to test with real music instead of generated audio, we recommend:

### "Race" by Andrew kn

**Artist:** Andrew kn (Andrewkn)  
**Source:** [Freesound.org](https://freesound.org/people/Andrewkn/sounds/527676/)  
**License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)  
**Duration:** 2:33  
**Format:** WAV, 44.1 kHz, 16-bit, Stereo  
**Type:** Ambient/Atmospheric/Soundscape

**Description:** A progressive ambient track with light and beautiful pad sounds. Perfect for testing music information retrieval features including tempo detection, structure analysis, and spectral features.

**Usage in rsona:**
- **NOT used in CI/CD** (CI uses generated audio for reproducibility)
- Available for manual download from Freesound (requires free account)
- Can be used for local testing with real music
- Not committed to repository

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

## CI/CD Generated Audio

The CI/CD workflow uses generated synthetic test audio for consistency:
- 15-second stereo test tone
- Two frequencies (220 Hz + 330 Hz) with tremolo effect
- Simulates musical beats for tempo detection testing
- Generated on-demand with sox
- No attribution needed (generated content)
- Reproducible and deterministic across all runs

## License Compliance

This project (rsona) is licensed under Apache 2.0.

**For CI/CD:** Uses generated audio - no licensing concerns.

**For manual testing with CC-BY 4.0 audio:**
- ✅ **Compatible** - CC-BY allows commercial use
- ✅ **Properly attributed** - See above
- ✅ **Not bundled** - Must be downloaded separately, not in git
- ✅ **Clearly documented** - This file provides attribution

## Additional Resources

- [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
- [Freesound Terms of Use](https://freesound.org/help/tos_web/)
- [Artist's Freesound Profile](https://freesound.org/people/Andrewkn/)

## Contributing

If you want to recommend other Creative Commons audio for testing:
1. Ensure proper licensing (CC0, CC-BY, or similar)
2. Add attribution to this file
3. Provide download instructions
4. Verify compatibility with Apache 2.0

**Note:** CI/CD will continue using generated audio for consistency.

## Questions?

For questions about audio licensing or attribution, please open an issue on the rsona repository.

---

*Thank you to Andrew kn and the Freesound community for providing Creative Commons audio that makes open source projects like rsona possible!*