# Changelog
All notable changes to **Scarlet** are documented here.

Scarlet follows a narrative‑driven development cycle.  
Each version reflects architectural, interpretability, and emotional‑memory milestones.

This project uses **Semantic Versioning** and the **Keep a Changelog** format.

## [1.6.2] – 2026‑03‑31
### Added
- Integrated full deep-history datasets for SOL, ETH, and BTC, each exceeding **100,000+ rows**.
- Enabled Scarlet to operate on long-horizon, multi-asset feature tensors without degrading performance.
- Added automatic LSTM windowing to support arbitrarily large datasets while preserving temporal fidelity.

### Fixed
- Resolved a critical cuDNN crash caused by feeding the LSTM full 100k‑step sequences.
  - Implemented a capped LSTM context window (configurable) to ensure GPU stability.
  - Ensured all LSTM inputs are contiguous and device‑aligned.
- Removed legacy double‑LSTM invocation that previously caused redundant compute and instability under large inputs.
- Eliminated timestamp contamination in the inference pipeline that surfaced once the LSTM crash was fixed.

### Improved
- Strengthened the feature pipeline to guarantee numeric‑only tensors and consistent schema across training, inference, and online RL.
- Improved runtime stability when operating with large multi‑asset buffers and long historical windows.
- Enhanced logging around feature counts, alignment, and tensor construction for easier debugging at scale.

### Notes
- This update marks Scarlet’s first fully stable run on **100k+ candle datasets** with live inference, transformer blocks, decoder, policy head, and online RL all active simultaneously.
- No changes were made to Scarlet’s forecasting logic; this release focuses on stability, scalability, and data‑pipeline integrity.

# 1.6.1
### Horizon‑Aligned Forecast Collapse & Fee‑Adjusted Policy Logic (2026‑03‑22)
## Major Improvements
- Updated policy delta collapse to match real trading cadence
Scarlet now collapses her 6‑step forecast vector using a weighted blend centered on the correct horizon for her 61‑minute cycle.
- New weights: [0, 0, 0.1, 0.6, 0.2, 0.1]
- Emphasizes step 4 (~60 minutes)
- Preserves multi‑horizon curvature
This ensures her policy delta reflects the candle she will actually trade on.
- Adjusted policy thresholds to account for increased trading fees
The delta scale was already correct — the update ensures Scarlet’s BUY/SELL gating logic respects the higher round‑trip fee environment without suppressing trades.
This keeps her behavior realistic and fee‑aware while maintaining sensitivity to micro‑deltas.
## Drift Detection Alignment
- Drift detection now uses the same horizon‑collapsed forecast as policy
Previously, drift compared Δ1 (15m) predictions to 61‑minute actual deltas, inflating drift and triggering unnecessary micro‑training.
Drift now uses the same weighted horizon‑4 collapse as the policy layer, ensuring:
- Correct temporal alignment
- Honest drift magnitude
- Stable online learning
- Fewer false positives
## Internal Logic Refinements
- Reordered drift block so forecast collapse occurs before drift computation.
- Ensured pred_deltas always reflect the horizon‑aligned forecast.
- Cleaned up per‑asset forecast vector extraction for consistency.
## Result
Scarlet now operates with fully aligned temporal geometry and fee‑aware policy logic:
- Forecast horizon
- Actual delta horizon
- Drift horizon
- Policy horizon
…all match the real‑world 61‑minute cycle (previously 16) and the updated higher fee environment.
This release significantly improves stability, coherence, and trading realism when faced with a high fee environment and low trading budget.
This effectively makes Scarlet a swing trader rather than a scalper.



## [1.6.0] – 2026‑02-28
### Added
- **Premium‑tier data pipeline** with full‑history support for research‑grade training.
- **Cycle‑aware charting hooks** for future visualization of Scarlet’s internal reasoning.
- **Improved emotional‑memory narrator lines**, including warm‑up completion, cache growth, and slope updates.
- **Enhanced drift‑detection system** using schema hashing and provenance tracking.
- **Thread‑safe memory writes** for stable long‑running research sessions.
- **Premium‑tier installer messaging** and support contact integration.

### Fixed
- Mismatch between **live sentiment‑aware Scarlet** and **offline sentiment‑blind Scarlet** during evaluation.
- Occasional memory corruption during rapid micro‑training cycles.
- Inconsistent warm‑up behavior when switching assets mid‑session.
- Rare deadlock conditions in the emotional‑memory subsystem.

### Changed
- Updated cycle cadence defaults (16‑minute research cadence).
- Refined emotional‑intelligence subsystem for more stable sentiment slopes.
- Improved internal API schema for multi‑asset research workflows.

### Added
- Ability to read own trade history file, learn from trades and update trading parameters based on profitable behavior.

---

## [1.5.0] – 2026‑02‑20
### Added
- **Decision Transparency Layer**: Scarlet now narrates trade suppression, posture vetoes, sizing logic, and internal reasoning.
- **CandleCache Inspector** for debugging training data and drift.
- **Candle Freshness Watchdog** to detect stale or inconsistent data.
- **Unified CandleCache class** for training + inference parity.

### Changed
- Major improvements to Scarlet’s interpretability pipeline.
- More consistent narrator tone and reasoning structure.

---

## [1.4.0] – 2026‑02‑10
### Added
- **Multi‑asset architecture** for cross‑symbol research.
- **Regime detector** and volatility metric for adaptive behavior.
- **Per‑subreddit sentiment slope system** for social‑signal research.

### Fixed
- Slope update jitter during high‑volatility periods.
- Memory growth inconsistencies during long sessions.

---

## [1.3.0] – 2026‑01‑30
### Added
- Narrator lines for warm‑up phases, cache saves, and slope updates.
- Startup summary block for Scarlet’s internal state.
- Corruption‑safe loader for emotional memory.

### Changed
- Improved emotional‑memory stability and clarity.

---

## [1.2.0] – 2026‑01‑15
### Added
- **Scarlet Proprietary License** and full licensing transition.
- Per‑file proprietary headers.
- Updated README and legal documentation.

### Changed
- Repository cleanup and restructuring for clarity.

---

## [1.1.0] – 2026‑01‑02
### Added
- Micro‑scalp mode (later deprecated for stability).
- Vectorized training improvements.
- Optional shaping‑signal enhancements.

### Fixed
- Training instability during rapid micro‑cycles.

---

## [1.0.0] – 2025‑12‑21
### Added
- First public research release of Scarlet.
- Emotional memory engine.
- Sentiment‑aware reasoning.
- Narratable decision‑making.
- Core API integration.

---

## [0.1.0] – 2025‑12‑06
### Added
- Early prototype of Scarlet’s architecture.
- Initial emotional‑memory scaffolding.
- First interpretable reasoning experiments.
