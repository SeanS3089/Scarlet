# Changelog
All notable changes to **Scarlet** are documented here.

Scarlet follows a narrative‑driven development cycle.  
Each version reflects architectural, interpretability, and emotional‑memory milestones.

This project uses **Semantic Versioning** and the **Keep a Changelog** format.

---

### [1.6.1] – 2026‑04‑10
### Added
-Full multi‑asset alignment across SOLUSD, ETHUSD, BTCUSD, and XRPUSD with strict schema guarantees.

-Unified multi‑horizon delta engine (8‑horizon, 2‑hour lookahead) with consistent cross‑asset indexing.

-GPU‑resident shaping tensors (price, ATR, VWAP, MACD, signal line, slope) for zero‑copy training.

-Pre‑tensorized offline dataset with full‑history slicing and instant GPU‑ready batches.

-Vectorized reward engine with fused shaping logic and multi‑asset ROI aggregation.

-High‑throughput training loop capable of saturating RTX 5080‑class GPUs at 70–85% utilization.

-Real‑time epoch execution (≈2 seconds per epoch) enabling rapid research iteration.

-Cross‑asset stability improvements for delta prediction, shaping consistency, and horizon alignment.

-Automatic best‑checkpoint tracking with negative‑reward optimization support.

### Fixed
-Incorrect XRP horizon alignment caused by mixed indexing in the old delta builder.

-CPU bottleneck in the offline dataloader caused by Pandas slicing, .fillna(), and per‑sample tensor creation.

-GPU starvation due to Python‑side padding and shaping dict construction inside __getitem__.

-Occasional reward‑shape mismatches between assets when slicing near sequence boundaries.

-Training loop stalls caused by CPU→GPU transfer overhead in shaping tensors.

-Validation instability from inconsistent shaping scales across assets.

### Changed
-Offline dataset now performs zero work in __getitem__ — all preprocessing is done once at load time.

-Reward scale normalized to realistic multi‑asset ROI magnitudes (loss now converges around −2e‑3 instead of e‑10).

-Training loop updated to support AMP, fused shaping, and GPU‑resident deltas.

-Collate function rewritten to perform GPU‑side padding for maximum throughput.

-Default offline training cadence updated to reflect new high‑speed pipeline.




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
