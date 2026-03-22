

Scarlet — Reinforcement‑Learning Market Engine
A year in the making, Scarlet is a Python‑based reinforcement‑learning market analysis and automated‑trading engine designed for experimentation, offline training, and real‑time market evaluation. She includes a full market‑data engineering pipeline, a custom indicator system, GPU‑accelerated training, and a modular architecture built for research and future automated‑trading extensions.
Automated trading is disabled by default for safety.
If you truly want to enable it, the toggle is located at the top of scarlet_core.py.
Scarlet currently integrates with the Gemini crypto exchange for live data and execution.

Features
Reinforcement Learning
- Offline RL training loop
- Policy + value networks (PyTorch)
- Replay buffer
- Custom loss functions and optimizers
- GPU‑accelerated training (CUDA 12.8, optimized for NVIDIA 5000‑series GPUs)
- Multi‑horizon forecasting (6‑step delta curve)
- Horizon‑aware policy collapse
Scarlet collapses her 6‑step forecast vector using a weighted blend centered on the correct horizon for her real‑world trading cadence (currently ~60 minutes).
This preserves multi‑horizon structure while aligning decisions with timing and fee constraints.
Market‑Data Engineering Pipeline
- High‑resolution OHLCV ingestion
- Cleaning, alignment, resampling
- Windowing and batching
- Multi‑window feature generation for RL state construction
- Normalization and scaling utilities for stable training
Custom Indicator Engine
- RSI
- MACD
- ATR
- Bollinger Bands
- VWAP
- Multi‑horizon slopes
- Volume
- Regime and sentiment hooks (extensible)
Architecture
- Modular, research‑friendly design
- Clear separation between:
- Offline training
- Live evaluation
- Optional automated trading
- Narratable, interpretable RL posture system
- Built for long‑term experimentation and safe iteration
- Horizon‑aligned drift detection
Drift compares the horizon‑collapsed forecast to the realized delta over the same time window, ensuring stable online learning and reducing false positives.
- Fee‑aware decision logic
Policy thresholds adapt to the current trading‑fee environment, preventing over‑trading while preserving sensitivity to meaningful signals.

                         ┌──────────────────────────┐
                         │      Market Feeds        │
                         │  (Crypto, OHLCV, APIs)   │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │   Data Engineering Pipeline   │
                       │  - Fetching / Cleaning        │
                       │  - Alignment / Resampling     │
                       │  - Windowing / Batching       │
                       └────────────┬──────────────────┘
                                    │
                                    ▼
                   ┌────────────────────────────────────┐
                   │      Indicator Engine (Custom)      │
                   │  RSI • MACD • ATR • Bollinger       │
                   │  VWAP • Slopes • Volume • Regimes   │
                   └──────────────────┬──────────────────┘
                                      │
                                      ▼
                   ┌────────────────────────────────────┐
                   │   Feature Constructor (Multi‑Win)   │
                   │  - State Assembly for RL Agent      │
                   │  - Normalization / Scaling          │
                   └──────────────────┬──────────────────┘
                                      │
                                      ▼
               ┌────────────────────────────────────────────┐
               │         Reinforcement‑Learning Core         │
               │  - Policy Network (PyTorch)                 │
               │  - Value Network                            │
               │  - Replay Buffer                            │
               │  - Loss Functions / Optimizers              │
               └──────────────────┬──────────────────────────┘
                                  │
                                  ▼
               ┌────────────────────────────────────────────┐
               │            Training Subsystem               │
               │  - GPU Acceleration (CUDA 12.8)             │
               │  - Offline Training Loop                    │
               │  - Checkpointing / Logging                  │
               └──────────────────┬──────────────────────────┘
                                  │
                                  ▼
               ┌────────────────────────────────────────────┐
               │           Live Evaluation Engine            │
               │  - Real‑Time Feature Updates                │
               │  - Policy Inference                         │
               │  - Horizon‑Aware Forecast Collapse          │
               │  - Drift Detection (Horizon‑Aligned)        │
               │  - Narration / Logging                      │
               │  - Safety Layer (Auto‑Trading Off by Default) │
               └────────────────────────────────────────────┘

  Notes
- Automated trading is disabled by default for safety.
- Scarlet now uses a weighted horizon‑4 collapse for policy and drift, aligned with her 61‑minute cadence and the current trading‑fee environment.

Why Scarlet Exists
Modern market‑analysis tools tend to fall into two extremes:
- Black‑box trading bots that hide their logic, can’t be trusted, and often fail silently.
- Academic RL research code that is powerful but unusable for real markets without massive engineering effort.
There is almost nothing in the middle — a transparent, narratable, research‑grade RL engine that can run on real market data, explain its reasoning, and remain safe by design.
Scarlet fills that gap.
She was built to be:
- A research platform for experimenting with reinforcement learning on real market structure
- A transparent system that narrates its decisions instead of hiding them
- A safe environment where automated trading is disabled by default
- A modular engine that can evolve into more advanced agents without rewriting the core
- A personal laboratory for exploring indicators, feature engineering, and RL architectures
Scarlet is not a bot.
She is a market‑reasoning engine — a place to study, test, and refine ideas safely.
Ultimately, the goal is to expand beyond crypto.
We live too reactively; Scarlet is a step toward seeing one move ahead instead of staring at our current footprint.

Professional Setup Assistance
Scarlet is expensive. Inquire for pricing.
If you need help installing, configuring, or extending Scarlet, I offer professional support at:
$250/hr
This includes:
- Environment setup (CUDA, PyTorch, GPU tuning)
- Exchange integration
- Custom indicator design
- RL architecture tuning
- Automated‑trading safety review
- Debugging and performance optimization
