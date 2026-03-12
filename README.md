\# \*\*Scarlet — Reinforcement‑Learning Market Engine\*\*



*See ROADMAP.md for long term plans* A year in the making, \*\*Scarlet\*\* is a Python‑based reinforcement‑learning market analysis and automated‑trading engine designed for experimentation, offline training, and real‑time market evaluation. She includes a full market‑data engineering pipeline, a custom indicator system, GPU‑accelerated training, and a modular architecture built for research and future automated‑trading extensions.



Automated trading is \*\*disabled by default\*\* for safety.  

If you truly want to enable it, the toggle is located at the top of `scarlet\_core.py`.



Scarlet currently integrates with the \*\*Gemini\*\* crypto exchange for live data and execution.



---



\# \*\*Features\*\*



\### \*\*Reinforcement Learning\*\*

\- Offline RL training loop  

\- Policy + value networks (PyTorch)  

\- Replay buffer  

\- Custom loss functions and optimizers  

\- GPU‑accelerated training (CUDA 12.8, optimized for NVIDIA 5000‑series GPUs)



\### \*\*Market‑Data Engineering Pipeline\*\*

\- High‑resolution OHLCV ingestion  

\- Cleaning, alignment, resampling  

\- Windowing and batching  

\- Multi‑window feature generation for RL state construction  

\- Normalization and scaling utilities for stable training



\### \*\*Custom Indicator Engine\*\*

\- RSI  

\- MACD  

\- ATR  

\- Bollinger Bands  

\- VWAP  

\- Multi‑horizon slopes  

\- Volume  

\- Regime and sentiment hooks (extensible)



\### \*\*Architecture\*\*

\- Modular, research‑friendly design  

\- Clear separation between:

&nbsp; - Offline training  

&nbsp; - Live evaluation  

&nbsp; - Optional automated trading  

\- Narratable, interpretable RL posture system  

\- Built for long‑term experimentation and safe iteration



---



\# \*\*System Architecture\*\*



```

&nbsp;                         ┌──────────────────────────┐

&nbsp;                         │      Market Feeds        │

&nbsp;                         │  (Crypto, OHLCV, APIs)   │

&nbsp;                         └────────────┬─────────────┘

&nbsp;                                      │

&nbsp;                                      ▼

&nbsp;                       ┌──────────────────────────────┐

&nbsp;                       │   Data Engineering Pipeline   │

&nbsp;                       │  - Fetching / Cleaning        │

&nbsp;                       │  - Alignment / Resampling     │

&nbsp;                       │  - Windowing / Batching       │

&nbsp;                       └────────────┬──────────────────┘

&nbsp;                                    │

&nbsp;                                    ▼

&nbsp;                   ┌────────────────────────────────────┐

&nbsp;                   │      Indicator Engine (Custom)      │

&nbsp;                   │  RSI • MACD • ATR • Bollinger       │

&nbsp;                   │  VWAP • Slopes • Volume • Regimes   │

&nbsp;                   └──────────────────┬──────────────────┘

&nbsp;                                      │

&nbsp;                                      ▼

&nbsp;                   ┌────────────────────────────────────┐

&nbsp;                   │   Feature Constructor (Multi‑Win)   │

&nbsp;                   │  - State Assembly for RL Agent      │

&nbsp;                   │  - Normalization / Scaling          │

&nbsp;                   └──────────────────┬──────────────────┘

&nbsp;                                      │

&nbsp;                                      ▼

&nbsp;               ┌────────────────────────────────────────────┐

&nbsp;               │         Reinforcement‑Learning Core         │

&nbsp;               │  - Policy Network (PyTorch)                 │

&nbsp;               │  - Value Network                            │

&nbsp;               │  - Replay Buffer                            │

&nbsp;               │  - Loss Functions / Optimizers              │

&nbsp;               └──────────────────┬──────────────────────────┘

&nbsp;                                  │

&nbsp;                                  ▼

&nbsp;               ┌────────────────────────────────────────────┐

&nbsp;               │            Training Subsystem               │

&nbsp;               │  - GPU Acceleration (CUDA 12.8)             │

&nbsp;               │  - Offline Training Loop                    │

&nbsp;               │  - Checkpointing / Logging                  │

&nbsp;               └──────────────────┬──────────────────────────┘

&nbsp;                                  │

&nbsp;                                  ▼

&nbsp;               ┌────────────────────────────────────────────┐

&nbsp;               │           Live Evaluation Engine            │

&nbsp;               │  - Real‑Time Feature Updates                │

&nbsp;               │  - Policy Inference                         │

&nbsp;               │  - Narration / Logging                      │

&nbsp;               │  - Safety Layer (Auto‑Trading Off by Default) │

&nbsp;               └────────────────────────────────────────────┘

```



---



\# \*\*Environment Setup (`scarlet\_gemini`)\*\*



```

cd D:\\Scarlet\_Works\\Scarlet

pip install --pre torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

pip install --pre torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

pip install -r requirements.txt

```



---



\# \*\*Running Scarlet\*\*



\### \*\*Anaconda Prompt (recommended)\*\*



```

conda activate scarlet\_gemini

D:

cd D:\\Scarlet\_Works\\Scarlet

```



\*\*Offline Trainer\*\*



```

python Scarlet\_Trainer.py

```



\*\*Main Live Loop\*\*



```

python Scarlet\_Core.py

```



---



\### \*\*Running Scarlet in CMD\*\*



\*\*Trainer\*\*



```

call conda activate scarlet\_gemini

D:

cd D:\\Scarlet\_Works\\Scarlet

python Scarlet\_Trainer.py

```



\*\*Main Loop\*\*



```

call conda activate scarlet\_gemini

D:

cd D:\\Scarlet\_Works\\Scarlet

python Scarlet\_Core.py

```



---



\# \*\*Notes\*\*

\- Automated trading is disabled by default for safety.  

\- Scarlet currently spawns multiple loops on startup; all but one exit after the first `time.sleep(16)`. This is a known issue under investigation.



---



\# \*\*Why Scarlet Exists\*\*



Modern market‑analysis tools tend to fall into two extremes:



\- \*\*Black‑box trading bots\*\* that hide their logic, can’t be trusted, and often fail silently.  

\- \*\*Academic RL research code\*\* that is powerful but unusable for real markets without massive engineering effort.



There is almost nothing in the middle — a transparent, narratable, research‑grade RL engine that can run on real market data, explain its reasoning, and remain safe by design.



Scarlet fills that gap.



She was built to be:



\- A research platform for experimenting with reinforcement learning on real market structure  

\- A transparent system that narrates its decisions instead of hiding them  

\- A safe environment where automated trading is disabled by default  

\- A modular engine that can evolve into more advanced agents without rewriting the core  

\- A personal laboratory for exploring indicators, feature engineering, and RL architectures  



Scarlet is not a bot.  

She is a \*\*market‑reasoning engine\*\* — a place to study, test, and refine ideas safely.



Ultimately, the goal is to expand beyond crypto.  

We live too reactively; Scarlet is a step toward seeing one move ahead instead of staring at our current footprint.



---



\# \*\*Professional Setup Assistance\*\*



Scarlet is open‑source and free to use.  

If you need help installing, configuring, or extending Scarlet, I offer professional support at:



\### \*\*$150/hr\*\*



This includes:



\- Environment setup (CUDA, PyTorch, GPU tuning)  

\- Exchange integration  

\- Custom indicator design  

\- RL architecture tuning  

\- Automated‑trading safety review  

\- Debugging and performance optimization  



---





