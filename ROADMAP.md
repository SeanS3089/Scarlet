---



\# \*\*Roadmap\*\*



Scarlet’s development follows a long‑horizon research arc. Each phase builds on the last, gradually transforming Scarlet from a crypto‑focused RL engine into a general market‑reasoning system capable of anticipating structure, sentiment, and regime shifts.



---



\## \*\*Phase 1 — Long‑Horizon Data Accumulation (5–12 months)\*\*  

Scarlet needs a deep, continuous history of real market structure before she can meaningfully learn.



Goals:  

\- Accumulate 5–12 months of high‑resolution crypto OHLCV  

\- Maintain uninterrupted data streams (Gemini now, more exchanges later)  

\- Clean, align, and resample data into a stable training corpus  

\- Capture volatility regimes, structural breaks, and seasonal patterns  

\- Build a rolling dataset large enough for RL generalization  



This is the foundation. Without it, no RL agent can stabilize.



---



\## \*\*Phase 2 — Improve Crypto Prediction Accuracy\*\*  

Once Scarlet has enough historical context, the next milestone is \*\*predictive stability\*\*.



Goals:  

\- Improve short‑horizon return prediction accuracy  

\- Reduce drift between offline training and live inference  

\- Strengthen multi‑window feature engineering  

\- Expand indicator and sentiment hooks  

\- Stabilize the RL policy under real‑world noise  

\- Validate predictions across multiple assets and volatility regimes  



This phase ends when Scarlet can consistently produce \*\*meaningful, non‑random predictive deltas\*\*.



---



\## \*\*Phase 3 — Reverse‑Engineer Sentiment from Market Predictions\*\*  

Once Scarlet can reliably predict price movement, the next step is to understand \*\*why\*\*.



Goals:  

\- Compare predicted deltas vs. real‑time sentiment  

\- Identify patterns where sentiment leads or lags price  

\- Build a mapping between market structure and emotional regimes  

\- Train models to infer sentiment shifts from price‑action alone  

\- Use RL state embeddings to detect latent “mood” vectors  



This is where Scarlet begins to \*reason\* about markets, not just react to them.



---



\## \*\*Phase 4 — Predict the Events That Trigger Sentiment Shifts\*\*  

With sentiment inference in place, Scarlet can begin learning \*\*what causes\*\* sentiment to change.



Goals:  

\- Add datasets that correlate with sentiment transitions:

&nbsp; - volatility spikes  

&nbsp; - liquidity shocks  

&nbsp; - funding‑rate changes  

&nbsp; - macro events  

&nbsp; - cross‑asset correlations  

\- Train models to anticipate sentiment regime shifts before they occur  

\- Build a classifier for “event type → expected sentiment response”  

\- Integrate event‑anticipation into the RL state  



This phase moves Scarlet toward \*anticipatory reasoning\* — seeing one step ahead.



---



\## \*\*Phase 5 — Multi‑Asset Expansion\*\*  

After crypto stabilization and sentiment reasoning:



Goals:  

\- Add equities, FX, commodities  

\- Add cross‑asset features (correlation, spreads, volatility clusters)  

\- Expand the RL state to include global market context  

\- Validate generalization across asset classes  



Scarlet evolves from a crypto engine into a \*\*general market‑reasoning system\*\*.



---



\## \*\*Phase 6 — Advanced RL Architectures\*\*  

With stable predictions and multi‑asset data:



Goals:  

\- Multi‑step forecasting  

\- Transformer‑based RL policies  

\- Regime‑aware agents  

\- Hierarchical RL (macro → micro decisions)  

\- Meta‑learning for rapid adaptation  



This is where Scarlet becomes a true research‑grade RL platform.



---



\## \*\*Phase 7 — Safe Automation Layer\*\*  

Only after prediction accuracy and stability are proven:



Goals:  

\- Expand the safety layer  

\- Add guardrails for execution  

\- Add anomaly detection  

\- Add kill‑switch logic  

\- Add human‑in‑the‑loop confirmation modes  



Automation remains \*\*opt‑in\*\* and \*\*heavily restricted\*\*.



---



\## \*\*Long‑Horizon Vision\*\*  

Scarlet’s ultimate purpose:



> \*We live too reactively. Scarlet is a step toward seeing one move ahead instead of staring at our current footprint.\*



Long‑term goals:  

\- Multi‑day and multi‑week forecasting  

\- Event anticipation  

\- Market‑structure inference  

\- Cross‑asset reasoning  

\- Narrative‑driven explanations of decisions  



Scarlet becomes not just a model, but a \*\*market‑reasoning companion\*\*.



---





