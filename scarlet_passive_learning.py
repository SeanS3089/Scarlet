

import pandas as pd
from decimal import Decimal
from dataclasses import replace
from typing import Dict, Any

from scarlet_config import ScarletPolicyConfig



def _safe_dec(x, default="0"):
    try:
        return Decimal(str(x))
    except Exception:
        return Decimal(default)


def load_memory(memory_log_path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(memory_log_path, on_bad_lines="skip")
        if df.empty:
            return None
        df.columns = [c.strip() for c in df.columns]
        return df
    except FileNotFoundError:
        return None
    except Exception:
        return None


def build_symbol_stats(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if df is None or df.empty:
        return {}

    if "symbol" not in df.columns:
        return {}

    stats: Dict[str, Dict[str, Any]] = {}

    for sym, sdf in df.groupby("symbol"):
        
        sdf = sdf.copy()

        if sdf.empty:
            continue

        # Filter to filled trades safely
        if "trade_filled" in sdf.columns:
            sdf = sdf.loc[sdf["trade_filled"].astype(str).str.upper() == "TRUE"].copy()

        if sdf.empty:
            continue

        if "profit" in sdf.columns:
            sdf.loc[:, "profit_dec"] = sdf["profit"].apply(_safe_dec)
        else:
            sdf.loc[:, "profit_dec"] = Decimal("0")

        wins = sdf.loc[sdf["profit_dec"] > 0]
        losses = sdf.loc[sdf["profit_dec"] < 0]

        
        if "policy_delta" in sdf.columns:
            sdf.loc[:, "policy_delta_dec"] = sdf["policy_delta"].apply(_safe_dec)
        else:
            sdf.loc[:, "policy_delta_dec"] = Decimal("0")

        
        if "actual_delta" in sdf.columns:
            sdf.loc[:, "actual_delta_dec"] = sdf["actual_delta"].apply(_safe_dec)
        else:
            sdf.loc[:, "actual_delta_dec"] = Decimal("0")

       
        if "volatility" in sdf.columns:
            sdf.loc[:, "vol_dec"] = sdf["volatility"].apply(_safe_dec)
        else:
            sdf.loc[:, "vol_dec"] = Decimal("0")

     
        n = len(sdf)
        stats[sym] = {
            "num_trades": n,
            "win_rate": float(len(wins) / n) if n > 0 else 0.0,
            "avg_profit": float(sum(sdf["profit_dec"]) / n) if n > 0 else 0.0,
            "avg_win": float(sum(wins["profit_dec"]) / len(wins)) if len(wins) > 0 else 0.0,
            "avg_loss": float(sum(losses["profit_dec"]) / len(losses)) if len(losses) > 0 else 0.0,
            "avg_policy_delta": float(sum(sdf["policy_delta_dec"]) / n) if n > 0 else 0.0,
            "avg_actual_delta": float(sum(sdf["actual_delta_dec"]) / n) if n > 0 else 0.0,
            "avg_volatility": float(sum(sdf["vol_dec"]) / n) if n > 0 else 0.0,
        }

    return stats

def update_policy_for_symbol(sym_stats, old_cfg: ScarletPolicyConfig, sym: str) -> ScarletPolicyConfig:
    if sym_stats["num_trades"] < 10:
        return old_cfg

    win_rate = Decimal(str(sym_stats["win_rate"]))
    avg_profit = Decimal(str(sym_stats["avg_profit"]))

    new_cfg = replace(old_cfg)

  
    old_thr = new_cfg.buy_delta_threshold[sym]
    if win_rate < Decimal("0.45"):
        new_thr = (old_thr * Decimal("1.10")).quantize(Decimal("0.0001"))
    elif win_rate > Decimal("0.60"):
        new_thr = (old_thr * Decimal("0.95")).quantize(Decimal("0.0001"))
    else:
        new_thr = old_thr
    new_cfg.buy_delta_threshold[sym] = new_thr

 
    old_ms = new_cfg.micro_scalp_factor[sym]
    if avg_profit < 0:
        new_ms = (old_ms * Decimal("0.80")).quantize(Decimal("0.0001"))
    else:
        bumped = (old_ms * Decimal("1.10")).quantize(Decimal("0.0001"))
        new_ms = min(bumped, Decimal("0.0200"))
    new_cfg.micro_scalp_factor[sym] = new_ms


    old_exp = new_cfg.max_exposure[sym]
    if avg_profit < 0:
        new_exp = (old_exp * Decimal("0.90")).quantize(Decimal("0.0001"))
    else:
        new_exp = old_exp
    new_cfg.max_exposure[sym] = new_exp

    return new_cfg


def narrate_updates(narrator, old_cfg: ScarletPolicyConfig, new_cfg: ScarletPolicyConfig, stats: Dict[str, Dict[str, Any]]) -> None:
    if old_cfg == new_cfg:
        narrator.narrate("📊 Passive learning → no changes (insufficient data or stable performance).")
        return

    narrator.narrate("📚 Passive learning cycle triggered — updating behavioral configuration.")

    for sym, s in stats.items():
        narrator.narrate(
            f"🧾 [{sym.upper()}] win_rate={s['win_rate']:.2f}, avg_profit={s['avg_profit']:.5f}"
        )

        if old_cfg.buy_delta_threshold[sym] != new_cfg.buy_delta_threshold[sym]:
            narrator.narrate(
                f"🎚️ [{sym.upper()}] Δ-threshold: {old_cfg.buy_delta_threshold[sym]} → {new_cfg.buy_delta_threshold[sym]}"
            )

        if old_cfg.micro_scalp_factor[sym] != new_cfg.micro_scalp_factor[sym]:
            narrator.narrate(
                f"🎚️ [{sym.upper()}] micro‑scalp: {old_cfg.micro_scalp_factor[sym]} → {new_cfg.micro_scalp_factor[sym]}"
            )

        if old_cfg.max_exposure[sym] != new_cfg.max_exposure[sym]:
            narrator.narrate(
                f"🛡️ [{sym.upper()}] exposure: {old_cfg.max_exposure[sym]} → {new_cfg.max_exposure[sym]}"
            )


def run_passive_learning(memory_log_path: str, config: ScarletPolicyConfig, narrator) -> ScarletPolicyConfig:
    df = load_memory(memory_log_path)
    if df is None or df.empty:
        narrator.narrate("📭 Passive learning → no memory yet, skipping.")
        return config

    stats = build_symbol_stats(df)

    new_cfg = config
    for sym, sym_stats in stats.items():
        new_cfg = update_policy_for_symbol(sym_stats, new_cfg, sym)

    narrate_updates(narrator, config, new_cfg, stats)
    return new_cfg