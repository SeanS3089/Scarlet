import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import glob
import os

FEATURES_SINGLE_ASSET = [
    "open", "high", "low", "close", "volume",
    "rsi",
    "bb_mid", "bb_upper", "bb_lower", "bb_width",
    "macd", "macd_signal", "macd_hist",
    "vwap",
    "atr",
    "vol_roll_std",
    "slope_10", "slope_30", "slope_40",
]

def compute_features(df):
    df = df.copy()

    # returns + log returns
    df["ret"] = df["close"].pct_change()
    df["log_ret"] = np.log(df["close"]).diff()

    # volatility
    df["vol_roll_std"] = df["log_ret"].rolling(48).std()

    # ATR
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-12)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ⭐ INSERT THIS LINE ⭐
    df = compute_bollinger_bands(df)

    # MACD
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = macd - signal

    # VWAP
    typical = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (typical * df["volume"]).rolling(48).sum() / (df["volume"].rolling(48).sum() + 1e-12)

    # slopes
    df["slope_10"] = df["close"].diff(10)
    df["slope_30"] = df["close"].diff(30)
    df["slope_40"] = df["close"].diff(40)

    return df

def compute_bollinger_bands(df, window=20, num_std=2):
    df = df.copy()
    mid = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std()

    df["bb_mid"] = mid
    df["bb_upper"] = mid + num_std * std
    df["bb_lower"] = mid - num_std * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (mid + 1e-12)

    return df


def load_scarlet_scaler():
    # Load ALL raw CSVs for ALL assets and cadences
    files = glob.glob(r"C:\Scarlet_Works\Scarlet\data_providers\cache\*_15m.csv")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp")
        df = compute_features(df)
        dfs.append(df)

    # Concatenate all assets
    big = pd.concat(dfs, ignore_index=True)

    # Fit scaler
    scaler = RobustScaler().fit(big[FEATURES_SINGLE_ASSET].fillna(0.0))

    return scaler


