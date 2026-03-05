from decimal import Decimal


def evaluate_tactical_flags(rsi, forecast, previous_forecast, volatility, recent_slope):
    """
    Evaluates tactical flags for override logic.
    Returns:
        flags (dict): Boolean flags for each tactical condition.
        notes (dict): Narratable commentary for each flag.
    """
    flags = {}
    notes = {}
    threshold = Decimal("0.05")       
    slope_threshold = Decimal("0.02")  
    

    rsi_bounds = {
    "floor": Decimal("30.0"),
    "ceiling": Decimal("70.0")
    }


    if rsi is None:
        flags['rsi_neutral'] = True
        notes['rsi_neutral'] = "RSI unavailable — fallback to neutral override."
    else:
        flags['rsi_overbought'] = rsi > rsi_bounds["ceiling"]
        flags['rsi_oversold'] = rsi < rsi_bounds["floor"]
        notes['rsi_overbought'] = f"RSI={rsi:.2f} exceeds upper bound ({rsi_bounds['ceiling']})."
        notes['rsi_oversold'] = f"RSI={rsi:.2f} below lower bound ({rsi_bounds['floor']})."

 
    if forecast is not None and previous_forecast is not None:
        delta = forecast - previous_forecast
        flags['momentum_positive'] = delta > threshold
        flags['momentum_negative'] = delta < -threshold
        notes['momentum_positive'] = f"Forecast momentum rising: Δ={delta:.5f} > {threshold}"
        notes['momentum_negative'] = f"Forecast momentum falling: Δ={delta:.5f} < -{threshold}"
    else:
        flags['momentum_neutral'] = True
        notes['momentum_neutral'] = "Forecast delta unavailable — momentum neutral."

 
    if volatility is not None:
        flags['volatility_high'] = volatility > 0.5
        flags['volatility_low'] = volatility < 0.1
        notes['volatility_high'] = f"Volatility={volatility:.3f} exceeds high threshold."
        notes['volatility_low'] = f"Volatility={volatility:.3f} below low threshold."

 
    if recent_slope is not None:
        flags['slope_positive'] = recent_slope > slope_threshold
        flags['slope_negative'] = recent_slope < -slope_threshold
        notes['slope_positive'] = f"Slope rising: {recent_slope:.5f} > {slope_threshold}"
        notes['slope_negative'] = f"Slope falling: {recent_slope:.5f} < -{slope_threshold}"

    return flags, notes

def compute_recent_slope(candles_df, window=5):
    """
    Compute slope of recent price movement using linear regression.
    Returns a float representing the slope (positive = upward momentum).
    """
    import numpy as np

    try:
        if candles_df is None or len(candles_df) < window:
            print(f"⚠️ Slope fallback — insufficient candle data (need {window}, got {len(candles_df)})")
            return 0.0

        recent_prices = candles_df["close"].tail(window).values
        x = np.arange(len(recent_prices))
        slope, _ = np.polyfit(x, recent_prices, 1)

        print(f"📈 Recent slope over {window} candles: {slope:.5f}")
        return float(slope)

    except Exception as e:
        print(f"⛔ Slope calculation failed: {e}")
        return 0.0

def compute_recent_profit(csv_path, window=14):
    import pandas as pd
    from decimal import Decimal

    df = pd.read_csv(csv_path)
    if "profit" not in df.columns:
        raise ValueError("Missing 'profit' column in CSV.")

    recent_profits = df["profit"].tail(window).fillna(0)
    total = sum(Decimal(str(p)) for p in recent_profits)
    print(f"📊 Recent profit total: {total}")
    return total
