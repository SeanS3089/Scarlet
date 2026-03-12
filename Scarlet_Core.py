
from ast import Return
import torch
import torch.nn as nn
from hotkeys import setup_hotkeys, forced_signal
import warnings 
DISABLE_TRADING = True  # I really reccomend watching Scarlet run before enabling trading
use_cpu_for_online = True  # I set up Scarlet to run online training on CPU as I like gaming too, this frees up the GPU
# Run a search for:   horizon_weights = Adjust for trading style
# Search for:        buffer_usd = Decimal("40.50")  TO CHANGE WALLET RESERVE

import sys, os, msvcrt
try:
    msvcrt.setmode(sys.stdout.fileno(), os.O_TEXT)
except Exception as e:
    sys.stderr.write(f"stdout reattach failed: {e}\n")

sys.stdout.flush()
last_regime = {
    "solusd": None,
    "ethusd": None,
    "btcusd": None,
}

LSTM_WINDOW = 256
import copy

from operator import iadd
from re import I
from tabnanny import verbose
import time

memory_log_path = r"D:\Scarlet_Works\Scarlet\scarlet_memory.csv"
memory_path = r"D:\Scarlet_Works\Scarlet\scarlet_memory.csv"

from dotenv import load_dotenv
load_dotenv()

epoch = 0
total_loss = None

from decimal import Decimal, InvalidOperation
import math

import json
from scarlet_passive_learning import run_passive_learning

import threading
from collections import deque
manual_training_requested = False

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import praw

from scarlet_config import ScarletPolicyConfig

# -1% loss
POLICY_CONFIG = ScarletPolicyConfig()
policy_config = POLICY_CONFIG
print("Policy config live:", policy_config.buy_delta_threshold,
      policy_config.sell_delta_threshold,
      policy_config.hold_zone)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hard_stop_loss = False

from torch.utils.tensorboard import SummaryWriter

# === Micro-Scalp Mode (Aggressive) ===
AGGRESSIVE_MICRO_SCALP_ENABLED = True

MICRO_SCALP_DUST_SOL = 0.000005
MICRO_SCALP_MIN_DELTA = 0.001
MICRO_SCALP_MAX_ATR = 0.35
MICRO_SCALP_SIZE_FACTOR = 0.01

MICRO_SCALP_TP = 0.0022
MICRO_SCALP_SL = -0.0015
MICRO_SCALP_MAX_ATR_EXIT = 0.45
MICRO_SCALP_MIN_DELTA_EXIT = -0.0005

cooldown_cycles = {
    "solusd": 0,
    "ethusd": 0,
    "btcusd": 0,
}

sys.path.append(r"D:\Scarlet_Works\Scarlet")
sys.path.append(r"D:\Scarlet_Works\Scarlet")
CACHE_DIR = r"D:\Scarlet_Works\Scarlet"

import joblib
import traceback
from typing import Tuple

from utils_io.candle import CandleDatasetV2

from gemini_time_utils import SmartNonceManager
pd.set_option("future.no_silent_downcasting", True)
prev_market_price = None

import numpy as np
np.random.seed(42)

import inspect

from cryptography.hazmat.primitives import serialization

import collections

from datetime import datetime

nonce_mgr = SmartNonceManager()

warnings.filterwarnings("ignore", message="This API is going to be deprecated", module="torch")

from Scarlet_Tactics import evaluate_tactical_flags
from Scarlet_Tactics import compute_recent_slope
from Scarlet_Tactics import compute_recent_profit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



INPUT_FEATURES = [

    "open_solusd", "high_solusd", "low_solusd", "close_solusd", "volume_solusd",


    "RSI_solusd",
    "BB_mid_solusd", "BB_upper_solusd", "BB_lower_solusd", "BB_width_solusd",
    "macd_line_solusd",      
    "signal_line_solusd",      
    "MACD_hist_solusd",
    "VWAP_solusd",
    "ATR_solusd",
    "volatility_solusd",

    "slope_10_solusd", "slope_30_solusd", "slope_40_solusd",


    "reddit_sentiment_solusd",
    "sentiment_slope_10_solusd",
    "sentiment_slope_25_solusd",
    "sentiment_slope_40_solusd",
    "sentiment_slope_40L_solusd",
    "sentiment_slope_120_solusd",
    "sentiment_slope_250_solusd",

    "open_ethusd", "high_ethusd", "low_ethusd", "close_ethusd", "volume_ethusd",

 
    "RSI_ethusd",
    "BB_mid_ethusd", "BB_upper_ethusd", "BB_lower_ethusd", "BB_width_ethusd",
    "macd_line_ethusd",         
    "signal_line_ethusd",       
    "MACD_hist_ethusd",
    "VWAP_ethusd",
    "ATR_ethusd",
    "volatility_ethusd",

   
    "slope_10_ethusd", "slope_30_ethusd", "slope_40_ethusd",

 
    "reddit_sentiment_ethusd",
    "sentiment_slope_10_ethusd",
    "sentiment_slope_25_ethusd",
    "sentiment_slope_40_ethusd",
    "sentiment_slope_40L_ethusd",
    "sentiment_slope_120_ethusd",
    "sentiment_slope_250_ethusd",


    "open_btcusd", "high_btcusd", "low_btcusd", "close_btcusd", "volume_btcusd",


    "RSI_btcusd",
    "BB_mid_btcusd", "BB_upper_btcusd", "BB_lower_btcusd", "BB_width_btcusd",
    "macd_line_btcusd",        
    "signal_line_btcusd",      
    "MACD_hist_btcusd",
    "VWAP_btcusd",
    "ATR_btcusd",
    "volatility_btcusd",

 
    "slope_10_btcusd", "slope_30_btcusd", "slope_40_btcusd",

   
    "reddit_sentiment_btcusd",
    "sentiment_slope_10_btcusd",
    "sentiment_slope_25_btcusd",
    "sentiment_slope_40_btcusd",
    "sentiment_slope_40L_btcusd",
    "sentiment_slope_120_btcusd",
    "sentiment_slope_250_btcusd",
]



import json
import base64
import time
import hmac
import hashlib
import requests
import os

from dotenv import load_dotenv
load_dotenv("D:\Scarlet_Works\Scarlet\.env")  # Or set system environmental variables

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_API_SECRET = os.environ["GEMINI_API_SECRET"].encode()
API_URL = "https://api.gemini.com"
GEMINI_API_URL = "https://api.gemini.com"

trade_nonce_mgr = SmartNonceManager(nonce_file="trade_nonce.txt")
balance_nonce_mgr = SmartNonceManager(nonce_file="balance_nonce.txt")

api_key = os.environ["GEMINI_API_KEY"]
api_secret = os.environ["GEMINI_API_SECRET"].encode()
key_name = os.environ["GEMINI_API_KEY"]
key_secret = os.environ["GEMINI_API_SECRET"].encode()

filepath = r"D:\Scarlet_Works\Scarlet\scarlet_memory.csv"

base_amount = 0.00000000001
max_amount = 0.25
profit_sensitivity = 40.0
rsi_bounds = {
    "floor": Decimal("30.0"),
    "ceiling": Decimal("70.0")
}

import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import pandas as pd

price_history = []

from collections import deque
import math

price_history = deque(maxlen=100)



import pandas as pd

def get_entry_info(df_or_path, symbol=None, narrator=None):
 

    import pandas as pd
    from decimal import Decimal


    if not isinstance(df_or_path, pd.DataFrame):
        try:
            df = pd.read_csv(df_or_path, on_bad_lines="skip")
        except Exception:
            return {
                "avg_entry_price": None,
                "exposure": None,
                "valid_anchor": False,
            }
    else:
        df = df_or_path.copy()


    if df.empty or "action" not in df.columns:
        return {
            "avg_entry_price": None,
            "exposure": None,
            "valid_anchor": False,
        }


    df.columns = [c.strip().lower() for c in df.columns]
    df["action"] = df["action"].astype(str).str.upper().str.strip()


    if symbol is not None and "symbol" in df.columns:
        df = df[df["symbol"].astype(str).str.lower() == symbol.lower()]

    if df.empty:
        return {
            "avg_entry_price": None,
            "exposure": None,
            "valid_anchor": False,
        }


    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)


    df["trade_filled"] = (
        df.get("trade_filled", "0")
        .astype(str)
        .str.strip()
        .str.lower()
    )

 
    valid = df[df["trade_filled"].isin(["1", "1.0", "true", "yes"])].copy()

    if valid.empty:
        return {
            "avg_entry_price": None,
            "exposure": None,
            "valid_anchor": False,
        }

 
    if "exit_note" in df.columns:
        reset_mask = (
            df["action"].str.contains("RESET", case=False, na=False)
            | df["exit_note"].astype(str).str.contains("reset", case=False, na=False)
        )
    else:
        reset_mask = df["action"].str.contains("RESET", case=False, na=False)

    last_reset_ts = None
    if reset_mask.any() and "timestamp" in df.columns:
        last_reset_ts = df.loc[reset_mask, "timestamp"].max()

    buys = valid[valid["action"] == "BUY"].copy()
    if last_reset_ts is not None and "timestamp" in buys.columns:
        buys = buys[buys["timestamp"] > last_reset_ts]


    sells = valid[valid["action"] == "SELL"].copy()
    if last_reset_ts is not None and "timestamp" in sells.columns:
        sells = sells[sells["timestamp"] > last_reset_ts]


    buy_amt = buys["amount"].astype(float).sum() if not buys.empty else 0.0
    sell_amt = sells["amount"].astype(float).sum() if not sells.empty else 0.0
    exposure = Decimal(str(buy_amt - sell_amt))


    if buy_amt > 0 and not buys.empty:
        weighted_sum = (
            buys["price"].astype(float) * buys["amount"].astype(float)
        ).sum()
        avg_entry = weighted_sum / buy_amt
    else:
        avg_entry = None


    dust_threshold = POLICY_CONFIG.dust_thresholds.get(
        symbol.lower(),
        POLICY_CONFIG.dust_thresholds["solusd"]  
    )


    valid_anchor = exposure > dust_threshold

    return {
        "avg_entry_price": avg_entry,
        "exposure": float(exposure),
        "valid_anchor": valid_anchor,
    }




from datetime import datetime
from decimal import Decimal



def should_trigger_buy(sol_balance_usd, usd_balance, forecast_delta, narrator=None, threshold=Decimal("0.01")):
    """
    Determines whether Scarlet should initiate a BUY based on:
    - wallet posture (flat/dust)
    - bullish forecast delta
    - USD availability

    forecast_delta MUST be fractional (e.g., 0.0123 for +1.23%).
    """


    flat = sol_balance_usd < 0.50
    dust = sol_balance_usd < 2.00


    bullish = forecast_delta > threshold


    usd_ok = usd_balance > 5  


    if flat or dust:
        if bullish and usd_ok:
            if narrator:
                narrator.narrate(
                    f"🟢 BUY trigger → flat/dust posture + bullish Δ={forecast_delta:.4f}"
                )
            return True, None

        reason = "forecast_too_weak" if not bullish else "insufficient_usd"
        if narrator:
            narrator.narrate(
                f"⚠️ BUY blocked → {reason} (Δ={forecast_delta:.4f}, usd={usd_balance:.2f})"
            )
        return False, reason

 
    if narrator:
        narrator.narrate("⏸ BUY suppressed → posture not flat/dust.")
    return False, "posture_not_flat"



def get_gemini_balances():
    path = "/v1/balances"
    url = API_URL + path
    balance_nonce_mgr = SmartNonceManager(nonce_file="balance_nonce.txt")

    for nonce in balance_nonce_mgr.get_safe_nonce_with_fallback(max_retries=3, delay_sec=12):
        payload = {
            "request": path,
            "nonce": str(nonce)
        }

        encoded_payload = base64.b64encode(json.dumps(payload).encode())
        signature = hmac.new(GEMINI_API_SECRET, encoded_payload, hashlib.sha384).hexdigest()

        if not GEMINI_API_KEY or not GEMINI_API_SECRET:
            print("❌ API credentials not found in environment.")
            return {}

        headers = {
            "X-GEMINI-APIKEY": GEMINI_API_KEY,
            "X-GEMINI-PAYLOAD": encoded_payload.decode(),
            "X-GEMINI-SIGNATURE": signature,
            "Content-Type": "text/plain"
        }

        try:
            response = requests.post(url, headers=headers, timeout=5)
        except Exception as e:
            print(f"❌ Network error fetching balances: {e}")
            continue

        try:
            if response.status_code == 200:
                balance_nonce_mgr._save_nonce(int(nonce))
                result = response.json()

                wallet = {}

                for entry in result:
                    currency = entry.get("currency")
                    if not currency:
                        continue

                    raw_available = entry.get("available")
                    raw_amount = entry.get("amount")

                    value_str = None

                    
                    if raw_available not in (None, "", "None"):
                        try:
                            Decimal(str(raw_available))
                            value_str = raw_available
                        except Exception:
                            pass

                    if value_str is None and raw_amount not in (None, "", "None"):
                        try:
                            Decimal(str(raw_amount))
                            value_str = raw_amount
                        except Exception:
                            pass

                    if value_str is None:
                        continue

                    try:
                        wallet[currency.upper()] = Decimal(str(value_str)).normalize()
                    except Exception:
                        continue

                return wallet or {}

            elif "InvalidNonce" in response.text:
                print("⚠️ Invalid nonce. Retrying...\n")
                continue

            else:
                print(f"❌ Gemini error: {response.status_code} - {response.text}")
                continue

        except json.JSONDecodeError:
            print("❌ Could not parse JSON response:", response.text)
            continue

    print("🚫 All balance attempts failed.")
    return {}


if not hasattr(collections, 'Mapping'):
    collections.Mapping = collections.abc.Mapping
if not hasattr(collections, 'MutableMapping'):
    collections.MutableMapping = collections.abc.MutableMapping
if not hasattr(collections, 'Callable'):
    collections.Callable = collections.abc.Callable
if not hasattr(collections, 'Sequence'):
    collections.Sequence = collections.abc.Sequence


import sys
import six
import importlib


_ = six.moves
sys.modules["six.moves"] = six.moves

try:
    urllib_module = importlib.import_module("six.moves.urllib")
    sys.modules["six.moves.urllib"] = urllib_module

    urllib_parse_module = importlib.import_module("six.moves.urllib.parse")
    sys.modules["six.moves.urllib.parse"] = urllib_parse_module
    
except ImportError as e:
   
    import urllib.parse
    sys.modules["six.moves.urllib.parse"] = urllib.parse
  





try:
    from six.moves.urllib.parse import urlparse
    print("urlparse works:", urlparse("http://example.com"))
except Exception as e:
    print("Error importing urlparse after injection:", e)


from pytorch_lightning.callbacks import Callback



import torch
torch.set_float32_matmul_precision("high")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import schedule
import time
import json
from sklearn.preprocessing import RobustScaler
import ta
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

import os
from datetime import datetime
from hybrid_forecaster import HybridForecasterD
from utils_io import fetch_order_book, update_crypto_data_file
import base64

order_book = fetch_order_book(symbol="solusd")

from dotenv import load_dotenv
from utils_io import execute_trade_gemini

import secrets
import logging
logging.getLogger("utils_io").setLevel(logging.WARNING)

import requests
import hmac
import hashlib
import uuid

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)


from utils_io import create_orderbook_dataframe

from cryptography.hazmat.primitives import serialization

import threading

class Narrator:
    def __init__(self, log_path="scarlet_narration_log.csv"):
        self.log_path = log_path
        self.lock = threading.Lock()

    def narrate(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

       
        with self.lock:
            try:
                with open(self.log_path, "a", encoding="utf-8") as log:
                    log.write(f"{timestamp} | {message}\n")
            except Exception as e:
                print(f"[Narrator ERROR] {e}")

        print(f"{timestamp} | {message}")


narrator = Narrator(log_path="scarlet_narration_log.csv")




CSV_PATH = r"D:\Scarlet_Works\Scarlet\crypto_data.csv"
MAX_ROWS = 50000

INPUT_SEQ_LEN =75 # Do not change
OUTPUT_SEQ_LEN =12  # Do not change
if not os.path.isfile(CSV_PATH):
    raise FileNotFoundError(f"{CSV_PATH} not found")







import os
from cryptography.hazmat.primitives import serialization

memory_path= r"D:\Scarlet_Works\Scarlet\scarlet_memory.csv" #Change to memory path you use









def get_wallet_snapshot(nonce_mgr=None):
    endpoint = "/v1/balances"
    url = API_URL + endpoint
    nonce = next(nonce_mgr.get_safe_nonce_with_fallback()) if nonce_mgr else int(time.time() * 1000)

    payload = {
        "request": endpoint,
        "nonce": str(nonce),
        "account": "primary"
    }

    clean_payload = json.dumps(payload, separators=(",", ":"))
    b64 = base64.b64encode(clean_payload.encode())
    signature = hmac.new(GEMINI_API_SECRET, b64, hashlib.sha384).hexdigest()

    headers = {
        "X-GEMINI-APIKEY": GEMINI_API_KEY,
        "X-GEMINI-PAYLOAD": b64.decode(),
        "X-GEMINI-SIGNATURE": signature,
        "Content-Type": "text/plain",
    }

    try:
        response = requests.post(url, headers=headers, timeout=5)

        if response.status_code != 200:
            print(f"❌ Gemini returned {response.status_code}: {response.text}")
            return {}

        result = response.json()

        if nonce_mgr:
            nonce_mgr._save_nonce(nonce)

        wallet = {}

 
        for entry in result:
            currency = entry.get("currency")
            if not currency:
                continue

            raw_available = entry.get("available")
            raw_amount = entry.get("amount")

       
            value_str = None

            if raw_available not in (None, "", "None"):
                try:
                    Decimal(str(raw_available))
                    value_str = raw_available
                except Exception:
                    pass

            if value_str is None and raw_amount not in (None, "", "None"):
                try:
                    Decimal(str(raw_amount))
                    value_str = raw_amount
                except Exception:
                    pass

            if value_str is None:
                continue

            try:
                wallet[currency.upper()] = Decimal(str(value_str)).normalize()
            except Exception:
                continue

        return wallet

    except Exception as e:
        print(f"🚨 Balance fetch exception: {e}")
        return {}






import utils_io.candle





from decimal import Decimal

def resolve_market_price(quote, fallback_close=None, field="last"):
    """
    Normalize Gemini pubticker, history list, or raw numeric into a Decimal market price.
    - quote: dict from pubticker, list of closes, or numeric/str price
    - fallback_close: optional close price from merged_df
    - field: which field to anchor to ("last", "ask", "bid")
    """

    if isinstance(quote, (int, float, Decimal, str)):
        try:
            return Decimal(str(quote))
        except Exception:
            pass

    if isinstance(quote, list) and quote:
        try:
            return Decimal(str(quote[-1])) 
        except Exception:
            pass


    if isinstance(quote, dict):
        val = (
            quote.get(field)
            or quote.get("last")
            or quote.get("ask")
            or quote.get("bid")
        )
        if val is not None:
            try:
                return Decimal(str(val))
            except Exception:
                pass


    if fallback_close is not None:
        try:
            return Decimal(str(fallback_close))
        except Exception:
            pass

    return None







import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime




def get_market_price(symbol="solusd", return_history=False, field="last"):
    """
    Fetches the latest market price from Gemini.
    - Returns Decimal or None.
    - If return_history=True → returns last 30 closes as floats.
    """


    if return_history:
        candles_df = load_market_data(symbol)
        if candles_df.empty:
            print("📉 No historical data retrieved — skipping Bollinger diagnostic.")
            return []
        return candles_df["close"].tail(30).astype(float).tolist()


    url = f"https://api.gemini.com/v1/pubticker/{symbol}"

    try:
        response = requests.get(url, timeout=5)
    except Exception as e:
        print(f"❌ Network error fetching market price: {e}")
        return None

    if response.status_code != 200:
        print(f"❌ Failed to fetch market price: HTTP {response.status_code}")
        return None

    try:
        data = response.json()
    except Exception:
        print("❌ Invalid JSON from Gemini.")
        return None


    candidates = [
        field,
        "last",
        "close",
        "price",
    ]

    for key in candidates:
        if key in data:
            val = data[key]
            try:
                return Decimal(str(val))
            except (InvalidOperation, TypeError):
                continue

    print(f"⚠️ No usable price field found in response: {data}")
    return None



def compute_cost_basis_and_roi(df, current_price):
    """
    Returns (avg_entry, roi_frac, exposure) using only filled BUYs.
    - Synthetic buys (trade_filled=False) are ignored.
    - Exposure = sum(dynamic_amount BUYs) - sum(dynamic_amount SELLs).
    - ROI is returned as a FRACTION (e.g., 0.0325 for +3.25%).
    """

    if df is None or df.empty:
        return None, None, Decimal("0.0")


    df = df.copy()
    df["action"] = df["action"].astype(str).str.upper()


    filled = df[df.get("trade_filled", True) == True]

    buys = filled[filled["action"] == "BUY"]
    sells = filled[filled["action"] == "SELL"]


    if buys.empty:
        return None, None, Decimal("0.0")

    if "dynamic_amount" in buys.columns and buys["dynamic_amount"].notna().any():
        weights = buys["dynamic_amount"].fillna(0).astype(float).to_numpy()
        prices = buys["entry_price"].fillna(np.nan).astype(float).to_numpy()
        mask = (~np.isnan(prices)) & (weights > 0)

        if mask.any():
            avg_entry = float(np.average(prices[mask], weights=weights[mask]))
        else:
            avg_entry = float(buys["entry_price"].dropna().astype(float).mean())
    else:
        avg_entry = float(buys["entry_price"].dropna().astype(float).mean())


    exposure = Decimal("0.0")
    if "dynamic_amount" in filled.columns:
        total_buys = Decimal(str(buys["dynamic_amount"].fillna(0).sum()))
        total_sells = Decimal(str(sells["dynamic_amount"].fillna(0).sum()))
        exposure = total_buys - total_sells


        if exposure < 0:
            exposure = Decimal("0.0")


        if exposure <= Decimal("0.0001"):
            exposure = Decimal("0.0")



    roi_frac = None
    try:
        if avg_entry and current_price:
            avg_entry_dec = Decimal(str(avg_entry))
            current_price_dec = Decimal(str(current_price))

            if avg_entry_dec > 0:
                roi_frac = (current_price_dec - avg_entry_dec) / avg_entry_dec
    except InvalidOperation:
        roi_frac = None

    return avg_entry, roi_frac, exposure




from decimal import Decimal as D




from utils_io import update_crypto_data_file
update_crypto_data_file()
import os
import torch
import traceback
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts





def build_or_load_model(device, input_dim, narrator=None):
    """
    Two-checkpoint system with backward compatibility:
    - Load online checkpoint if present
    - Else load offline checkpoint
    - Support both old ('model_state') and new ('model_state_dict') formats
    - Always use a fresh optimizer for online RL
    """

    OFFLINE_CKPT = r"D:\Scarlet_Works\Scarlet\checkpoints\bestmodel.ckpt"
    ONLINE_CKPT  = r"D:\Scarlet_Works\Scarlet\checkpoints\online_best.ckpt"

    os.makedirs(os.path.dirname(OFFLINE_CKPT), exist_ok=True)
    input_dim = len(INPUT_FEATURES)


    model = HybridForecasterD(
        price_dim=input_dim,
        hidden_size=512,
        num_layers=2,
        output_seq_len=12,
        dropout=0.2,
        use_gradient_checkpointing=True,
        narrator=narrator,
    ).to(device)

    ONLINE_LR = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=ONLINE_LR)


    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=2, eta_min=5e-7
    )

    loss_scheduler = LossScheduler(
        rl_weight_start=0.0,
        rl_weight_max=0.10,
        rl_weight_growth=1.005,
        sched_sampling_start=1.0,
        sched_sampling_end=0.5,
        sched_sampling_decay=0.995,
        supervised_batches_per_epoch=10,
        min_supervised_batches=5,
        supervised_decay=0.995,
        reward_ramp=0.05,
        plateau_threshold=0.002,
        stagnation_epochs_trigger=20,
        stagnation_window=5,
    )

    epoch = 0

 
    def extract_state(ckpt, key_new, key_old):
        if key_new in ckpt:
            return ckpt[key_new]
        if key_old in ckpt:
            return ckpt[key_old]
        raise RuntimeError(f"Checkpoint missing both '{key_new}' and '{key_old}'")


    if os.path.isfile(ONLINE_CKPT):
        ckpt = torch.load(ONLINE_CKPT, map_location=device)

        state = extract_state(ckpt, "model_state_dict", "model_state")
        model.load_state_dict(state)

        epoch = ckpt.get("epoch", 0)

        if narrator:
            narrator.narrate("🧬 Loaded online micro‑checkpoint (adapted brain).")

        return model, optimizer, lr_scheduler, loss_scheduler, epoch


    if os.path.isfile(OFFLINE_CKPT):
        ckpt = torch.load(OFFLINE_CKPT, map_location=device)

        state = extract_state(ckpt, "model_state_dict", "model_state")
        model.load_state_dict(state)

        epoch = ckpt.get("epoch", 0)

        if narrator:
            narrator.narrate("🧬 Loaded offline baseline model.")

        return model, optimizer, lr_scheduler, loss_scheduler, epoch


    torch.save({"model_state_dict": model.state_dict()}, OFFLINE_CKPT)

    if narrator:
        narrator.narrate("✨ Created fresh model and saved offline baseline.")

    return model, optimizer, lr_scheduler, loss_scheduler, epoch
import torch

import torch
from decimal import Decimal


def run_model(model, input_data, device):
    """
    Runs the model and returns a scalar forecast suitable for Scarlet's policy engine.
    """

    model.eval()  
    input_data = input_data.to(device)

    with torch.no_grad():
        output = model(input_data)

   
    if isinstance(output, tuple):
        output = output[0]

   
    if isinstance(output, torch.Tensor):
        output = output.squeeze().detach().cpu().numpy()

    try:
        forecast = float(output)
    except Exception:
        print("⚠️ Model output not convertible to float — returning HOLD.")
        return "HOLD", Decimal("0.0")

   
    market_price = get_market_price()
    if market_price is None:
        print("⚠️ Market price unavailable — returning HOLD.")
        return "HOLD", Decimal("0.0")


    action, delta = predict_action(
        market_price=Decimal(str(market_price)),
        model_forecast=Decimal(str(forecast)),
        policy_config=POLICY_CONFIG,
        is_flat=False,   
        symbol=sym,      
    )




    return action, delta





def load_market_data(symbol="solusd"):
    url = f"https://api.gemini.com/v2/candles/{symbol}/15m"
    response = requests.get(url)

    if response.status_code == 200 and response.headers.get("Content-Type", "").startswith("application/json"):
        raw = response.json()
        if not raw or not isinstance(raw, list):
            print("❌ Unexpected candle format — empty or non-list")
            return pd.DataFrame()

        return pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])

    print("❌ Failed to fetch live candles")
    return pd.DataFrame()



import torch
import random




import torch

import json

import requests



import csv

def safe_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

from decimal import Decimal
import csv
import pandas as pd






def clean_row(row):
    return {
        k: str(v).encode("utf-8", errors="replace").decode("utf-8")
        if isinstance(v, str) else v
        for k, v in row.items()
    }





from decimal import Decimal

from decimal import Decimal
state_flags = globals().get("state_flags", {})



from typing import Tuple
from decimal import Decimal, InvalidOperation


def predict_action(
    market_price: Decimal,
    model_forecast: Decimal,
    policy_config,
    is_flat: bool,
    symbol: str,
):
    """
    Modern, symbol‑aware action inference.
    Uses per‑asset thresholds from ScarletPolicyConfig.
    """


    hold_z = policy_config.hold_zone[symbol]
    buy_th = policy_config.buy_delta_threshold[symbol]
    sell_th = policy_config.sell_delta_threshold[symbol]


    if market_price <= 0:
        return "HOLD", Decimal("0")

    forecast_delta = (model_forecast - market_price) / market_price

    if is_flat:
        return "HOLD", forecast_delta


    if abs(forecast_delta) <= hold_z:
        return "HOLD", forecast_delta


    if forecast_delta > buy_th:
        return "BUY", forecast_delta


    if forecast_delta < -sell_th:
        return "SELL", forecast_delta

    return "HOLD", forecast_delta

from decimal import Decimal
from datetime import datetime




def is_buy_allowed(
    forecast_delta: Decimal,
    atr_ratio: Decimal,
    volatility: Decimal,
    market_price: Decimal,
    vwap: Decimal | None = None,
    narrator=None,
):
    """
    Updated BUY gating for Scarlet's micro‑delta model.
    - Scaled fee floor (0.10% instead of 0.95%)
    - ATR boost tuned for 15m horizons
    - Forecast boost meaningful at micro‑delta scale
    - VWAP tension remains a soft penalty
    """

   
    FEE_RATE = Decimal("0.0008")        # 0.08% round trip
    MIN_NET_GAIN = Decimal("0.0002")    # +0.02% after fees
    fee_floor = FEE_RATE + MIN_NET_GAIN # ≈ 0.0010 (0.10%)

    atr_boost = min(Decimal("0.0020"), atr_ratio * Decimal("1.0"))

  
    forecast_boost = max(Decimal("0.0"), forecast_delta * Decimal("1.0"))

    buy_threshold = fee_floor + atr_boost - forecast_boost
    buy_threshold = max(buy_threshold, fee_floor)

    if narrator:
        narrator.narrate(
            f"🧮 BUY Threshold → fee_floor={fee_floor:.4f}, "
            f"atr_boost={atr_boost:.4f}, forecast_boost={forecast_boost:.4f}, "
            f"base_final={buy_threshold:.4f}"
        )


    if vwap is not None and abs(forecast_delta) < Decimal("0.01"):
        extension = (market_price - vwap) / vwap

    
        if extension > Decimal("0.015"):  # 1.5%
            if narrator:
                narrator.narrate(
                    f"⚠️ Price > VWAP by {extension:.4f} (>1.5%) → BUY suppressed"
                )
            return False


        vwap_penalty = max(Decimal("0"), extension) * Decimal("0.3")
        buy_threshold += vwap_penalty

        if narrator:
            narrator.narrate(
                f"📉 VWAP penalty → extension={extension:.4f}, "
                f"penalty={vwap_penalty:.4f}, new_threshold={buy_threshold:.4f}"
            )

    if forecast_delta >= buy_threshold:
        if narrator:
            narrator.narrate("🟢 BUY allowed (forecast exceeds threshold)")
        return True

    if narrator:
        narrator.narrate(
            f"❌ BUY suppressed (Δ={forecast_delta:.4f} < threshold={buy_threshold:.4f})"
        )
    return False
def log_event(message, log_file="scarlet_events.log"):
    timestamp = datetime.utcnow().isoformat()
    with open(log_file, "a", encoding="utf-8", errors="replace") as f:
        f.write(f"{timestamp} — {message}\n")
    print(message)



def check_profit_margin_suppressor(
    market_price,
    avg_entry_price,
    recent_profit,
    reset_recent,
    commentary,
    multiplier=Decimal("0.7"),
    forecast=None,
    volatility=None,
):

    try:
        avg_entry_dec = Decimal(str(avg_entry_price))
        if avg_entry_dec <= 0 or reset_recent:
            raise Exception()
    except Exception:
        commentary.append("🧮 Profit margin check skipped — cost basis reset or zero.")
        return False, None

    if forecast and forecast > Decimal("0.4"):
        commentary.append("🧠 Forecast override — suppressor bypassed.")
        return False, None

    if volatility and volatility > Decimal("0.5"):
        commentary.append("🧠 Volatility override — suppressor bypassed.")
        return False, None


    gain_ratio = (market_price - avg_entry_dec) / avg_entry_dec
    min_gain_pct = max(
        Decimal("0.0025"),
        Decimal("0.0025") + multiplier * Decimal(str(recent_profit)),
    )

    commentary.append(
        f"🧮 Entry gain: {gain_ratio:.4f} vs Threshold: {min_gain_pct:.4f}"
    )

    if gain_ratio < min_gain_pct:
        commentary.append("🔕 Suppressor: Entry gain below dynamic threshold.")
        return True, "profit_margin_below_threshold"

    return False, None

from decimal import Decimal, ROUND_DOWN




from decimal import Decimal








from decimal import Decimal
from datetime import datetime
import csv




def resolve_exit_note(response, market_price):
    """
    Determines which price source was used for SELL anchor logging.
    """

    if not response:
        return "no_response_trade_suppressed"

    try:
        if response.get("filled_price") not in (None, "0.00", ""):
            return "from_filled_price"

        if response.get("avg_execution_price") not in (None, "0.00", ""):
            return "from_avg_execution_price"

        return "fallback_to_market_price"

    except Exception:
        return "fallback_due_to_error"


class MarginCalculator:
    """
    Computes an adaptive SELL margin based on forecast + volatility.
    Produces a margin in *percent units* (0.0025 = 0.25%).
    """

    def __init__(
        self,
        base_margin=Decimal("0.0025"),      
        forecast_weight=Decimal("0.5"),   
        volatility_weight=Decimal("0.3"),   
        max_margin=Decimal("0.05"),         
    ):
        self.base_margin = base_margin
        self.forecast_weight = forecast_weight
        self.volatility_weight = volatility_weight
        self.max_margin = max_margin

    def compute(self, volatility, forecast):
        """
        Returns (margin: Decimal, commentary: str)
        """

        try:
            forecast = Decimal(str(forecast or "0.0"))
            volatility = Decimal(str(volatility or "0.0"))

       
            dynamic_margin = (
                self.base_margin
                + self.forecast_weight * forecast
                + self.volatility_weight * volatility
            )

       
            margin = max(self.base_margin, min(dynamic_margin, self.max_margin))

            commentary = (
                f"🧮 Adaptive margin = {margin:.4f} "
                f"(base={self.base_margin:.4f}, "
                f"forecast={forecast:.2f}, "
                f"volatility={volatility:.2f})"
            )

            return margin, commentary

        except Exception as e:
            error_msg = f"⛔ Margin calculation failed: {e}"
            return self.base_margin, error_msg


from decimal import Decimal, InvalidOperation
import pandas as pd
import csv




    





from decimal import Decimal
from typing import Optional, List, Tuple

class CostBasisVerdict:
    """
    Unified SELL gating logic.
    Posture‑safe, anchor‑aware, ROI‑aware, fee‑aware, and now regime‑aware.
    """

 
    NEUTRAL_BAND = Decimal("0.0025")     
    LOSS_THRESHOLD = Decimal("-0.0300")   

 
    FEE_RATE = Decimal("0.0070")
    MIN_NET_PROFIT = Decimal("0.0010")
    FEE_AWARE_THRESHOLD = FEE_RATE + MIN_NET_PROFIT   

    def __init__(
        self,
        action: str,
        market_price: Decimal,
        avg_entry: Decimal,
        exposure: Decimal,
        forecast_delta: Decimal,
        volatility: Decimal = None,
        slope_value: Decimal = None,
        flat_score: Decimal = None,   
        narrator=None,
    ):
        self.action = action.upper()
        self.market_price = Decimal(str(market_price))
        self.avg_entry = avg_entry
        self.exposure = exposure
        self.forecast_delta = Decimal(str(forecast_delta))
        self.volatility = volatility
        self.slope_value = slope_value
        self.flat_score = flat_score or Decimal("0")   
        self.narrator = narrator

        self.should_execute = False
        self.reason = ""
        self.entry_price = avg_entry

        self._evaluate()

    
    def _evaluate(self):

       
        if self.action != "SELL":
            self.should_execute = True
            self.reason = "BUY allowed — SELL gating applies only to SELL."
            return

        
        if self.exposure is None or self.exposure <= Decimal("0.0001"):
            self.should_execute = False
            self.reason = "SELL suppressed — no meaningful exposure."
            return

       
        if self.avg_entry is None:
            self.should_execute = False
            self.reason = "SELL suppressed — no valid avg_entry anchor."
            return

      
        roi = (self.market_price - self.avg_entry) / self.avg_entry

  
        if self.flat_score > Decimal("0.7"):
         
            if abs(roi) < Decimal("0.005"):  
                self.should_execute = False
                self.reason = (
                    "Flat regime HOLD → low volatility, no directional edge."
                )
                return

    
        calculator = MarginCalculator(
            base_margin=Decimal("0.03"),        
            forecast_weight=Decimal("0.25"),
            volatility_weight=Decimal("0.50"),
            max_margin=Decimal("0.10"),         
        )

        threshold_pct, margin_comment = calculator.compute(
            self.volatility, self.forecast_delta
        )
        adaptive_threshold = threshold_pct / Decimal("100")

        if self.narrator:
            self.narrator.narrate(
                f"🧮 Adaptive ROI Threshold: {threshold_pct:.2f}% → {margin_comment}"
            )

       
        effective_threshold = max(adaptive_threshold, self.FEE_AWARE_THRESHOLD)

        if roi >= effective_threshold:
            self.should_execute = True
            self.reason = (
                f"ROI {roi:.4%} ≥ effective threshold {effective_threshold:.4%} "
                f"(adaptive={adaptive_threshold:.4%}, fee_floor={self.FEE_AWARE_THRESHOLD:.4%})."
            )
            return

     
        if roi <= self.LOSS_THRESHOLD:
            self.should_execute = True
            self.reason = (
                f"ROI {roi:.4%} ≤ loss threshold {self.LOSS_THRESHOLD:.4%} → controlled loss SELL."
            )
            return

     
        if abs(roi) < self.NEUTRAL_BAND:

          
            if self.forecast_delta < Decimal("-0.002"):  

             
                high_atr = False
                if self.volatility is not None:
                    atr_ratio = self.volatility / self.market_price
                    high_atr = atr_ratio > Decimal("0.004")   

             
                losing_region = roi < 0

                if high_atr or losing_region:
                    self.should_execute = True
                    self.reason = (
                        "Neutral-band SELL allowed → forecast strongly bearish "
                        "AND (high ATR danger OR exiting losing region)."
                    )
                    return

          
            self.should_execute = False
            self.reason = "Neutral-band HOLD → no danger conditions met."
            return

  
        self.should_execute = False
        self.reason = "ROI in safe zone → HOLD."


    def narrate(self):
        if self.narrator:
            self.narrator.narrate(
                f"🧪 Execution verdict → execute={self.should_execute}, "
                f"action=SELL, reason={self.reason}"
            )            
            
            
import pandas as pd
import numpy as np

class LegacyIndicatorPolicy:
    def __init__(self, price_series, window_rsi=14, window_bb=20, std_dev_multiplier=2):
        self.commentary = []
        self.price_series = price_series
        self.window_rsi = window_rsi
        self.window_bb = window_bb
        self.std_dev_multiplier = std_dev_multiplier

    def calculate_rsi(self):
        delta = self.price_series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1/self.window_rsi, min_periods=self.window_rsi).mean()
        avg_loss = loss.ewm(alpha=1/self.window_rsi, min_periods=self.window_rsi).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def calculate_bollinger_bands(self):
        rolling_mean = self.price_series.rolling(self.window_bb).mean()
        rolling_std = self.price_series.rolling(self.window_bb).std()

        upper_band = rolling_mean + self.std_dev_multiplier * rolling_std
        lower_band = rolling_mean - self.std_dev_multiplier * rolling_std

        return upper_band.iloc[-1], lower_band.iloc[-1], self.price_series.iloc[-1]

    def evaluate(self):
        verdicts = []
        commentary = []

        rsi = self.calculate_rsi()
        upper_band, lower_band, current_price = self.calculate_bollinger_bands()

     
        if rsi >= 70:
            verdicts.append("sell_rsi_overbought")
            commentary.append(f"📉 RSI = {rsi:.2f} → Overbought (SELL pressure)")
        elif rsi <= 30:
            verdicts.append("buy_rsi_oversold")
            commentary.append(f"📈 RSI = {rsi:.2f} → Oversold (BUY pressure)")
        else:
            verdicts.append("info_rsi_neutral")
            commentary.append(f"📊 RSI = {rsi:.2f} → Neutral")

    
        if current_price > upper_band:
            verdicts.append("sell_bb_upper_break")
            commentary.append(
                f"📊 Price {current_price:.2f} > upper band {upper_band:.2f} → SELL pressure"
            )
        elif current_price < lower_band:
            verdicts.append("buy_bb_lower_break")
            commentary.append(
                f"📊 Price {current_price:.2f} < lower band {lower_band:.2f} → BUY pressure"
            )
        else:
            verdicts.append("info_bb_neutral")
            commentary.append(
                f"📊 Price {current_price:.2f} within Bollinger range"
            )

        return verdicts, commentary


from decimal import Decimal, ROUND_DOWN
from decimal import Decimal, InvalidOperation
import numpy as np



def compute_volatility(series, window=10):
    """
    Unified volatility helper.
    Accepts: list, numpy array, pandas Series, or torch tensor.
    Returns: Decimal std of fractional returns over the last `window` periods.
    """
    try:
      
        if hasattr(series, "detach"): 
            arr = series.detach().cpu().numpy().astype(float)
        elif hasattr(series, "to_numpy"): 
            arr = series.to_numpy(dtype=float)
        else: 
            arr = np.array(series, dtype=float)

      
        if len(arr) < window + 1:
            return Decimal("0.0")

     
        window_prices = arr[-(window + 1):]

    
        if np.any(window_prices <= 0):
            return Decimal("0.0")

       
        returns = np.diff(window_prices) / window_prices[:-1]

       
        vol = np.std(returns)

        if np.isnan(vol) or np.isinf(vol):
            return Decimal("0.0")

        return Decimal(str(vol))

    except Exception:
        return Decimal("0.0")

def resolve_trade_size(
    action,
    forecast_decimal,
    market_price,
    usd_balance,
    sol_balance,
    asset_balance,
    override_triggered,
    volatility,
    atr_value,
    limit_price=None,
    hard_stop_loss=False,

   
    min_trade_size=Decimal("0.0000000001"),
    max_trade_size=Decimal("1.0"),
    micro_scalp_factor=Decimal("0.01"),
):
    """
    Fully multi‑asset‑safe sizing function.
    All config values must be passed in as Decimals, never dicts.
    """

    suppressors = []

  
    if hard_stop_loss:
        return Decimal(str(asset_balance)), []

   
    usd_balance = Decimal(str(usd_balance))
    asset_balance = Decimal(str(asset_balance))
    market_price = Decimal(str(market_price))
    volatility = Decimal(str(volatility))
    forecast_decimal = Decimal(str(forecast_decimal))
    atr_value = Decimal(str(atr_value))

    action_upper = action.upper()

  
    confidence = min(abs(forecast_decimal), Decimal("1.0"))

  
    volatility_factor = min(Decimal("2.5"), Decimal("1.0") + volatility)

    dynamic_amount = (
        micro_scalp_factor
        + (confidence ** 2 * Decimal("0.10"))
        + (atr_value * Decimal("0.01"))
    ) * volatility_factor

   
    atr_ratio = atr_value / market_price if market_price > 0 else Decimal("0")


    atr_min = Decimal("0.002")
    atr_max = Decimal("0.050")

    if action_upper == "SELL" and atr_ratio < atr_min and not override_triggered:
        suppressors.append("sell_atr_too_low")
        return Decimal("0.0"), suppressors

    if atr_ratio > atr_max and not override_triggered:
        suppressors.append("atr_too_high")
        return Decimal("0.0"), suppressors

   
    dynamic_amount = max(dynamic_amount, min_trade_size)
    dynamic_amount = min(dynamic_amount, max_trade_size)


    if action_upper == "SELL":
        safe_amount = min(dynamic_amount, asset_balance).quantize(
            Decimal("0.0001"), rounding=ROUND_DOWN
        )

        if safe_amount < min_trade_size and not override_triggered:
            suppressors.append("sell_size_below_minimum")
            return Decimal("0.0"), suppressors

        if override_triggered:
            safe_amount = max(safe_amount, min_trade_size)

        return safe_amount, suppressors

   
    if action_upper == "BUY":
        effective_price = Decimal(str(limit_price or market_price))
        buffer_usd = Decimal("2.50")
        max_cost_usd = max(Decimal("0.0"), usd_balance - buffer_usd)

        max_safe_sol = (max_cost_usd / effective_price).quantize(
            Decimal("0.0001"), rounding=ROUND_DOWN
        )

       
        remaining_exposure = max_trade_size  

        safe_amount = min(dynamic_amount, max_safe_sol, remaining_exposure)

        if safe_amount < min_trade_size and not override_triggered:
            suppressors.append("buy_size_below_minimum@BUY")
            return Decimal("0.0"), suppressors

        safe_amount = safe_amount.quantize(Decimal("0.0001"), rounding=ROUND_DOWN)

        if override_triggered:
            safe_amount = max(safe_amount, min_trade_size)

        return safe_amount, suppressors

   
    suppressors.append(f"unknown_action@{action_upper}")
    return Decimal("0.0"), suppressors



from decimal import Decimal

import pandas as pd

class PositionState:
    def __init__(self):
        self.side = None               
        self.entry_time = None
        self.entry_price = None
        self.size = Decimal("0.0")

      
        self.exposure_sol = Decimal("0.0")
        self.last_exit_roi = None
        self.last_exit_pnl = None

       
        self.cooldown_cycles = 0
        self.last_loss = None

        self.synthetic_posture = None  
        self.synthetic_reason = None

 
    def inject_synthetic_posture(self, action: str, reason: str = None):
        self.synthetic_posture = action.upper()
        self.synthetic_reason = reason

 
    def on_buy(self, entry_price, size, entry_time=None):
        """Open or add to a long position."""
        if self.side == "SHORT":
            raise RuntimeError("Cannot BUY while SHORT — must exit first.")

        self.side = "LONG"
        self.entry_price = Decimal(str(entry_price))
        self.size = Decimal(str(size))
        self.entry_time = entry_time
        self.exposure_sol += self.size

        self.synthetic_posture = None
        self.synthetic_reason = None

    def on_sell_short(self, entry_price, size, entry_time=None):
        """Open or add to a short position."""
        if self.side == "LONG":
            raise RuntimeError("Cannot SHORT while LONG — must exit first.")

        self.side = "SHORT"
        self.entry_price = Decimal(str(entry_price))
        self.size = Decimal(str(size))
        self.entry_time = entry_time

        self.synthetic_posture = None
        self.synthetic_reason = None

    def on_exit(self, exit_price=None):
        """Close any position and compute ROI/PnL."""
        if self.side and exit_price is not None and self.entry_price:
            exit_price = Decimal(str(exit_price))
            entry_price = Decimal(str(self.entry_price))

            self.last_exit_roi = (exit_price - entry_price) / entry_price
            self.last_exit_pnl = (exit_price - entry_price) * self.size

           
            if self.last_exit_roi < 0:
                self.last_loss = self.last_exit_roi

     
        self.side = None
        self.entry_time = None
        self.entry_price = None
        self.size = Decimal("0.0")

        self.synthetic_posture = None
        self.synthetic_reason = None


    def is_open(self) -> bool:
        return self.side is not None and self.size > 0

    def in_cooldown(self) -> bool:
        return self.cooldown_cycles > 0

    def reduce_cooldown(self):
        if self.cooldown_cycles > 0:
            self.cooldown_cycles -= 1

    def __repr__(self):
        return (
            f"PositionState(side={self.side}, entry_price={self.entry_price}, "
            f"size={self.size}, exposure={self.exposure_sol}, "
            f"cooldown={self.cooldown_cycles}, last_loss={self.last_loss}, "
            f"synthetic_posture={self.synthetic_posture}, "
            f"synthetic_reason={self.synthetic_reason})"
        )




position_state = PositionState()


def normalize_timestamp_column(df, col="timestamp"):
    def parse_ts(x):
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return pd.NaT

            s = str(x).strip()

         
            try:
                val = int(float(s))  
                if val > 1e12:       
                    return pd.to_datetime(val, unit="ms", utc=True)
                elif val > 1e9:      
                    return pd.to_datetime(val, unit="s", utc=True)
            except Exception:
                pass

            
            return pd.to_datetime(s, utc=True, errors="coerce")

        except Exception:
            return pd.NaT

    df[col] = df[col].apply(parse_ts)
    return df[df[col].notnull()].reset_index(drop=True)


def reset_if_dust(
    asset_balance,
    df,
    current_price,
    memory_path,
    narrator,
    symbol,
    forecast_delta=None,
    avg_slope=None,
    position_state=None,
):
    """
    Multi‑asset dust reset logic (posture‑safe).
    - Filters memory by symbol
    - Uses per‑asset dust thresholds
    - Skips reset if meaningful exposure exists
    - Skips reset if valid BUYs exist *after last reset*
    - Logs RESET_REBASE with symbol included
    """

    from decimal import Decimal
    import pandas as pd

    sym_key = (
        symbol.lower()
        .replace("-", "")
        .replace("_", "")
        .replace("/", "")
        .strip()
    )

    dust_threshold = POLICY_CONFIG.dust_thresholds.get(sym_key, Decimal("0.00001"))

   
    if current_price is None or pd.isna(current_price) or current_price <= 0:
        narrator.narrate("⚠️ Invalid baseline price — skipping reset.")
        return df, False, False

  
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    df["action"] = df["action"].astype(str).str.upper().str.strip()
    df["trade_filled"] = (
        df.get("trade_filled", "0")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

  
    if "symbol" in df.columns:
        df = df[df["symbol"].astype(str).str.lower() == sym_key]


    if df.empty:
        if asset_balance < dust_threshold:
            reset_row = {
                "timestamp": pd.Timestamp.now(tz="UTC"),
                "action": "RESET_REBASE",
                "symbol": sym_key,
                "price": None,
                "amount": float(asset_balance),
                "trade_filled": "0",
                "exit_note": "reset_if_dust",
                "reset_baseline": float(current_price),
            }
            df = pd.DataFrame([reset_row])
            df.to_csv(memory_path, index=False)
            narrator.narrate(
                f"🧹 Logged RESET_REBASE (empty asset memory) → baseline={float(current_price):.4f}"
            )
            return df, True, False

        return df, False, False

    if asset_balance >= dust_threshold:
        return df, False, False

    if position_state and position_state.is_open():
        narrator.narrate("🛡️ Position open → skipping dust reset.")
        return df, False, False

    reset_mask = df["action"].str.contains("RESET", case=False, na=False)
    last_reset_ts = None
    if reset_mask.any():
        last_reset_ts = df.loc[reset_mask, "timestamp"].max()

 
    mask_filled = df["trade_filled"].isin(["1", "1.0", "true", "yes"])
    buys = df[(df["action"] == "BUY") & mask_filled]

    if last_reset_ts is not None:
        buys = buys[buys["timestamp"] > last_reset_ts]

 
    if not buys.empty:
        narrator.narrate(
            f"🛡️ Valid BUYs exist for {sym_key} after last reset — skipping dust reset."
        )
        return df, False, False


    reset_row = {
        "timestamp": pd.Timestamp.now(tz="UTC"),
        "action": "RESET_REBASE",
        "symbol": sym_key,
        "price": None,
        "amount": float(asset_balance),
        "trade_filled": "0",
        "exit_note": "reset_if_dust",
        "reset_baseline": float(current_price),
    }

    df = pd.concat([df, pd.DataFrame([reset_row])], ignore_index=True)
    df.to_csv(memory_path, index=False)

    narrator.narrate(
        f"🧹 Logged RESET_REBASE ({sym_key}) → baseline={float(current_price):.4f}"
    )

    return df, True, False


def reset_avg_entry(
    df,
    sol_balance,
    recent_trades,
    current_price,
    memory_path,
    narrator,
    position_state=None,
):
    """
    Reset cost basis when position is effectively flat (balance below dust threshold).
    Appends a RESET_REBASE row to the memory log so future cost basis calculations
    have a clean baseline.
    """

    dust_threshold = Decimal("0.00001")


    if sol_balance >= dust_threshold:
        return df

    narrator.narrate(
        f"🧹 Position exited → SOL balance = {sol_balance:.8f} < dust threshold = {dust_threshold}"
    )


    required_cols = [
        "timestamp", "action", "entry_price", "exit_price", "dynamic_amount",
        "market_price", "profit", "trade_filled", "exit_note", "side",
        "reset_baseline"
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    df["action"] = df["action"].astype(str).str.upper()


    if "trade_filled" not in df.columns:
        df["trade_filled"] = True

    if not recent_trades.empty and "SELL" in recent_trades["action"].values:
        last_sell_time = recent_trades.loc[
            recent_trades["action"] == "SELL", "timestamp"
        ].max()

        reset_mask = (df["action"] == "BUY") & (df["timestamp"] <= last_sell_time)
        reset_count = int(reset_mask.sum())

        df.loc[reset_mask, "trade_filled"] = False
        narrator.narrate(
            f"🧹 Reset trade_filled on {reset_count} BUY entries before last SELL"
        )
    else:
        narrator.narrate("ℹ️ No SELLs found in recent trades — no BUY resets applied")


    if not df.empty and df.iloc[-1]["action"] in ("RESET", "RESET_REBASE"):
        narrator.narrate("⚠️ Skipping duplicate RESET log")
        return df


    if position_state:
        position_state.synthetic_posture = None
        position_state.synthetic_reason = None

 
    reset_row = {
        "timestamp": pd.Timestamp.now(tz="UTC"),
        "action": "RESET_REBASE",
        "entry_price": None,
        "exit_price": None,
        "dynamic_amount": 0.0,
        "market_price": float(current_price),
        "profit": 0.0,
        "trade_filled": False,
        "exit_note": "reset_avg_entry",
        "side": "",
        "reset_baseline": float(current_price),
    }

    df = pd.concat([df, pd.DataFrame([reset_row])], ignore_index=True)
    df.to_csv(memory_path, index=False)

    narrator.narrate(
        f"🧹 Logged RESET_REBASE to memory → {memory_path} (baseline={current_price:.4f})"
    )

    return df



def log(metric_name: str, value):
    """
    Safe metric logger for Decimal, float, int, or None.
    Always prints cleanly without raising formatting errors.
    """
    try:
        if value is None:
            print(f"[LOG] {metric_name} = None")
        else:
           
            from decimal import Decimal, InvalidOperation
            try:
                v = Decimal(str(value))
                print(f"[LOG] {metric_name} = {v:.4f}")
            except InvalidOperation:
                print(f"[LOG] {metric_name} = {value}")
    except Exception as e:
        print(f"[LOG] {metric_name} = <logging error: {e}>")
        
        
        


def get_wallet_balance(asset, nonce_mgr=None):
    snapshot = get_wallet_snapshot(nonce_mgr)
    value = snapshot.get(asset, Decimal("0"))
    return Decimal(value)



class EntryAnchor:
    def __init__(
        self,
        price: Decimal,
        amount: Decimal,
        symbol: str,
        action: str,
        trade_filled: str = "1",
        exit_note: str = "",
        timestamp=None
    ):
        self.price = float(price)
        self.amount = float(amount)
        self.symbol = symbol.lower()
        self.action = action.upper()         
        self.trade_filled = trade_filled      
        self.exit_note = exit_note           
        self.timestamp = timestamp or pd.Timestamp.now(tz="UTC")

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "action": self.action,
            "symbol": self.symbol,
            "price": self.price,
            "amount": self.amount,
            "trade_filled": self.trade_filled,
            "exit_note": self.exit_note,
        }# ------------------------------------------------------------
import os
import numpy as np
import pandas as pd
from decimal import Decimal, InvalidOperation

def is_micro_scalp_candidate_aggressive(
    sol_balance: float,
    forecast_delta: float,
    short_slope: float,
    medium_slope: float,
    atr_value: float,
    market_price: float,
) -> bool:
    """
    Aggressive micro‑scalp candidate check.
    Policy‑aware, posture‑safe, and ATR‑normalized.
    """

  
    if not AGGRESSIVE_MICRO_SCALP_ENABLED:
        return False

    dust_limit = POLICY_CONFIG.micro_scalp_dust.get("solusd", Decimal("0.0"))
    if sol_balance > dust_limit:

        return False

   
    min_delta = getattr(POLICY_CONFIG, "micro_scalp_min_delta", 0.0015)
    if forecast_delta < min_delta:
        return False

  
    if short_slope <= 0:
        return False

   
    min_medium_slope = getattr(POLICY_CONFIG, "micro_scalp_min_medium_slope", -0.01)
    if medium_slope < min_medium_slope:
        return False

  
    if market_price <= 0:
        return False  

    atr_ratio = float(atr_value) / float(market_price)

    atr_max = float(getattr(POLICY_CONFIG, "atr_max_ratio", 0.05))
    if atr_ratio > atr_max:
        return False

    return True

def compute_micro_scalp_size_aggressive(
    usd_balance: float,
    forecast_delta: float,
    atr_value: float,
    sol_balance: float,
) -> float:
    """
    Policy-aware aggressive micro-scalp sizing.
    """

 
    micro_factor = POLICY_CONFIG.micro_scalp_factor

    confidence = min(abs(forecast_delta), 1.0)

    volatility_factor = min(2.5, 1.0 + float(POLICY_CONFIG.volatility))

    atr_boost = float(atr_value) * 0.01

    dynamic_amount = (
        micro_factor
        + confidence ** 2 * 0.10
        + atr_boost
    ) * volatility_factor

    remaining_exposure = max(
        0.0,
        float(POLICY_CONFIG.max_exposure_sol) - float(sol_balance)
    )

    max_affordable = usd_balance / float(POLICY_CONFIG.last_price)

    safe_amount = min(
        dynamic_amount,
        remaining_exposure,
        max_affordable
    )

    return max(safe_amount, float(POLICY_CONFIG.min_trade_size))


def should_exit_micro_scalp_aggressive(
    roi: float | None,
    forecast_delta: float,
    short_slope: float,
    atr_value: float,
    last_price: float = None,
) -> bool:
    """
    Policy-aware aggressive micro-scalp exit logic.
    """

    if not AGGRESSIVE_MICRO_SCALP_ENABLED:
        return False

    tp = POLICY_CONFIG.micro_scalp_take_profit
    sl = POLICY_CONFIG.micro_scalp_stop_loss
    min_delta_exit = POLICY_CONFIG.buy_delta_threshold * 0.5

    if last_price:
        atr_ratio = atr_value / max(1e-9, last_price)
    else:
        atr_ratio = 0.0

    if roi is None:
        if forecast_delta < min_delta_exit:
            return True
        if short_slope < 0:
            return True
        if atr_ratio > POLICY_CONFIG.atr_max_ratio:  
            return True
        return False


    if roi >= tp:
        return True
    if roi <= sl:
        return True

    if forecast_delta < min_delta_exit:
        return True
    if short_slope < 0:
        return True
    if atr_ratio > POLICY_CONFIG.atr_max_ratio: 
        return True

    return False

def classify_regime(
    flat_score,
    slope_10,
    slope_30,
    slope_40,
    atr,
    volatility,
    bb_width,
    macd_hist,
    recent_delta,
    narrator=None,
    base=None,
):
    """
    Multi‑feature regime classifier for Scarlet.
    Returns one of:
        FLAT, UP, DOWN, VOLATILE, BREAKOUT, NORMAL
    """

    if flat_score > 0.70 and abs(slope_30) < 0.01 and bb_width < 0.015:
        regime = "FLAT"
        if narrator:
            narrator.narrate(
                f"[Regime:{base}] flat_score={float(flat_score):.2f}, "
                f"slope30={float(slope_30):.4f}, bb_width={float(bb_width):.4f} → FLAT"
            )
        return regime

    if volatility > 0.012 or atr > 0.02:
        regime = "VOLATILE"
        if narrator:
            narrator.narrate(
                f"[Regime:{base}] vol={float(volatility):.4f}, atr={float(atr):.4f} → VOLATILE"
            )
        return regime


    if abs(slope_40) > 0.04 and bb_width > 0.025:
        regime = "BREAKOUT"
        if narrator:
            narrator.narrate(
                f"[Regime:{base}] slope40={float(slope_40):.4f}, bb_width={float(bb_width):.4f} → BREAKOUT"
            )
        return regime

    if slope_40 > 0.02 and macd_hist > 0:
        regime = "UP"
        if narrator:
            narrator.narrate(
                f"[Regime:{base}] slope40={float(slope_40):.4f}, macd={float(macd_hist):.4f} → UP"
            )
        return regime

    if slope_40 < -0.02 and macd_hist < 0:
        regime = "DOWN"
        if narrator:
            narrator.narrate(
                f"[Regime:{base}] slope40={float(slope_40):.4f}, macd={float(macd_hist):.4f} → DOWN"
            )
        return regime

    regime = "NORMAL"
    if narrator:
        narrator.narrate(
            f"[Regime:{base}] flat_score={float(flat_score):.2f}, "
            f"slope40={float(slope_40):.4f}, vol={float(volatility):.4f} → NORMAL"
        )
    return regime

def execute_trade_part1(
    base,
    model_forecast,
    market_price,
    action,
    symbol,
    price_tensor,
    avg_entry_price,
    crypto_df,
    scaler,
    trade_nonce_mgr,
    flat_score,
    override_triggered=False,
    suppressors=None,
    amount=None,
    asset_balance=None,
    adaptive_threshold=Decimal("0.01"),
    latest_volatility=Decimal("0.0"),
    profit_log="scarlet_profit.csv",
    memory_log="scarlet_memory.csv",
    forecast_np=None,
    entry_info=None,
    portfolio_posture=None,
    forecast_strength=None,
    price_series=None,
    price_index=0,
    roi_reason=None,
    slope_10=None,
    slope_30=None,
    slope_40=None,
    narrator=None,
    mode=None,
    sol_balance=None,
    usd_balance=None,
    
):

    if price_series is None:
        raise ValueError("price_series is required for LegacyIndicatorPolicy.")

    suppressors = suppressors or []
    action_upper = str(action).upper()

    indicator_policy = LegacyIndicatorPolicy(price_series)
    indicator_verdicts, indicator_commentary = indicator_policy.evaluate()

    suppressors.extend(indicator_verdicts)
    if narrator:
        for line in indicator_commentary:
            narrator.narrate(line)

    rsi = indicator_policy.calculate_rsi()
    upper_bb, lower_bb, current_price = indicator_policy.calculate_bollinger_bands()
    sym_key = str(symbol).lower()

  
    if narrator:
        narrator.narrate(
            f"💰 Wallet Snapshot (part1) → {sym_key.upper().replace('USD','')}: {asset_balance}, USD: {usd_balance}"
        )


  
    df = pd.read_csv(memory_log, on_bad_lines="skip") if os.path.exists(memory_log) else pd.DataFrame()
    df = normalize_timestamp_column(df, "timestamp") if not df.empty else df

 
    sym_key = str(symbol).lower()

    
    has_buys = False
    if not df.empty and "action" in df.columns and "symbol" in df.columns:
        df["action_norm"] = df["action"].astype(str).str.upper()
        df["symbol_norm"] = df["symbol"].astype(str).str.lower()
        has_buys = ((df["action_norm"] == "BUY") & (df["symbol_norm"] == sym_key)).any()


    if avg_entry_price is None:
        try:
            entry_info = get_entry_info(memory_log, symbol=sym_key, narrator=narrator)
        except Exception as e:
            entry_info = {"avg_entry_price": None, "exposure": None, "valid_anchor": False}
            if narrator:
                narrator.narrate(f"⚠️ get_entry_info failed → {e}")

        avg_entry_ledger = entry_info.get("avg_entry_price")
        exposure_ledger = entry_info.get("exposure")
        valid_anchor = entry_info.get("valid_anchor", False)

        already_recovered = False
        if portfolio_posture is not None:
            recovered_flags = getattr(portfolio_posture, "anchor_recovered", {})
            if not isinstance(recovered_flags, dict):
                recovered_flags = {}
            already_recovered = recovered_flags.get(sym_key, False)

        if valid_anchor and avg_entry_ledger is not None:
            avg_entry_price = Decimal(str(avg_entry_ledger))

            if narrator and not already_recovered:
                narrator.narrate(
                    f"📘 Fresh anchor → exposure recovered from dust "
                    f"({sym_key.upper().replace('USD','')} wallet={asset_balance}) "
                    f"→ avg_entry={float(avg_entry_price):.3f}"
                )

            if portfolio_posture is not None:
                recovered_flags[sym_key] = True
                portfolio_posture.anchor_recovered = recovered_flags

        else:
        
            if narrator:
                if not has_buys:
                    narrator.narrate("⚠️ No valid BUYs in memory — skipping ROI exit logic.")
                else:
                    narrator.narrate("⚠️ Recovery failed but BUYs exist — posture anchor missing.")

           
            if not has_buys:
                if not df.empty and str(df.iloc[-1].get("action", "")).upper() == "RESET_REBASE":
                    if narrator:
                        narrator.narrate("⚠️ Skipping duplicate RESET_REBASE log")
                else:
                    baseline = float(Decimal(str(market_price)))
                    log_reset(memory_log, baseline, rebase=True)
                    if narrator:
                        narrator.narrate(
                            f"🧹 Logged RESET_REBASE → {memory_log} (baseline={baseline:.4f})"
                        )
            else:
                if narrator:
                    narrator.narrate(
                        "⚠️ Recovery failed but BUYs exist — skipping RESET to avoid loop."
                    )

            
            if not has_buys:
               
                if not df.empty and str(df.iloc[-1].get("action", "")).upper() == "RESET_REBASE":
                    if narrator:
                        narrator.narrate("⚠️ Skipping duplicate RESET_REBASE log")
                else:
                   
                    baseline = float(Decimal(str(market_price)))
                    log_reset(memory_log, baseline, rebase=True)
                    if narrator:
                        narrator.narrate(
                            f"🧹 Logged RESET_REBASE → {memory_log} (baseline={baseline:.4f})"
                        )
            else:
                if narrator:
                    narrator.narrate(
                        "⚠️ Recovery failed but BUYs exist — skipping RESET to avoid loop."
                    )

    
    try:
        forecast_decimal = Decimal(str(model_forecast))
        if not forecast_decimal.is_finite():
            forecast_decimal = Decimal("0.0")
    except InvalidOperation:
        forecast_decimal = Decimal("0.0")

    market_price_dec = Decimal(str(market_price))
    forecast_price = market_price_dec * (Decimal("1.0") + forecast_decimal)
    confidence = min(abs(forecast_decimal), Decimal("1.0"))

    print(f"🧪 Forecast→Price: {float(forecast_price):.4f}, Δ={float(forecast_decimal):.4f}")

    def safe_decimal(x, default="0.0"):
        try:
            if x is None:
                return Decimal(default)
            val = float(x)
            if not (val == val and abs(val) != float("inf")):
                return Decimal(default)
            return Decimal(str(val))
        except Exception:
            return Decimal(default)


    
    vol_tensor = compute_volatility(price_tensor, window=10)
    volatility = safe_decimal(vol_tensor, default=str(latest_volatility))


    def safe_decimal(x, default="0.0"):
        try:
            if x is None:
                return Decimal(default)
            val = float(x)
            if not (val == val and abs(val) != float("inf")):
                return Decimal(default)
            return Decimal(str(val))
        except Exception:
            return Decimal(default)

    def _compute_vwap(df, price_col, volume_col, window=75):
        try:
            if volume_col in df.columns:
                p = df[price_col].tail(window).to_numpy(dtype=float)
                v = df[volume_col].tail(window).to_numpy(dtype=float)
                if v.sum() > 0:
                    return float((p * v).sum() / v.sum())
            return float(df[price_col].tail(window).mean())
        except Exception:
            return 0.0

    sym = symbol.lower()
    price_col = f"close_{sym}"
    volume_col = f"volume_{sym}"

    vwap_raw = _compute_vwap(crypto_df, price_col, volume_col)
    vwap_value = safe_decimal(vwap_raw)

    atr_col = f"ATR_{sym}"
    vol_col = f"volatility_{sym}"

    high_col  = f"high_{sym}"
    low_col   = f"low_{sym}"
    close_col = f"close_{sym}"

    if atr_col not in crypto_df.columns:
        if high_col in crypto_df.columns and low_col in crypto_df.columns and close_col in crypto_df.columns:
            try:
                crypto_df[atr_col] = compute_atr(
                    crypto_df[high_col],
                    crypto_df[low_col],
                    crypto_df[close_col]
                ).fillna(0.0)


                if narrator:
                    narrator.narrate(f"📏 ATR computed on‑the‑fly for {sym} (fallback).")

            except Exception:
                crypto_df[atr_col] = 0.0
                if narrator:
                    narrator.narrate(f"⚠️ ATR compute failed for {sym} — defaulting to 0")
        else:
            crypto_df[atr_col] = 0.0
            if narrator:
                narrator.narrate(f"⚠️ ATR columns missing for {sym} — defaulting to 0")

    atr_value = safe_decimal(crypto_df[atr_col].iloc[-1])
    atr_ratio = atr_value / market_price_dec if market_price_dec > 0 else Decimal("0")

    if vol_col not in crypto_df.columns:
        if close_col in crypto_df.columns:
            try:
                close_series = crypto_df[close_col].to_numpy(dtype=float)
                vol_tensor = compute_volatility(close_series, window=10)
                crypto_df[vol_col] = vol_tensor

                if narrator:
                    narrator.narrate(f"🌪️ Volatility computed on‑the‑fly for {sym} (fallback).")

            except Exception:
                crypto_df[vol_col] = 0.0
                if narrator:
                    narrator.narrate(f"⚠️ Volatility compute failed for {sym} — defaulting to 0")
        else:
            crypto_df[vol_col] = 0.0
            if narrator:
                narrator.narrate(f"⚠️ Volatility column missing for {sym} — defaulting to 0")

    vol_value = safe_decimal(crypto_df[vol_col].iloc[-1])

    def ema(series: np.ndarray, span: int) -> np.ndarray:
        alpha = 2.0 / (span + 1)
        ema_vals = np.zeros_like(series, dtype=float)
        ema_vals[0] = series[0]
        for i in range(1, len(series)):
            ema_vals[i] = alpha * series[i] + (1 - alpha) * ema_vals[i - 1]
        return ema_vals

    def compute_macd_and_signal(close_prices: np.ndarray, fast=16, slow=32, signal=12):
        if len(close_prices) < slow:
            raise ValueError("Not enough data to compute MACD")
        macd_line = ema(close_prices, fast) - ema(close_prices, slow)
        signal_line = ema(macd_line, signal)
        hist = macd_line - signal_line
        return macd_line[-1], signal_line[-1], hist[-1]

    close_prices_np = crypto_df["close_solusd"].to_numpy(dtype=float)
    macd_line_raw, signal_line_raw, _ = compute_macd_and_signal(close_prices_np)
    macd_line = safe_decimal(macd_line_raw)
    signal_line = safe_decimal(signal_line_raw)
    short_slope = safe_decimal(slope_10)
    medium_slope = safe_decimal(slope_30)
    long_slope = safe_decimal(slope_40)

    def compute_slope(series, window):
                if len(series) < window:
                    return Decimal("0.0")
                y = series[-window:]
                x = np.arange(window, dtype=float)
                m, _ = np.polyfit(x, y, 1)
                return safe_decimal(m)

    if mode == "eval":
        try:
            for sym in ["solusd", "ethusd", "btcusd"]:
                close_col = f"close_{sym}"
                high_col  = f"high_{sym}"
                low_col   = f"low_{sym}"
                vol_col   = f"volume_{sym}"
                atr_col   = f"ATR_{sym}"

                live_price = get_live_price(sym)
                crypto_df.loc[crypto_df.index[-1], close_col] = float(live_price)

                vwap_value = safe_decimal(
                    _compute_vwap(crypto_df, price_col=close_col, volume_col=vol_col)
                )

                close_np = crypto_df[close_col].to_numpy(dtype=float)
                macd_line_raw, signal_line_raw, _ = compute_macd_and_signal(close_np)

                short_slope  = compute_slope(close_np, 10)
                medium_slope = compute_slope(close_np, 30)
                long_slope   = compute_slope(close_np, 40)

                if high_col in crypto_df.columns and low_col in crypto_df.columns:
                    crypto_df[atr_col] = compute_atr(
                        crypto_df[high_col],
                        crypto_df[low_col],
                        crypto_df[close_col]
                    ).fillna(0.0)


            if narrator:
                narrator.narrate("🔄 Live alignment applied → indicators updated for SOL, ETH, BTC")

        except Exception as e:
            if narrator:
                narrator.narrate(f"⚠️ Live alignment failed: {e}")


    roi_threshold = POLICY_CONFIG.min_roi_for_confident_buy[symbol] - (confidence * Decimal("0.005"))


    DUST_THRESHOLD = POLICY_CONFIG.dust_thresholds.get(
        symbol.lower(),
        POLICY_CONFIG.dust_thresholds["solusd"]
    )

    previous_exposure = None
    if entry_info and "exposure" in entry_info:
        try:
            previous_exposure = Decimal(str(entry_info["exposure"]))
        except:
            previous_exposure = None


    if previous_exposure is not None and previous_exposure <= DUST_THRESHOLD:
        previous_exposure = None

    wallet_dec = Decimal(str(asset_balance or 0))

    exposure = None


    valid_buys_exist = (
        entry_info is not None
        and previous_exposure is not None
        and previous_exposure > DUST_THRESHOLD
    )


    anchor_flag = f"{sym_key}_anchored"
    already_anchored = getattr(portfolio_posture, anchor_flag, False)

    if action_upper != "SELL":

        if (previous_exposure is None or previous_exposure <= DUST_THRESHOLD) \
            and wallet_dec > DUST_THRESHOLD \
            and not valid_buys_exist \
            and not already_anchored:

            if narrator:
                narrator.narrate(
                    f"📘 Fresh anchor → exposure recovered from dust "
                    f"(wallet={wallet_dec}) → avg_entry={market_price_dec}"
                )

            entry_info = {
                "avg_entry_price": float(market_price_dec),
                "exposure": float(wallet_dec),
                "timestamp": datetime.utcnow().isoformat(),
            }
            exposure = wallet_dec

            setattr(portfolio_posture, anchor_flag, True)

    elif valid_buys_exist:
        exposure = previous_exposure

    if wallet_dec <= DUST_THRESHOLD:
        setattr(portfolio_posture, anchor_flag, False)
        if narrator:
            narrator.narrate("📉 Flat posture → no cost basis.")
        entry_info = None
        exposure = None

    mem_flat = (exposure is None) or (Decimal(str(exposure)) <= Decimal("0"))
    wallet_flat = (wallet_dec is None) or (wallet_dec <= DUST_THRESHOLD)
    is_flat = mem_flat and wallet_flat


    action_upper = str(action).upper()


    if narrator:
        narrator.narrate(
            f"[Policy] received_final_action={action_upper}, "
            f"Δ={forecast_decimal:.4f}, ATR={atr_value:.4f}, vol={vol_value:.4f}"
        )


    if mode == "eval" and action_upper == "SELL":

        if narrator:
            narrator.narrate("🧷 Posture lock → SELL preserved in eval mode.")
        action_upper = "SELL"



    verdict = CostBasisVerdict(
        action=action_upper,
        market_price=market_price_dec,
        avg_entry=avg_entry_price,
        exposure=exposure,
        forecast_delta=forecast_decimal,
        volatility=volatility,
        slope_value=short_slope,
        
        narrator=narrator,
    )

    micro_scalp_candidate_aggressive = is_micro_scalp_candidate_aggressive(
        sol_balance=sol_balance,
        forecast_delta=float(forecast_decimal),
        short_slope=float(short_slope),
        medium_slope=float(medium_slope),
        atr_value=float(atr_value),
        market_price=float(market_price_dec),
    )

    if narrator:
        narrator.narrate(
            f"🎯 Micro‑scalp candidate (aggressive) → {micro_scalp_candidate_aggressive}"
        )

    context = {
        "forecast_decimal": forecast_decimal,
        "confidence": confidence,
        "volatility": volatility,
        "vwap_value": vwap_value,
        "atr_value": atr_value,
        "macd_line": macd_line,
        "signal_line": signal_line,
        "action_infer": action_upper,
        "verdict": verdict,
        "suppressors": suppressors,
        "action_upper": action_upper,
        "rsi": rsi,
        "upper_bb": upper_bb,
        "lower_bb": lower_bb,
        "current_price": current_price,
        "usd_balance": usd_balance,
        "short_slope": short_slope,
        "medium_slope": medium_slope,
        "long_slope": long_slope,
        "micro_scalp_candidate_aggressive": micro_scalp_candidate_aggressive,
        
        
        

    }

    asset_balance = sol_balance
    context["sol_balance"] = asset_balance                 
    context[f"{symbol.lower()}_balance"] = asset_balance   
    context["portfolio_posture"] = portfolio_posture

    return context




def execute_trade_part2(
    context,
    position_state,
    narrator=None,
    trade_nonce_mgr=None,
    symbol="SOL",
    memory_log="scarlet_memory.csv",
    profit_log="scarlet_profit.csv",
):
    """
    Part 2: Policy-aware BUY/SELL/HOLD decision.
    - HONORS incoming action_upper from Part 1 (already gated at top level)
    - SELL: STOP‑LOSS override → cost-basis verdict → ROI gating
    - BUY: forecast-driven, micro-scalp-aware, and size-aware
    - HOLD: no execution, no BUY sizing
    """
    if context.get("forced"):
        return {
            "execute": True,
            "side": "BUY",
            "reason": "forced_buy_override",
            "safe_amount": context.get("safe_amount"),  
            "symbolic_amount": None,
        }
    if context.get("panic_exit"):
        return {
            "execute": True,
            "side": "SELL",
            "reason": "panic_exit_override",
            "safe_amount": context.get("safe_amount"),
            "symbolic_amount": None,
        }

    forecast_decimal = context["forecast_decimal"]
    verdict = context["verdict"]
    suppressors = context["suppressors"]
    action_upper = str(context["action_upper"]).upper()
    current_price = context["current_price"]

    balance_key = f"{symbol.lower()}_balance"
    asset_balance = context.get(balance_key, Decimal("0"))


    usd_balance = context["usd_balance"]
    micro_scalp_candidate_aggressive = context["micro_scalp_candidate_aggressive"]


    policy_key = f"{symbol.lower()}usd"
    current_price = context["current_price"]
    usd_balance = context["usd_balance"]

    if context.get("forced", False):
        if narrator:
            narrator.narrate(
                "⚡ Forced BUY override triggered by hotkey → bypassing suppressors and policy"
            )

        MIN_NOTIONAL = POLICY_CONFIG.min_notional.get(policy_key, Decimal("1.00"))
        BUY_FRACTION = POLICY_CONFIG.buy_fraction.get(policy_key, Decimal("0.25"))

        if usd_balance is not None:
            notional = usd_balance * BUY_FRACTION
            safe_amount = (
                Decimal("0")
                if notional < MIN_NOTIONAL
                else notional / Decimal(str(current_price))
            )
        else:
            safe_amount = Decimal("0")

        context["forced"] = False

        return {
            "execute": True,
            "side": "BUY",
            "reason": "forced_buy_hotkey",
            "safe_amount": safe_amount,
            "symbolic_amount": safe_amount,
            "suppressors": [],
        }

    if context.get("panic_exit", False):
        if narrator:
            narrator.narrate(
                "🚨 PANIC EXIT override triggered → bypassing all logic and SELLING immediately"
            )

        context["panic_exit"] = False

        return {
            "execute": True,
            "side": "SELL",
            "reason": "panic_exit_hotkey",
            "safe_amount": context.get("safe_amount", Decimal("0")),
            "symbolic_amount": None,
            "suppressors": [],
        }

    if context.get("panic_exit", False):
        if narrator:
            narrator.narrate(
                "🚨 PANIC EXIT override triggered → bypassing all logic and SELLING immediately"
            )

        context["panic_exit"] = False  

        return {
            "execute": True,
            "reason": "panic_exit_hotkey",
            "side": "SELL",
            "suppressors": [],
            "safe_amount": asset_balance,
            "symbolic_amount": None,
        }
  
    if action_upper == "HOLD":
        if narrator:
            narrator.narrate(
                "⏸ Part2 → HOLD received from policy layer → no BUY/SELL sizing, no execution."
            )
        return {
            "execute": False,
            "reason": "policy_hold",
            "side": "HOLD",
            "suppressors": suppressors,
            "safe_amount": Decimal("0"),
            "symbolic_amount": None,
        }

 
    if action_upper == "SELL":
        if narrator:
            narrator.narrate("🔻 Policy requested SELL → evaluating cost-basis verdict.")

        should_execute = verdict.should_execute
        final_reason = verdict.reason

        avg_entry = verdict.entry_price
        if avg_entry is not None and avg_entry > 0:
            entry_dec = Decimal(str(avg_entry))
            curr_dec = Decimal(str(current_price))
            roi = (curr_dec - entry_dec) / entry_dec


            if roi >= Decimal("0.0125") and forecast_delta < 0:
                if narrator:
                    narrator.narrate(
                        f"💰 Hard override → ROI {roi:.4f} ≥ 1.25% and forecast bearish "
                        f"({forecast_delta:.4f}) → FORCED SELL."
                    )
                return {
                    "execute": True,
                    "reason": "take_profit_override",
                    "side": "SELL",
                    "suppressors": suppressors,
                    "safe_amount": asset_balance,
                    "symbolic_amount": None,
                }

            stop_loss_threshold = POLICY_CONFIG.stop_loss_threshold.get(
                policy_key, POLICY_CONFIG.big_loss_threshold
            )

            if roi <= stop_loss_threshold:
                if narrator:
                    narrator.narrate(
                        f"🛑 STOP‑LOSS triggered → ROI={roi:.4f} ≤ {stop_loss_threshold:.4f}"
                    )
                return {
                    "execute": True,
                    "reason": "stop_loss_triggered",
                    "side": "SELL",
                    "suppressors": suppressors,
                    "safe_amount": asset_balance,
                    "symbolic_amount": None,
                }

            if roi <= POLICY_CONFIG.big_loss_threshold:
                if narrator:
                    narrator.narrate(
                        f"⚠️ Big-loss override → ROI={roi:.4f} ≤ {POLICY_CONFIG.big_loss_threshold:.4f}"
                    )
                should_execute = True
                final_reason = "big_loss_override"

            elif roi < POLICY_CONFIG.min_roi_for_sell:
                if narrator:
                    narrator.narrate(
                        f"🛑 SELL blocked → ROI {roi:.4f} below minimum "
                        f"{POLICY_CONFIG.min_roi_for_sell:.4f}"
                    )
                return {
                    "execute": False,
                    "reason": "roi_below_min_sell",
                    "side": "HOLD",
                    "suppressors": suppressors,
                    "safe_amount": Decimal("0"),
                    "symbolic_amount": None,
                }

        if narrator:
            narrator.narrate(
                f"📉 Cost-basis verdict → execute={should_execute}, reason={final_reason}"
            )

        return {
            "execute": should_execute,
            "reason": final_reason,
            "side": "SELL",
            "suppressors": suppressors,
            "safe_amount": asset_balance,
            "symbolic_amount": None,
        }


    if action_upper != "BUY":
        if narrator:
            narrator.narrate(
                f"⚠️ Part2 received unknown action_upper={action_upper} → no execution."
            )
        return {
            "execute": False,
            "reason": "unknown_action",
            "side": str(action_upper),
            "suppressors": suppressors,
            "safe_amount": Decimal("0"),
            "symbolic_amount": None,
        }

    MIN_NOTIONAL = POLICY_CONFIG.min_notional.get(policy_key, Decimal("1.00"))
    BUY_FRACTION = POLICY_CONFIG.buy_fraction.get(policy_key, Decimal("0.25"))

    if usd_balance is not None:
        notional = usd_balance * BUY_FRACTION
        safe_amount = (
            Decimal("0")
            if notional < MIN_NOTIONAL
            else notional / Decimal(str(current_price))
        )
    else:
        safe_amount = Decimal("0")

    symbolic_amount = None
    if micro_scalp_candidate_aggressive and safe_amount == 0:
        symbolic_amount = POLICY_CONFIG.symbolic_micro_scalp_size.get(policy_key, Decimal("0"))

    
        exchange_min = POLICY_CONFIG.exchange_minimums.get(policy_key, Decimal("0"))

        if symbolic_amount >= exchange_min:
            if narrator:
                narrator.narrate(
                    f"⚡ Symbolic micro‑scalp override → using {symbolic_amount} "
                    f"(meets exchange minimum {exchange_min})"
                )
            return {
                "execute": True,
                "reason": "symbolic_micro_scalp_override",
                "side": "BUY",
                "suppressors": suppressors,
                "safe_amount": symbolic_amount,
                "symbolic_amount": symbolic_amount,
            }
        else:
            if narrator:
                narrator.narrate(
                    f"🛑 Symbolic micro‑scalp override blocked → "
                    f"{symbolic_amount} < exchange minimum {exchange_min}"
                )


    EXCHANGE_MIN = POLICY_CONFIG.exchange_minimums.get(policy_key, Decimal("0"))

    if safe_amount < EXCHANGE_MIN:
        if narrator:
            narrator.narrate(
                f"⚠️ BUY amount {safe_amount} below exchange minimum {EXCHANGE_MIN} → skipping execution."
            )
        return {
            "execute": False,
            "reason": "below_exchange_minimum",
            "side": "BUY",
            "suppressors": suppressors,
            "safe_amount": safe_amount,
            "symbolic_amount": safe_amount,
        }

    blocking_buy = [
        s for s in suppressors
        if "@BUY" in s and not s.startswith("info_") and not s.startswith("no_")
    ]

    if blocking_buy:
        if narrator:
            narrator.narrate(
                f"🛑 BUY blocked by suppressors → {', '.join(blocking_buy)}"
            )
        return {
            "execute": False,
            "reason": ", ".join(blocking_buy),
            "side": "BUY",
            "suppressors": suppressors,
            "safe_amount": safe_amount,
            "symbolic_amount": symbolic_amount,
        }


    if micro_scalp_candidate_aggressive:
        if narrator:
            narrator.narrate("🎯 Micro-scalp BUY candidate (aggressive) → ALLOWED")
        return {
            "execute": True,
            "reason": "micro_scalp_candidate_aggressive",
            "side": "BUY",
            "suppressors": suppressors,
            "safe_amount": safe_amount,
            "symbolic_amount": safe_amount,
        }


    if forecast_decimal > POLICY_CONFIG.buy_delta_threshold:
        if narrator:
            narrator.narrate(
                f"📈 BUY allowed → Δ={forecast_decimal:.4f} exceeds threshold"
            )
        return {
            "execute": True,
            "reason": "forecast_supports_buy",
            "side": "BUY",
            "suppressors": suppressors,
            "safe_amount": safe_amount,
            "symbolic_amount": safe_amount,
        }

 
    if narrator:
        narrator.narrate(
            f"🛑 BUY suppressed → Δ={forecast_decimal:.4f} below threshold"
        )

    return {
        "execute": False,
        "reason": "forecast_insufficient",
        "side": "BUY",
        "suppressors": suppressors,
        "safe_amount": safe_amount,
        "symbolic_amount": safe_amount,
    }
from decimal import Decimal
import json
import os


def new_func(action_upper, market_price, memory_log, narrator, mode, response, executed_amount, symbol):
    try:
        sym_key = symbol.lower()

     
        if mode == "eval" and action_upper == "BUY":
            avg_execution_price = Decimal(str(response.get("avg_execution_price", market_price)))

            anchor = EntryAnchor(
                price=avg_execution_price,
                amount=executed_amount,
                symbol=sym_key,
                action="BUY",
                trade_filled="1",
                exit_note="real_entry_anchor"
            )

            row = anchor.to_dict()
            log_trade_fill(memory_log, row)

            with open(memory_log, "a") as f:
                f.flush()
                os.fsync(f.fileno())

            if narrator:
                narrator.narrate(
                    f"🟢 BUY anchor logged @ {avg_execution_price} for {executed_amount} {sym_key.upper()}"
                )

            return {
                "execute": True,
                "action": "BUY",
                "symbol": sym_key,
                "price": float(avg_execution_price),
                "amount": float(executed_amount),
                "row": row,
                "avg_entry_price": float(avg_execution_price),
                "exposure_delta": float(executed_amount),
            }

  
        if mode == "eval" and action_upper == "SELL":
            sell_price = Decimal(str(response.get("avg_execution_price", market_price)))

            anchor = EntryAnchor(
                price=sell_price,
                amount=executed_amount,
                symbol=sym_key,
                action="SELL",
                trade_filled="1",
                exit_note=resolve_exit_note(response, market_price)
            )

            row = anchor.to_dict()
            log_trade_fill(memory_log, row)

            with open(memory_log, "a") as f:
                f.flush()
                os.fsync(f.fileno())

            if narrator:
                narrator.narrate(
                    f"🔻 SELL anchor logged @ {sell_price} for {executed_amount} {sym_key.upper()}"
                )

            return {
                "execute": True,
                "action": "SELL",
                "symbol": sym_key,
                "price": float(sell_price),
                "amount": float(executed_amount),
                "row": row,
                "avg_entry_price": None,
                "exposure_delta": -float(executed_amount),
            }

  
        if not response.get("execute", False):
            if narrator:
                narrator.narrate("🛑 Trade not logged — execution failed or rejected.")

            return {
                "execute": False,
                "action": action_upper,
                "symbol": sym_key,
                "reason": "execution_failed_or_rejected",
            }

    except Exception as e:
        if narrator:
            narrator.narrate(f"⚠️ Trade execution failed — exception caught: {str(e)}")

        return {
            "execute": False,
            "action": action_upper,
            "symbol": sym_key,
            "reason": "exception_in_execute",
        }

def execute_trade_part3(
    roi: Decimal,
    final_execute: bool,
    trade_reason: str,
    safe_amount: Decimal,
    action_upper: str,
    market_price: Decimal,
    override_triggered: bool,
    symbol: str,
    trade_nonce_mgr,
    memory_log: str,
    symbolic_amount: Decimal | None = None,
    narrator=None,
    mode="eval",
):
    import pandas as pd
    from datetime import datetime
    from decimal import Decimal


    response = {}

    executed_amount = Decimal("0")
    final_price = Decimal(str(market_price))
    sym_key = symbol.lower()

    if final_execute:

   
        if action_upper == "BUY":
            if symbolic_amount and symbolic_amount > 0:
                dynamic_amt = symbolic_amount
                if narrator:
                    narrator.narrate(
                        f"⚡ Using symbolic/micro‑scalp BUY amount {dynamic_amt} {sym_key.upper()} "
                        f"(safe_amount={safe_amount})."
                    )
            elif safe_amount > 0:
                dynamic_amt = safe_amount
            else:
                if narrator:
                    narrator.narrate(
                        "🛑 BUY suppressed — safe_amount is zero and no symbolic override provided."
                    )
                return {
                    "execute": False,
                    "reason": "no_buy_size",
                    "action": "BUY",
                    "amount": 0.0,
                    "price": float(final_price),
                    "override": override_triggered,
                }

  
            if sym_key == "btcusd":
                dynamic_amt = dynamic_amt.quantize(Decimal("0.00000001"))   # 8 decimals
            elif sym_key == "ethusd":
                dynamic_amt = dynamic_amt.quantize(Decimal("0.0000001"))    # 7 decimals
            else:
                dynamic_amt = dynamic_amt.quantize(Decimal("0.0001"))       # 4 decimals

        elif action_upper == "SELL":
            if safe_amount > 0:
                dynamic_amt = safe_amount
            else:
                if narrator:
                    narrator.narrate("🛑 SELL blocked — no balance available.")
                return {
                    "execute": False,
                    "reason": "no_balance",
                    "action": "SELL",
                    "amount": 0.0,
                    "price": float(final_price),
                    "override": override_triggered,
                }

        else:
            if narrator:
                narrator.narrate(f"⚠️ Unknown action: {action_upper}")
            return {
                "execute": False,
                "reason": "unknown_action",
                "action": action_upper,
                "amount": 0.0,
                "price": float(final_price),
                "override": override_triggered,
            }

        payload = f"{action_upper} {dynamic_amt:.8f} @ ${final_price:.2f}"
        if narrator:
            narrator.narrate(f"📦 Final Gemini Payload → {payload}")

        response = {
            "action": action_upper,
            "amount": float(dynamic_amt),
            "price": float(final_price),
            "reason": trade_reason,
            "override": override_triggered,
            "execute": True,
        }

        if DISABLE_TRADING:
            if narrator:
                narrator.narrate("🚫 Trading disabled — skipping execution.")
            return {
                "execute": False,
                "reason": "trading_disabled",
                "action": action_upper,
                "amount": float(dynamic_amt),
                "price": float(final_price),
                "override": override_triggered,
            }

        try:
            raw_resp = execute_trade_gemini(
                symbol=sym_key,
                amount=str(dynamic_amt),
                action=action_upper.lower(),
                nonce_mgr=trade_nonce_mgr,
                override=override_triggered,
            )


            if isinstance(raw_resp, dict):
                response.update(raw_resp)
            else:
                if narrator:
                    narrator.narrate(
                        "⚠️ Trade API returned no payload or non-dict — keeping local response."
                    )

            executed_amount = Decimal(str(response.get("executed_amount", "0")))
            trade_filled = executed_amount > 0

            if trade_filled:
                try:
                    new_func(
                        action_upper,
                        market_price,
                        memory_log,
                        narrator,
                        mode,
                        response,
                        executed_amount,
                        symbol=sym_key,
                    )
                except Exception as e:
                    if narrator:
                        narrator.narrate(f"⚠️ Anchor logging failed: {e}")

                try:
                    trade_row = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "action": action_upper,
                        "entry_price": float(final_price),
                        "exit_price": None,
                        "dynamic_amount": float(executed_amount),
                        "market_price": float(final_price),
                        "profit": None,
                        "trade_filled": True,
                        "exit_note": "",
                        "side": "long" if action_upper == "BUY" else "sell",
                        "reset_baseline": "",
                        "price": float(final_price),
                        "amount": float(executed_amount),
                        "symbol": sym_key,
                        "reason": trade_reason,
                    }



                    if os.path.exists(memory_log):
                        df_trades = pd.read_csv(memory_log, on_bad_lines="skip")
                        df_trades = pd.concat(
                            [df_trades, pd.DataFrame([trade_row])],
                            ignore_index=True,
                        )
                    else:
                        df_trades = pd.DataFrame([trade_row])

                    df_trades.to_csv(memory_log, index=False)

                except Exception as e:
                    if narrator:
                        narrator.narrate(f"⚠️ Trade result logging failed: {e}")

                if action_upper == "BUY":
                    try:
                        df = pd.read_csv(memory_log)
                        entry_info = get_entry_info(df, symbol=sym_key)

                        if narrator:
                            narrator.narrate(
                                f"📘 Updated exposure={entry_info['exposure']}, "
                                f"avg_entry={entry_info['avg_entry_price']}"
                            )
                    except Exception as e:
                        if narrator:
                            narrator.narrate(f"⚠️ Failed to update BUY posture: {e}")


                if action_upper == "SELL":
                    try:
                        df = pd.read_csv(memory_log)
                        entry_info = get_entry_info(df, symbol=sym_key)
                        exposure_after_sell = Decimal(str(entry_info["exposure"]))

                        if exposure_after_sell <= Decimal("0.0001"):
                            if narrator:
                                narrator.narrate(
                                    f"🧹 Exposure dropped to dust ({exposure_after_sell}) "
                                    f"→ inserting RESET anchor for {sym_key}."
                                )

                            reset_row = {
                                "action": "RESET",
                                "symbol": sym_key,
                                "reset_baseline": float(market_price),
                                "price": float(market_price),
                                "amount": 0.0,
                                "trade_filled": True,
                                "exit_note": "auto_reset_on_dust",
                                "timestamp": datetime.utcnow().isoformat(),
                            }

                            df = pd.concat([df, pd.DataFrame([reset_row])], ignore_index=True)
                            df.to_csv(memory_log, index=False)

                    except Exception as e:
                        if narrator:
                            narrator.narrate(f"⚠️ Auto-reset failed: {e}")

            else:
   
                if narrator:
                    narrator.narrate(
                        f"🧭 Final Verdict → execute=False, reason={trade_reason}"
                    )

                response = {
                    "execute": False,
                    "reason": trade_reason,
                    "action": action_upper,
                    "amount": float(safe_amount),
                    "price": float(final_price),
                    "override": override_triggered,
                }

        except Exception as e:
            if narrator:
                narrator.narrate(f"⚠️ Trade execution failed — exception caught: {str(e)}")
            response = {
                "execute": False,
                "reason": "exception_in_execute",
                "action": action_upper,
                "amount": float(safe_amount),
                "price": float(final_price),
                "override": override_triggered,
            }

        return response

    return {
        "execute": True or False,
        "action": action_upper,
        "symbol": symbol.lower(),
        "amount": float(executed_amount),
        "price": float(final_price),
        "override": override_triggered,

  
        "avg_entry_price": float(final_price) if action_upper == "BUY" else None,
        "exposure_delta": float(executed_amount) if action_upper == "BUY" else -float(executed_amount),
        "trade_filled": (response or {}).get("trade_filled", False),
    }




def log_trade_anchor_json(memory_log_path, anchor_dict):
    """Append a trade anchor to the memory log as a JSON line."""
    try:
        with open(memory_log_path, "a") as f:
            f.write(json.dumps(anchor_dict) + "\n")
    except Exception as e:
        print(f"[TradeLog] Failed to write anchor: {e}")




def execute_trade(
    model_forecast,
    market_price,
    action_upper,
    symbol,
    price_tensor,
    avg_entry_price,
    crypto_df,
    scaler,
    trade_nonce_mgr,
    override_triggered=False,
    portfolio_posture=None,
    suppressors=None,
    amount=None,
    adaptive_threshold=Decimal("0.01"),
    latest_volatility=Decimal("0"),
    profit_log="scarlet_profit.csv",
    memory_log="scarlet_memory.csv",
    forecast_np=None,
    entry_info=None,
    forecast_strength=None,
    forecast_delta=None,   
    price_series=None,
    roi_reason=None,
    slope_10=None,
    slope_30=None,
    slope_40=None,
    position_state=None,
    sol_balance=None,
    usd_balance=None,
    asset_balance=None,
    mode="eval",
    narrator=None,
    previous_close=None,
    context=None,
    flat_score=None,
):

    """
    Orchestrates the full trade cycle: indicators, ROI verdict, suppressors, execution.
    Returns a structured response dict with forecast_delta, actual_delta, and reward
    for live training.
    """

  
    if context is not None and context.get("forced"):
        action_upper = "BUY"

    part1_ctx = execute_trade_part1(
        base=sym.replace("usd", "").upper(),
        model_forecast=model_forecast,
        market_price=market_price,
        action=action_upper,
        symbol=sym,                    
        price_tensor=price_tensor,
        asset_balance=asset_balance,
        avg_entry_price=avg_entry_price,
        crypto_df=crypto_df,
        scaler=scaler,
        trade_nonce_mgr=trade_nonce_mgr,
        override_triggered=override_triggered,
        suppressors=suppressors,
        amount=amount,
        adaptive_threshold=adaptive_threshold,
        latest_volatility=latest_volatility,
        profit_log=profit_log,
        memory_log=memory_log,
        forecast_np=forecast_np,
        entry_info=entry_info,
        forecast_strength=forecast_strength,
        price_series=price_series,
        roi_reason=roi_reason,
        slope_10=slope_10,
        slope_30=slope_30,
        slope_40=slope_40,
        flat_score=flat_score,
        narrator=narrator,
        mode=mode,
        sol_balance=sol_balance,
        usd_balance=usd_balance,
        portfolio_posture=portfolio_posture,
    )
    

   
    if context is not None:

        part1_ctx["hotkey_context"] = context

        if context.get("forced"):
            part1_ctx["forced"] = True
            part1_ctx["forced_symbol"] = context.get("forced_symbol")

        if context.get("panic_exit"):
            part1_ctx["panic_exit"] = True


    part2_ctx = execute_trade_part2(
        context=part1_ctx,
        position_state=position_state,
        narrator=narrator,
        trade_nonce_mgr=trade_nonce_mgr,
        symbol=symbol,
        memory_log=memory_log,
        profit_log=profit_log,
    )

   
    policy_key = f"{symbol.lower()}usd"

    DUST_THRESHOLD = POLICY_CONFIG.dust_thresholds.get(policy_key, Decimal("0.00001"))

    balance_key = f"{symbol.lower()}_balance"
    sol_balance_ctx = part1_ctx.get(balance_key)

    entry_info_ctx = part1_ctx.get("entry_info") or entry_info or {}
    exposure = entry_info_ctx.get("exposure")

    flat_by_wallet = (
        sol_balance_ctx is not None
        and Decimal(str(sol_balance_ctx)) <= DUST_THRESHOLD
    )
    flat_by_exposure = (
        exposure is not None
        and Decimal(str(exposure)) <= Decimal("0")
    )

   
    from Scarlet_Core import cooldown_cycles 

    if part2_ctx["side"] == "SELL" and part2_ctx["execute"]:
        try:
            entry_price = Decimal(str(avg_entry_price)) if avg_entry_price is not None else None
            exit_price = Decimal(str(part1_ctx["current_price"]))

            if entry_price is not None and entry_price > 0:
                realized_roi = (exit_price - entry_price) / entry_price
            else:
                realized_roi = Decimal("0")

           
            if realized_roi <= POLICY_CONFIG.big_loss_threshold:
                cooldown_cycles[symbol] = POLICY_CONFIG.cooldown_cycles_after_big_loss
                if narrator:
                    narrator.narrate(
                        f"🧊 {symbol.upper()} big-loss cooldown → "
                        f"{POLICY_CONFIG.cooldown_cycles_after_big_loss} cycles "
                        f"(ROI={realized_roi:.4f})"
                    )

            else:
                cooldown_cycles[symbol] = POLICY_CONFIG.cooldown_cycles_after_sell
                if narrator:
                    narrator.narrate(
                        f"🧊 {symbol.upper()} post-SELL cooldown → "
                        f"{POLICY_CONFIG.cooldown_cycles_after_sell} cycles "
                        f"(ROI={realized_roi:.4f})"
                    )

        except Exception as e:
            if narrator:
                narrator.narrate(f"⚠️ Failed to apply cooldown logic: {e}")

 
    symbolic_amount = part2_ctx.get("symbolic_amount") 

    if "safe_amount" in part2_ctx and part2_ctx["safe_amount"] is not None:
        final_safe_amount = Decimal(str(part2_ctx["safe_amount"]))
    elif amount is not None:
        final_safe_amount = Decimal(str(amount))
    else:
        final_safe_amount = Decimal("0")


    response = execute_trade_part3(
        roi=Decimal("0"),
        final_execute=part2_ctx["execute"],
        trade_reason=part2_ctx["reason"],
        safe_amount=final_safe_amount,
        action_upper=part2_ctx["side"],
        market_price=part1_ctx["current_price"],
        override_triggered=override_triggered,
        symbol=symbol,
        trade_nonce_mgr=trade_nonce_mgr,
        memory_log=memory_log,
        symbolic_amount=symbolic_amount,
        narrator=narrator,
    )

    if response is None or not isinstance(response, dict):
        if narrator:
            narrator.narrate(
                "🧯 execute_trade_part3 returned no payload or non-dict — applying safe default."
            )
        response = {
            "execute": False,
            "reason": "null_or_invalid_response",
            "action": part2_ctx.get("side", str(action_upper).upper()),
            "amount": float(final_safe_amount),
            "price": float(part1_ctx.get("current_price", market_price)),
            "override": override_triggered,
        }

    forecast_delta = None
    actual_delta = None
    reward = 0.0


    if forecast_delta is None:
        try:
            forecast_decimal = part1_ctx.get("forecast_decimal", Decimal("0.0"))
            forecast_delta = float(forecast_decimal)
        except Exception:
            forecast_delta = None


    try:
        anchor_price = Decimal(str(part1_ctx["current_price"]))
    except Exception:
        anchor_price = None

   
    if mode == "eval":
        try:
            if anchor_price is not None and forecast_delta is not None:
                timestamp = datetime.utcnow().isoformat()
                with open(memory_log, "a") as f:
                    f.write(
                        f"{timestamp},FORECAST_ANCHOR,{float(anchor_price):.6f},{forecast_delta:.6f}\n"
                    )
        except Exception as e:
            if narrator:
                narrator.narrate(f"⚠️ Failed to log forecast anchor: {e}")

    actual_delta = None
    reward = 0.0

    if (
        part2_ctx["side"] == "SELL"
        and part2_ctx["execute"]
        and avg_entry_price is not None
    ):
        try:
            entry_price = Decimal(str(avg_entry_price))
            exit_price = Decimal(str(part1_ctx["current_price"]))

            if entry_price > 0:
                actual_delta_dec = (exit_price - entry_price) / entry_price
                actual_delta = float(actual_delta_dec)

                if forecast_delta is not None:
                    reward = forecast_delta * actual_delta

                if mode == "eval":
                    timestamp = datetime.utcnow().isoformat()
                    with open(memory_log, "a") as f:
                        f.write(
                            f"{timestamp},ROI_EVENT,{float(exit_price):.6f},{actual_delta:.6f},{reward:.6f}\n"
                        )
                    if narrator:
                        narrator.narrate(
                            f"📊 ROI event → entry={entry_price:.4f}, exit={exit_price:.4f}, "
                            f"Δ_actual={actual_delta_dec:.4f}, Δ_forecast={forecast_decimal:.4f}, "
                            f"reward={reward:.6f}"
                        )
        except Exception as e:
            if narrator:
                narrator.narrate(f"⚠️ Failed to compute ROI event: {e}")
            actual_delta = None


    response["forecast_delta"] = forecast_delta
    response["actual_delta"] = actual_delta
    response["reward"] = reward
    response["regime"] = part1_ctx.get("regime")
    response["flat_score"] = part1_ctx.get("flat_score")

    return response       

import time
import secrets
import jwt
from cryptography.hazmat.primitives import serialization
from dotenv import load_dotenv



import smtplib
from email.mime.text import MIMEText

import smtplib
import smtplib
from email.mime.text import MIMEText


emails_sent = 0
last_window_start = 0

from hybrid_forecaster import HybridBalancer













def get_market_price(symbol="solusd", return_history=False):
    """Fetch latest market price or historical candles from Gemini API."""

 
    if return_history:
        candles_df = load_market_data(symbol)
        if candles_df.empty:
            print("📉 No historical data retrieved — skipping Bollinger diagnostic.")
            return []
        return candles_df["close"].tail(30).tolist()


    url = f"https://api.gemini.com/v1/pubticker/{symbol}"
    response = fetch_market_data(url)

    


    if not isinstance(response, dict):
        print("⚠️ Invalid API response type — expected dict.")
        return 0.0

    raw_last = response.get("last")
    if raw_last is None:
        print("⚠️ Missing 'last' field in API response.")
        return 0.0


    try:
        price = float(raw_last)
        if price != price or abs(price) == float("inf"):  # NaN or inf
            raise ValueError("Non-finite price")
    except Exception:
        print(f"⚠️ Failed to parse price from response: {raw_last}")
        return 0.0

    print(f"💲 Market Price: {price}")
    return price



import requests

def fetch_market_data(url):
    """Fetch market data from the specified API endpoint with safe fallbacks."""
    try:

        response = requests.get(url, timeout=5)


        response.raise_for_status()


        try:
            data = response.json()
        except ValueError:
            print("❌ Failed to decode JSON from market data response.")
            return {}

        
        return data

    except requests.exceptions.Timeout:
        print("⏳ Market data request timed out.")
        return {}

    except requests.exceptions.ConnectionError as e:
        print(f"🌐 Connection error while fetching market data: {e}")
        return {}

    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP error while fetching market data: {e}")
        return {}

    except requests.exceptions.RequestException as e:
        print(f"❌ General request failure: {e}")
        return {}


def get_market_volume(symbol="solusd"):
    """Fetch 24h trading volume from Gemini public API."""
    url = f"https://api.gemini.com/v1/pubticker/{symbol}"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()

        try:
            data = response.json()
        except ValueError:
            print("❌ Failed to decode JSON from volume response.")
            return None

        volume_block = data.get("volume", {})
        if not isinstance(volume_block, dict):
            print("❌ Volume field missing or malformed.")
            return None

 
        base_asset = symbol.replace("usd", "").upper()

        raw_vol = volume_block.get(base_asset)
        if raw_vol is None:
            print(f"⚠️ No volume field for asset '{base_asset}' in response.")
            return None

        try:
            vol = float(raw_vol)
            if vol != vol or abs(vol) == float("inf"):  
                raise ValueError("Non-finite volume")
            return vol
        except Exception:
            print(f"⚠️ Invalid volume value: {raw_vol}")
            return None

    except requests.exceptions.Timeout:
        print("⏳ Volume request timed out.")
        return None

    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to fetch volume: {e}")
        return None



import requests



    
    



import jwt
from cryptography.hazmat.primitives import serialization
import time
import secrets
import requests
import os



import base64
import json


import schedule
import time



def log_reset(memory_path, current_price, rebase=False):
    """
    Safe RESET / RESET_REBASE logger.
    Prevents wiping valid BUYs and ensures full-schema consistency.
    """

    import pandas as pd
    import os
    from datetime import datetime

    action = "RESET_REBASE" if rebase else "RESET"

    try:
        price_val = float(current_price)
    except Exception:
        price_val = None

    schema = [
        "timestamp", "action",
        "entry_price", "exit_price",
        "dynamic_amount", "market_price",
        "profit", "trade_filled",
        "exit_note", "side",
        "reset_baseline",
        "price", "amount",
        "symbol", "reason"
    ]

    if os.path.exists(memory_path):
        try:
            df = pd.read_csv(memory_path, on_bad_lines="skip")
        except Exception:
            df = pd.DataFrame(columns=schema)
    else:
        df = pd.DataFrame(columns=schema)

    for col in schema:
        if col not in df.columns:
            df[col] = None

    if not df.empty:
        last_action = str(df.iloc[-1]["action"]).upper()
        if last_action in ("RESET", "RESET_REBASE"):
            return df

    if not df.empty:

        reset_idx = df[df["action"].isin(["RESET", "RESET_REBASE"])].index
        last_reset_idx = reset_idx[-1] if len(reset_idx) > 0 else -1

        buys_after_reset = df[
            (df.index > last_reset_idx) &
            (df["action"] == "BUY") &
            (df["trade_filled"].astype(str) == "1")
        ]

        if not buys_after_reset.empty:

            return df


    reset_row = {
        "timestamp": datetime.utcnow().isoformat(),
        "action": action,
        "entry_price": None,
        "exit_price": None,
        "dynamic_amount": None,
        "market_price": price_val,
        "profit": 0.0,
        "trade_filled": False,
        "exit_note": "log_reset",
        "side": "",
        "reset_baseline": price_val,
        "price": None,
        "amount": None,
        "symbol": None,
        "reason": "reset_rebase" if rebase else "reset",
    }

    df = pd.concat([df, pd.DataFrame([reset_row])], ignore_index=True)

    df.to_csv(memory_path, index=False)
    return df

class LossScheduler:
    def __init__(
        self,
        rl_weight_start=0.5,
        rl_weight_max=2.0,
        rl_weight_growth=1.07,
        sched_sampling_start=0.3,
        sched_sampling_end=0.1,
        sched_sampling_decay=0.995,
        supervised_batches_per_epoch=10,
        min_supervised_batches=5,
        supervised_decay=0.995,
        reward_ramp=0.05,
        plateau_threshold=0.01,
        stagnation_epochs_trigger=3,
        stagnation_window=3,
    ):
    
        self.rl_weight_start = rl_weight_start
        self.rl_weight_max = rl_weight_max
        self.rl_weight_growth = rl_weight_growth

        self.sched_sampling_start = sched_sampling_start
        self.sched_sampling_end = sched_sampling_end
        self.sched_sampling_decay = sched_sampling_decay

        self.supervised_batches_per_epoch = supervised_batches_per_epoch
        self.min_supervised_batches = min_supervised_batches
        self.supervised_decay = supervised_decay

  
        self.reward_ramp = reward_ramp

      
        self.plateau_threshold = plateau_threshold
        self.stagnation_epochs_trigger = stagnation_epochs_trigger
        self.stagnation_window = stagnation_window
        self.val_loss_history = deque(maxlen=stagnation_window)
        self.stagnation_counter = 0

   
        self.rl_batches_this_epoch = 0
        self.supervised_batches_this_epoch = 0
        self.min_rl_ratio = 0.5

      
        self.current_supervised_batches = supervised_batches_per_epoch
        self.base_lr = None 

    def enforce_lr_floor(self, optimizer, min_lr=5e-7):
        for pg in optimizer.param_groups:
            if pg["lr"] < min_lr:
                pg["lr"] = min_lr
                return f"🧘 Learning rate floor enforced → {min_lr:.6f}"
        return None

    def record_supervised_batch(self):
        self.supervised_batches_this_epoch += 1

    def record_rl_batch(self):
        self.rl_batches_this_epoch += 1

    def rl_ratio_met(self):
        sup = self.supervised_batches_this_epoch
        if sup == 0:
            return True
        return self.rl_batches_this_epoch >= sup * self.min_rl_ratio


    def update(self, epoch, val_loss=None, prev_val_loss=None):


        rl_weight = min(
            self.rl_weight_start * (self.rl_weight_growth ** epoch),
            self.rl_weight_max,
        )

        if val_loss is not None:
            self.val_loss_history.append(float(val_loss))

            if len(self.val_loss_history) == self.stagnation_window:
                window_range = max(self.val_loss_history) - min(self.val_loss_history)

                if window_range < self.plateau_threshold:
                    self.stagnation_counter += 1
                    rl_weight = min(rl_weight * 1.1, self.rl_weight_max)
                else:
     
                    self.stagnation_counter = max(0, self.stagnation_counter - 1)

  
        sched_sampling_prob = max(
            self.sched_sampling_end,
            self.sched_sampling_start * (self.sched_sampling_decay ** epoch),
        )

        supervised_batches = max(
            self.min_supervised_batches,
            int(self.supervised_batches_per_epoch * (self.supervised_decay ** epoch)),
        )
        self.current_supervised_batches = supervised_batches

        return rl_weight, sched_sampling_prob, supervised_batches


    def is_supervised_batch(self, batch_idx: int) -> bool:
        return batch_idx < self.current_supervised_batches


    def reward_multiplier(self, epoch, val_loss=None, prev_val_loss=None):


        base = 1.0 + self.reward_ramp * epoch
        if self.stagnation_counter >= self.stagnation_epochs_trigger:
            base *= 1.2
        return base


   

    def override_rl_weight(self):
        if self.stagnation_counter >= self.stagnation_epochs_trigger:
            return self.rl_weight_max
        return None

    def narrate_batch_balance(self):
        if not self.rl_ratio_met():
            return (
                f"⚠️ RL underrepresented → "
                f"{self.rl_batches_this_epoch} RL vs {self.supervised_batches_this_epoch} supervised. "
                f"Extending RL phase."
            )
        return (
            
        )

  
    def check_mid_epoch_balance(self, batch_idx: int, total_batches: int):
        halfway = total_batches // 2
        if batch_idx == halfway and not self.rl_ratio_met():
            return self.narrate_batch_balance()
        return None
    
    def set_base_lr(self, lr):
        self.base_lr = lr



import requests
import pandas as pd
import numpy as np
import torch




def compute_rsi(series, window=70):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
 
    return rsi.fillna(50)

def compute_bollinger(series, window=70, num_std=2):
    mid = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = upper - lower
    return mid, upper, lower, width


def compute_atr(high, low, close, window=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean().fillna(0)

def scale_live_features(df, scaler, numeric_inputs):
    """
    Apply the trained RobustScaler to the numeric input columns only.
    Ensures column presence, correct order, and safe numeric conversion.
    """

 
    missing = [col for col in numeric_inputs if col not in df.columns]
    if missing:
        raise KeyError(f"Missing numeric input columns: {missing}")


    block = df[numeric_inputs]

  
    try:
        block = block.astype("float32")
    except Exception as e:
        raise ValueError(f"Failed to convert live features to float32: {e}")

    if block.isnull().any().any():
        raise ValueError(f"NaNs detected in live numeric features: {block}")

    try:
        scaled = scaler.transform(block)
    except Exception as e:
        raise RuntimeError(f"Scaler transform failed: {e}")

   
    df[numeric_inputs] = scaled

    return df



import praw
reddit = praw.Reddit(
    client_id="e0dyJ5GZbvKoQtPHBcBObQ",
    client_secret="bxGuX7b5QutphmYF7gy7IutnsSHyCg",
    user_agent="sentiment_api by u/Honest_Mood",
    username="Honest_Mood",
    password="Marten4321!",
)


def ema(values, alpha):
    """Compute exponential moving average over a list of floats."""
    if not values:
        return 0.0
    ema_val = values[0]
    for v in values[1:]:
        ema_val = alpha * v + (1 - alpha) * ema_val
    return ema_val


def compute_long_term_sentiment(history, half_life_hours=12):
    """
    Compute long-term sentiment using a time-decayed EMA over a 3-day window.
    history: list of (timestamp, sentiment)
    """

    if not history:
        return 0.0

  
    history = sorted(history, key=lambda x: x[0])

    now = time.time()
    filtered = []

    for ts, s in history:
        age_hours = (now - ts) / 3600.0
        if age_hours <= 72:
            filtered.append((age_hours, s))

    if not filtered:
        return 0.0


    alpha = 1 - math.exp(-math.log(2) / half_life_hours)

 
    ema_value = filtered[0][1]  

    for age_hours, s in filtered[1:]:
       
        decay = math.exp(-math.log(2) * (age_hours / half_life_hours))
        ema_value = decay * ema_value + (1 - decay) * s

    return float(ema_value)
import threading
import time

from collections import deque
import time


from collections import deque
import time
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


SENTIMENT_HISTORY = deque(maxlen=15000)


_last_sentiment_fetch = 0
_last_sentiment_value = 0.0


import time
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification




SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
sentiment_model.to(device)
sentiment_model.eval()

# --- Weighted subreddits ---
SENTIMENT_CONFIG = {
    "SOL": {
        "subreddits": {
            "solana": 0.50,
            "sol": 0.25,
            "cryptomarkets": 0.1,
            "cryptomarket": 0.1,
            "crypto": 0.05,
            "cryptocurrency": 0.05,
        },
        "keywords": [
            "solana",
            "sol",
            "sol crypto",
            "solana crypto",
            "crypto",
            "cryptocurrency",
        ],
    },

    "ETH": {
        "subreddits": {
            "ethereum": 0.50,
            "ethtrader": 0.30,
        },
        "keywords": [
            "ethereum",
            "eth",
            "eth crypto",
            "ethereum crypto",
            "crypto",
            "cryptocurrency",
        ],
    },

    "BTC": {
        "subreddits": {
            "bitcoin": 0.50,
            "btc": 0.30,
        },
        "keywords": [
            "bitcoin",
            "btc",
            "btc crypto",
            "bitcoin crypto",
            "cryptocurrency",
            "crypto",
        ],
    },
}

def compute_sentiment_gpu(text_list, batch_size=32):
    """
    Compute sentiment scores for a list of text posts using GPU acceleration.
    Returns a list of sentiment values in [-1, 1].
    """

    if not text_list:
        return []

    scores = []

    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]

 
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )


        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = sentiment_model(**inputs)

        logits = outputs.logits.detach().cpu()
        probs = torch.softmax(logits, dim=-1)

        for p in probs:
            neg, neu, pos = p.tolist()
            score = pos - neg  # map to [-1, 1]
            scores.append(score)

    return scores

import os
import json
import time
from collections import deque

CACHE_PATH = r"D:\Scarlet_Works\Scarlet\sentiment_cache.json"




def load_sentiment_cache():
    global _last_sentiment_fetch, _last_sentiment_value, SENTIMENT_HISTORY

    print("[Sentiment] Attempting to load cache...")   

    if not os.path.exists(CACHE_PATH):
        print("[Sentiment] No cache file found.")
        return

    try:
        with open(CACHE_PATH, "r") as f:
            data = json.load(f)

        print("[Sentiment] Raw cache keys:", data.keys())  

        _last_sentiment_fetch = data.get("last_fetch", 0)
        _last_sentiment_value = data.get("last_value", 0.0)

        history = data.get("history", [])
        print("[Sentiment] History entries in file:", len(history)) 

        for t, s in history:
            try:
                ts = float(t)
                SENTIMENT_HISTORY.append((ts, float(s)))
            except Exception:
                continue


        print(f"[Sentiment] Cache loaded → {len(SENTIMENT_HISTORY)} history points")

    except Exception as e:
        print(f"[Sentiment] Failed to load cache: {e}")
    _last_sentiment_fetch = 0 





def save_sentiment_cache(called_from_sentiment=False):
    if not called_from_sentiment:
        print("[Sentiment] Save blocked — only sentiment loop may save")
        return

    try:
        import time
        ONE_WEEK_SECONDS = 64 * 24 * 60 * 60
        now = time.time()

        trimmed_history = [
            (t, s)
            for t, s in SENTIMENT_HISTORY
            if now - float(t) <= ONE_WEEK_SECONDS
        ]

        data = {
            "last_fetch": _last_sentiment_fetch,
            "last_value": _last_sentiment_value,
            "history": [(float(t), float(s)) for t, s in trimmed_history],
        }

        with open(CACHE_PATH, "w") as f:
            json.dump(data, f)

        print(f"[Sentiment] Cache saved to disk ({len(trimmed_history)} entries, 64 day limit)")

        if len(trimmed_history) != len(SENTIMENT_HISTORY):
            SENTIMENT_HISTORY.clear()
            SENTIMENT_HISTORY.extend(trimmed_history)

    except Exception as e:
        print(f"[Sentiment] Failed to save cache: {e}")


def fetch_weighted_reddit_sentiment(limit=99):

    

    global _last_sentiment_fetch, _last_sentiment_value

    now = time.time()

    if now - _last_sentiment_fetch < 61:
        return _last_sentiment_value

    total = 0.0
    weight_sum = 0.0
    total_posts_fetched = 0

    for asset, cfg in SENTIMENT_CONFIG.items():
        subs = cfg["subreddits"]
        keywords = cfg["keywords"]

        for sub, w in subs.items():
            try:
                subreddit = reddit.subreddit(sub)
                collected = []

                for kw in keywords:
                    posts = subreddit.search(
                        kw, sort="new", time_filter="hour", limit=limit
                    )
                    for post in posts:
                        collected.append(post.title)
                        if hasattr(post, "selftext"):
                            collected.append(post.selftext)

                
                total_posts_fetched += len(collected)

                if not collected:
                    continue

                scores = compute_sentiment_gpu(collected)
                if not scores:
                    continue

                avg_score = sum(scores) / len(scores)

                total += avg_score * w
                weight_sum += w

            except Exception:
                continue

 
    STS = total / weight_sum if weight_sum > 0 else 0.0


    LTS = compute_long_term_sentiment(SENTIMENT_HISTORY)

    tau_hours = 6.0
    alpha = math.exp(-1.0 / tau_hours)

    blended = alpha * STS + (1 - alpha) * LTS

    _last_sentiment_fetch = now
    _last_sentiment_value = blended

    SENTIMENT_HISTORY.append((now, blended))
    save_sentiment_cache(called_from_sentiment=True)

    return blended



def get_sentiment_series(minutes=4320):
    """Return sentiment values from the last N minutes (default: 3 days)."""
    cutoff = time.time() - (minutes * 60)
    return [s for t, s in SENTIMENT_HISTORY if t >= cutoff]


def compute_sentiment_slopes_from_history():
    """
    Compute sentiment slopes for:
      • short-term: 10, 25, 40
      • long-term: 40, 120, 250
    with safe fallbacks.
    """
    series = get_sentiment_series()

    if len(series) < 2:
        return (0.0, 0.0, 0.0,   
                0.0, 0.0, 0.0)   

    s = pd.Series(series)
    pct = s.pct_change()


    slope_10  = pct.rolling(10).mean().iloc[-1]
    slope_25  = pct.rolling(25).mean().iloc[-1]
    slope_40s = pct.rolling(40).mean().iloc[-1]  

    slope_40l = pct.rolling(40).mean().iloc[-1]  
    slope_120 = pct.rolling(120).mean().iloc[-1]
    slope_250 = pct.rolling(250).mean().iloc[-1]

    return (
        0.0 if pd.isna(slope_10)  else slope_10,
        0.0 if pd.isna(slope_25)  else slope_25,
        0.0 if pd.isna(slope_40s) else slope_40s,
        0.0 if pd.isna(slope_40l) else slope_40l,
        0.0 if pd.isna(slope_120) else slope_120,
        0.0 if pd.isna(slope_250) else slope_250,
    )

def export_sentiment_history_csv(path="sentiment_history.csv"):
    """Export 3-day rolling sentiment history to CSV for offline training."""
    rows = []
    for ts, s in SENTIMENT_HISTORY:
        rows.append({
            "timestamp": ts,
            "sentiment": s
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"[Sentiment] Exported {len(rows)} rows → {path}")


def sentiment_loop():
    print("💓 Sentiment thread started — 61s cadence active.")

    load_sentiment_cache()

 
    next_tick = time.time()

    while True:
        now = time.time()

       
        if now >= next_tick:
            try:
                sentiment = fetch_weighted_reddit_sentiment()
                print(f"[Sentiment] Updated → {sentiment:.4f}")
            except Exception as e:
                print(f"[Sentiment] Error in sentiment loop: {e}")

           
            next_tick = now + 61

       
        sleep_time = max(0, next_tick - time.time())
        time.sleep(sleep_time)

_last_sentiment_fetch = 0
_last_sentiment_value = 0.0








def append_to_cache(cache_df, new_df):
    """
    Safely append new candles to the cache.
    Ensures column order, prevents misalignment, and avoids timestamp corruption.
    """

   
    cols = ["timestamp", "open", "high", "low", "close", "volume"]

   
    if new_df is None or new_df.empty:
        return cache_df.reindex(columns=cols).copy()

 
    new_df = new_df.dropna(axis=1, how="all")

  
    if new_df.empty:
        return cache_df.reindex(columns=cols).copy()

   
    cache_df = cache_df.reindex(columns=cols)
    new_df   = new_df.reindex(columns=cols)

    try:
        combined = pd.concat([cache_df, new_df], ignore_index=True)
    except Exception:
       
        return cache_df.copy()

    return combined


def load_cached_candles(symbol):
    path = f"D:/Scarlet_Works/Scarlet/candle_cache_{symbol}.csv"
    cols = ["timestamp", "open", "high", "low", "close", "volume"]

    if not os.path.isfile(path):
        return pd.DataFrame(columns=cols)

    try:
        df = pd.read_csv(path)
    except Exception:
        print(f"⚠️ Cache for {symbol} is corrupted — returning empty")
        return pd.DataFrame(columns=cols)

   
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

    df = df[cols]

 
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

  
    df = df[df["timestamp"] > pd.Timestamp("2015-01-01", tz="UTC")]

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df



def build_feature_tensor(
    device,
    scaler,
    df,
    timeframe="15m",
    limit=None,
    narrator=None,
):
    """
    Build a scaled feature tensor using the SAME 51‑feature schema
    used in training, micro‑training, and offline training.
    """

    if df is None:
        raise ValueError("build_feature_tensor() requires df=engineered merged_df")

    enriched_df = df.copy()

  
    if limit is not None and len(enriched_df) > limit:
        enriched_df = enriched_df.iloc[-limit:].copy()
    print("INPUT_FEATURES COUNT:", len(INPUT_FEATURES))
    print("FIRST 10 FEATURES:", INPUT_FEATURES[:10])
    missing = [c for c in enriched_df.columns if c not in INPUT_FEATURES]
    print("COLUMNS NOT IN INPUT_FEATURES:", missing)

    X = enriched_df[INPUT_FEATURES].fillna(0.0).values

   
    X_scaled = scaler.transform(X)

   
    features_tensor = torch.tensor(
        X_scaled, dtype=torch.float32, device=device
    ).unsqueeze(0)

    if narrator:
        narrator.narrate(
            f"🧱 Feature tensor built → shape={features_tensor.shape}, timeframe={timeframe}"
        )
   
    return features_tensor, enriched_df



def add_engineered_features(df, symbols, narrator=None):
    """
    Engineered features that EXACTLY match INPUT_FEATURES.
    """

    enriched = df.copy()


    def compute_rsi(series, window=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        return (100 - (100 / (1 + rs))).fillna(0.5)

    def compute_bollinger_bands(series, window=20, num_std=2):
        mid = series.rolling(window).mean()
        std = series.rolling(window).std()
        upper = mid + num_std * std
        lower = mid - num_std * std
        return mid, upper, lower

    def compute_macd(series):
        ema12 = series.ewm(span=12, adjust=False).mean()
        ema26 = series.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return macd.fillna(0), signal.fillna(0), hist.fillna(0)

    def compute_vwap(price, volume):
        return (price * volume).cumsum() / (volume.cumsum() + 1e-9)

    def compute_atr(high, low, close, window=14):
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(window).mean().fillna(0)

    def slope(series, window):
        return series.diff(window).fillna(0)

    for sym in symbols:
        close = enriched[f"close_{sym}"]
        high  = enriched[f"high_{sym}"]
        low   = enriched[f"low_{sym}"]
        vol   = enriched[f"volume_{sym}"]

        enriched[f"RSI_{sym}"] = compute_rsi(close)

        mid, upper, lower = compute_bollinger_bands(close)
        enriched[f"BB_mid_{sym}"] = mid.fillna(close)
        enriched[f"BB_upper_{sym}"] = upper.fillna(close)
        enriched[f"BB_lower_{sym}"] = lower.fillna(close)
        enriched[f"BB_width_{sym}"] = ((upper - lower) / (mid + 1e-8)).fillna(0)


        macd_line, signal_line, macd_hist = compute_macd(close)
        enriched[f"macd_line_{sym}"] = macd_line
        enriched[f"signal_line_{sym}"] = signal_line
        enriched[f"MACD_hist_{sym}"] = macd_hist   

        enriched[f"VWAP_{sym}"] = compute_vwap(close, vol)
        enriched[f"ATR_{sym}"] = compute_atr(high, low, close)

   
        enriched[f"volatility_{sym}"] = close.pct_change().rolling(20).std().fillna(0)

     
        enriched[f"slope_10_{sym}"] = slope(close, 10)
        enriched[f"slope_30_{sym}"] = slope(close, 30)
        enriched[f"slope_40_{sym}"] = slope(close, 40)

   
        enriched[f"reddit_sentiment_{sym}"] = 0.0
        enriched[f"sentiment_slope_10_{sym}"] = 0.0
        enriched[f"sentiment_slope_25_{sym}"] = 0.0
        enriched[f"sentiment_slope_40_{sym}"] = 0.0
        enriched[f"sentiment_slope_40L_{sym}"] = 0.0
        enriched[f"sentiment_slope_120_{sym}"] = 0.0
        enriched[f"sentiment_slope_250_{sym}"] = 0.0

    if narrator:
        narrator.narrate("🧪 Engineered features generated to EXACTLY match INPUT_FEATURES schema")

    return enriched

ASSET_ORDER = ["solusd", "ethusd", "btcusd"]

def demo_inference(
    model,
    enriched_df,
    device,
    _input_features,  
    _numeric_inputs,   
    scaler,
    sym,
):
    model.eval()

    X = enriched_df[INPUT_FEATURES].fillna(0.0).values

    X_scaled = scaler.transform(X)

    seq_tensor = torch.tensor(
        X_scaled, dtype=torch.float32, device=device
    ).unsqueeze(0)

    with torch.no_grad():
        out = model(seq_tensor)

    ASSET_ORDER = ["solusd", "ethusd", "btcusd"]
    asset_idx = ASSET_ORDER.index(sym)

    prices     = out["prices"]              
    strengths  = out["strengths"]             
    deltas     = out["deltas"]               

    forecast_price    = float(prices[0, asset_idx].item())
    forecast_strength = float(strengths[0, asset_idx].item())

    forecast_vec = deltas[0, asset_idx]      
    forecast_delta = forecast_vec.cpu().numpy()
    forecast_np = forecast_delta.copy()

    current_price_tensor = torch.tensor([
        enriched_df["close_solusd"].iloc[-1],
        enriched_df["close_ethusd"].iloc[-1],
        enriched_df["close_btcusd"].iloc[-1],
    ], dtype=torch.float32, device=device)

    atr_tensor = torch.tensor([
        enriched_df["ATR_solusd"].iloc[-1],
        enriched_df["ATR_ethusd"].iloc[-1],
        enriched_df["ATR_btcusd"].iloc[-1],
    ], dtype=torch.float32, device=device)

    vwap_tensor = torch.tensor([
        enriched_df["VWAP_solusd"].iloc[-1] if "VWAP_solusd" in enriched_df else enriched_df["close_solusd"].iloc[-1],
        enriched_df["VWAP_ethusd"].iloc[-1] if "VWAP_ethusd" in enriched_df else enriched_df["close_ethusd"].iloc[-1],
        enriched_df["VWAP_btcusd"].iloc[-1] if "VWAP_btcusd" in enriched_df else enriched_df["close_btcusd"].iloc[-1],
    ], dtype=torch.float32, device=device)

    macd_line_tensor = torch.tensor([
        enriched_df["macd_line_solusd"].iloc[-1],
        enriched_df["macd_line_ethusd"].iloc[-1],
        enriched_df["macd_line_btcusd"].iloc[-1],
    ], dtype=torch.float32, device=device)

    signal_line_tensor = torch.tensor([
        enriched_df["signal_line_solusd"].iloc[-1],
        enriched_df["signal_line_ethusd"].iloc[-1],
        enriched_df["signal_line_btcusd"].iloc[-1],
    ], dtype=torch.float32, device=device)

    slope_tensor = torch.tensor([
        enriched_df["slope_10_solusd"].iloc[-1],
        enriched_df["slope_10_ethusd"].iloc[-1],
        enriched_df["slope_10_btcusd"].iloc[-1],
    ], dtype=torch.float32, device=device)

   
    try:
        sent = fetch_weighted_reddit_sentiment()
        s10, s25, s40s, s40l, s120, s250 = compute_sentiment_slopes_from_history()

        sentiment_tensor = torch.tensor(
            [sent, sent, sent], dtype=torch.float32, device=device
        )

        sentiment_slopes_tensor = torch.tensor(
            [
                [s10, s25, s40s, s40l, s120, s250],
                [s10, s25, s40s, s40l, s120, s250],
                [s10, s25, s40s, s40l, s120, s250],
            ],
            dtype=torch.float32,
            device=device,
        )

    except Exception:
        sentiment_tensor = torch.zeros(3, dtype=torch.float32, device=device)
        sentiment_slopes_tensor = torch.zeros((3, 6), dtype=torch.float32, device=device)

    return (
        forecast_price,
        forecast_delta,     
        forecast_strength,
        forecast_np,
        seq_tensor,
        current_price_tensor,
        atr_tensor,
        vwap_tensor,
        macd_line_tensor,
        signal_line_tensor,
        slope_tensor,
        sentiment_tensor,
        sentiment_slopes_tensor,
    )

from collections import deque


forecast_errors = deque(maxlen=20)  

from collections import deque
from datetime import datetime, timedelta


forecast_errors = deque(maxlen=20)  
last_drift_retrain_time = None      




import os, time, csv
import pandas as pd
import torch

from datetime import datetime




forecast_horizon = timedelta(minutes=2)
forecast_buffer = []




from decimal import Decimal
from datetime import datetime, timedelta

class DirectionalRewardShaper:
    def __init__(
        self,
        narrator,
        max_reward=0.10,
        min_reward=-0.05,
        magnitude_threshold=0.02,
        gating_threshold=0.04,
        direction_penalty=0.03,
        direction_bonus_scale=0.04,
        scale=1.0,
        alpha=0.5,
        mag_weight=0.10,      
        eps=1e-8,           
    ):
        self.narrator = narrator
        self.max_reward = max_reward
        self.min_reward = min_reward

        self.magnitude_threshold = magnitude_threshold
        self.gating_threshold = gating_threshold

        self.direction_penalty = direction_penalty
        self.direction_bonus_scale = direction_bonus_scale

        self.scale = scale
        self.alpha = alpha

        self.mag_weight = mag_weight
        self.eps = eps

        self.prev_reward = None  

    def shape(self, forecast_delta, actual_delta, volatility=None):
 
        forecast_delta = forecast_delta.float()
        actual_delta   = actual_delta.float()

      
        if volatility is None:
            volatility = actual_delta.abs() + self.eps


        error = forecast_delta - actual_delta
        abs_error = error.abs()

        reward = 0.02 - abs_error / max(self.scale, 1e-6)

        wrong_dir = (forecast_delta * actual_delta) < 0
        reward = reward - wrong_dir * self.direction_penalty

        same_dir = (forecast_delta * actual_delta) > 0
        direction_bonus = self.direction_bonus_scale * (1 - abs_error / max(self.scale, 1e-6))
        direction_bonus = torch.clamp(direction_bonus, min=0.0)
        reward = reward + same_dir * direction_bonus

        mag_err = torch.abs(torch.abs(forecast_delta) - torch.abs(actual_delta))
        mag_norm = mag_err / (volatility + self.eps)

        R_mag = torch.clamp(1.0 - mag_norm, min=0.0, max=1.0)

        reward = reward + self.mag_weight * R_mag

        reward = torch.clamp(reward, self.min_reward, self.max_reward)

        if self.prev_reward is None:
            smoothed = reward
        else:
            smoothed = self.alpha * reward + (1 - self.alpha) * self.prev_reward

        self.prev_reward = smoothed.detach()

        gated_mask = abs_error > self.gating_threshold

        mean_abs_error = abs_error.mean().item()
        mean_error = error.mean().item()
        mean_same_dir = same_dir.float().mean().item()
        mean_mag = R_mag.mean().item()

        

        return smoothed, gated_mask


RewardShaper = DirectionalRewardShaper


def log_trade_fill(memory_path, trade_row, narrator=None):
    """
    Logs a confirmed BUY or SELL to the memory file.
    Clean, modern schema aligned with EntryAnchor.
    """

    import pandas as pd
    import os


    schema = [
        "timestamp",
        "action",
        "symbol",
        "price",
        "amount",
        "trade_filled",
        "exit_note",
    ]

    if os.path.isfile(memory_path):
        df = pd.read_csv(memory_path, on_bad_lines="skip")
    else:
        df = pd.DataFrame(columns=schema)

    for col in schema:
        if col not in df.columns:
            df[col] = None

    clean_row = {}
    for col in schema:
        clean_row[col] = trade_row.get(col, None)

    clean_row["action"] = str(clean_row["action"]).upper()
    clean_row["trade_filled"] = "1" 

    df = pd.concat([df, pd.DataFrame([clean_row])], ignore_index=True)

    df.to_csv(memory_path, index=False)

    if narrator:
        narrator.narrate(
            f"📘 Logged trade → {clean_row['action']} {clean_row['symbol'].upper()} @ {clean_row['price']}"
        )

    return df

def shaping_collate_fn(batch):

    inputs = torch.stack([item["inputs"] for item in batch], dim=0)     
    targets = torch.stack([item["targets"] for item in batch], dim=0)    

    def stack_vec(key):
 
        return torch.stack([item["shaping"][key] for item in batch], dim=0) 

    shaping = {
        "current_price": stack_vec("current_price"),  
        "atr":           stack_vec("atr"),           
        "vwap":          stack_vec("vwap"),         
        "macd_hist":     stack_vec("macd_hist"),       
        "signal_line":   stack_vec("signal_line"),  
        "slope":         stack_vec("slope"),          
    }

    return {
        "inputs": inputs,
        "targets": targets,     
        "shaping": shaping,
    }



def make_dataloaders(merged_df=None, batch_size=256):
    """
    Unified dataloader builder for Scarlet.
    Produces:
        - 51‑feature engineered dataset
        - CandleDatasetV2 (shaping‑aware)
        - train/val loaders with shaping_collate_fn
    """

    symbols = ["solusd", "ethusd", "btcusd"]


    if merged_df is None:
        merged_df = fetch_and_align_assets(symbols, timeframe="15m")
        if merged_df.empty:
            raise RuntimeError("❌ No candle data retrieved from Gemini")
        narrator.narrate(f"📊 Live dataset loaded → {len(merged_df)} candles")
    else:
        narrator.narrate(f"📊 Offline dataset received → {len(merged_df)} candles")

  
    merged_df = add_engineered_features(
        merged_df,
        symbols=symbols,
        narrator=narrator
    )


    sentiment_cols = [
        "reddit_sentiment",
        "sentiment_slope_10",
        "sentiment_slope_25",
        "sentiment_slope_40",
        "sentiment_slope_40L",
        "sentiment_slope_120",
        "sentiment_slope_250",
    ]
    for col in sentiment_cols:
        if col not in merged_df.columns:
            merged_df[col] = 0.0


    INPUT_FEATURES = [
    
        "open_solusd", "high_solusd", "low_solusd", "close_solusd", "volume_solusd",

        "RSI_solusd",
        "BB_mid_solusd", "BB_upper_solusd", "BB_lower_solusd", "BB_width_solusd",
        "macd_line_solusd",         
        "signal_line_solusd",       
        "MACD_hist_solusd",
        "VWAP_solusd",
        "ATR_solusd",
        "volatility_solusd",

    
        "slope_10_solusd", "slope_30_solusd", "slope_40_solusd",

        "reddit_sentiment_solusd",
        "sentiment_slope_10_solusd",
        "sentiment_slope_25_solusd",
        "sentiment_slope_40_solusd",
        "sentiment_slope_40L_solusd",
        "sentiment_slope_120_solusd",
        "sentiment_slope_250_solusd",


        "open_ethusd", "high_ethusd", "low_ethusd", "close_ethusd", "volume_ethusd",

        
        "RSI_ethusd",
        "BB_mid_ethusd", "BB_upper_ethusd", "BB_lower_ethusd", "BB_width_ethusd",
        "macd_line_ethusd",         
        "signal_line_ethusd",      
        "MACD_hist_ethusd",
        "VWAP_ethusd",
        "ATR_ethusd",
        "volatility_ethusd",

       
        "slope_10_ethusd", "slope_30_ethusd", "slope_40_ethusd",

      
        "reddit_sentiment_ethusd",
        "sentiment_slope_10_ethusd",
        "sentiment_slope_25_ethusd",
        "sentiment_slope_40_ethusd",
        "sentiment_slope_40L_ethusd",
        "sentiment_slope_120_ethusd",
        "sentiment_slope_250_ethusd",

  
        "open_btcusd", "high_btcusd", "low_btcusd", "close_btcusd", "volume_btcusd",

      
        "RSI_btcusd",
        "BB_mid_btcusd", "BB_upper_btcusd", "BB_lower_btcusd", "BB_width_btcusd",
        "macd_line_btcusd",         
        "signal_line_btcusd",     
        "MACD_hist_btcusd",
        "VWAP_btcusd",
        "ATR_btcusd",
        "volatility_btcusd",

       
        "slope_10_btcusd", "slope_30_btcusd", "slope_40_btcusd",

      
        "reddit_sentiment_btcusd",
        "sentiment_slope_10_btcusd",
        "sentiment_slope_25_btcusd",
        "sentiment_slope_40_btcusd",
        "sentiment_slope_40L_btcusd",
        "sentiment_slope_120_btcusd",
        "sentiment_slope_250_btcusd",
    ]

    output_features = [f"close_{sym}" for sym in symbols]
    close_idx = output_features.index("close_solusd")


    ds = CandleDatasetV2(
        df=merged_df.copy(),
        INPUT_FEATURES=INPUT_FEATURES,
        output_features=output_features,
        input_seq_len=INPUT_SEQ_LEN,
        output_seq_len=OUTPUT_SEQ_LEN,
        return_diff=True,
        mode="train",
    )

   
    n_train = int(0.8 * len(ds))
    n_val = len(ds) - n_train
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])

  
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=shaping_collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=shaping_collate_fn,
        drop_last=True,
    )


    market_tensor = torch.tensor(
        merged_df[INPUT_FEATURES].values,
        dtype=torch.float32
    )

    return market_tensor, merged_df, train_loader, val_loader, INPUT_FEATURES, close_idx



def fetch_recent_window(symbol="solusd", timeframe="15m", limit=1000, min_candles=128):
    """
    Fetches up to `limit` recent candles from Gemini.
    Returns a clean DataFrame with timestamp, open, high, low, close, volume.
    Fully guarded against malformed rows and float→timestamp corruption.
    """

    url = f"https://api.gemini.com/v2/candles/{symbol}/{timeframe}"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        if not isinstance(data, list) or len(data) == 0:
            print(f"⚠️ Unexpected candle format for {symbol}: {type(data)}")
            return pd.DataFrame()

        raw = data[:limit][::-1]

        cleaned = []
        for row in raw:
           
            if not isinstance(row, (list, tuple)) or len(row) != 6:
                continue

            ts, o, h, l, c, v = row

           
            if not isinstance(ts, (int, float)):
                continue
            if ts <= 0:
                continue

            if ts < 1_000_000_000:  
                continue

            cleaned.append([ts, o, h, l, c, v])

        if not cleaned:
            print(f"⚠️ No valid candles survived filtering for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(
            cleaned,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

    
        ts = df["timestamp"].astype("int64")

        if ts.max() < 10**12:
            df["timestamp"] = pd.to_datetime(ts, unit="s", utc=True)
        else:
            df["timestamp"] = pd.to_datetime(ts, unit="ms", utc=True)

        df = df[df["timestamp"] > pd.Timestamp("2015-01-01", tz="UTC")]
        df = df.sort_values("timestamp").reset_index(drop=True)

        if len(df) < min_candles:
            print(
                f"⚠️ Only {len(df)} candles available for {symbol} ({timeframe}), "
                f"less than requested {min_candles}"
            )

        return df

    except Exception as e:
        print(f"❌ Failed to fetch candles for {symbol}: {e}")
        return pd.DataFrame()


def fetch_historical_candles(symbol="solusd", timeframe="15m", limit=1000, min_candles=128):
    return fetch_recent_window(symbol, timeframe, limit, min_candles)





def backfill_asset_history(symbol, timeframe="15m", cache_df=None, max_candles=50000):
    base_url = f"https://api.gemini.com/v2/candles/{symbol}/{timeframe}"

    if cache_df is None:
        cache_df = pd.DataFrame()

    if not cache_df.empty:
        cache_df["timestamp"] = pd.to_datetime(cache_df["timestamp"], utc=True)

    print(f"🔎 Starting backfill for {symbol} (current={len(cache_df)} candles)")

    while len(cache_df) < max_candles:

        if cache_df.empty:
            url = base_url
            print(f"📥 {symbol}: Fetching initial 1000-row window (no cache)")
        else:
            oldest_ts = cache_df["timestamp"].min()
            end_ms = int(oldest_ts.timestamp() * 1000)
            url = f"{base_url}?end={end_ms}"
            print(f"📥 {symbol}: Fetching older candles before {oldest_ts}")

        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            raw = resp.json()
        except Exception as e:
            print(f"❌ Backfill fetch error for {symbol}: {e}")
            break

        rows = list(reversed(raw))
        if not rows:
            print(f"⚠️ {symbol}: No older candles available — stopping backfill")
            break

        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        df = pd.DataFrame(rows, columns=cols)

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df[df["timestamp"] > pd.Timestamp("2000-01-01", tz="UTC")]

        new_timestamps = set(df["timestamp"]) - set(cache_df["timestamp"])
        if len(new_timestamps) == 0:
            print(f"⚠️ {symbol}: No older candles available — stopping backfill")
            break

        before = len(cache_df)
        cache_df = pd.concat([df, cache_df], ignore_index=True)
        cache_df = cache_df.drop_duplicates(subset=["timestamp"])
        cache_df = cache_df.sort_values("timestamp").reset_index(drop=True)
        after = len(cache_df)

        print(f"📚 {symbol}: Backfill added {after - before} rows → total={after}")

    if len(cache_df) > max_candles:
        cache_df = cache_df.iloc[-max_candles:].reset_index(drop=True)

    print(f"🏁 Backfill complete for {symbol} → {len(cache_df)} candles")


    save_asset_cache(symbol, cache_df)

    return cache_df

def fetch_incremental_candles(symbol, timeframe="15m", cache_df=None, max_candles=50000):
    """
    Fetches only NEW candles newer than the last cached timestamp.
    """

    base_url = f"https://api.gemini.com/v2/candles/{symbol}/{timeframe}"

    if cache_df is None or cache_df.empty:
        print(f"📥 {symbol}: No cache → fetching initial window")
        return backfill_asset_history(symbol, timeframe, cache_df, max_candles)

    last_ts = cache_df["timestamp"].iloc[-1]
    start_ms = int(pd.Timestamp(last_ts).timestamp() * 1000)

    url = f"{base_url}?start={start_ms}"
    print(f"📥 {symbol}: Fetching NEW candles since {last_ts}")

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as e:
        print(f"❌ Incremental fetch error for {symbol}: {e}")
        return cache_df

    rows = list(reversed(raw))
    if not rows:
        print(f"⚠️ {symbol}: No new candles returned")
        return cache_df

    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(rows, columns=cols)

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df[df["timestamp"] > pd.Timestamp("2000-01-01", tz="UTC")]

    before = len(cache_df)

    cache_df = pd.concat([cache_df, df], ignore_index=True)
    cache_df = cache_df.drop_duplicates(subset=["timestamp"])
    cache_df = cache_df.sort_values("timestamp").reset_index(drop=True)

    after = len(cache_df)

    print(f"📚 {symbol}: Incremental added {after - before} rows → total={after}")

    if len(cache_df) > max_candles:
        cache_df = cache_df.iloc[-max_candles:].reset_index(drop=True)

    return cache_df


def save_asset_cache(symbol, df):
    """
    Atomic, safe cache writer.
    Writes the candle DataFrame to disk without risking corruption.
    """
    path = f"D:/Scarlet_Works/Scarlet/candle_cache_{symbol}.csv"
    

   
    os.makedirs(os.path.dirname(path), exist_ok=True)

 
    tmp_path = path + ".tmp"
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)  
    except Exception as e:
        print(f"❌ Failed to save cache for {symbol}: {e}")







def load_full_cache(sym, target=50000):
    """
    Safe full-cache loader.
    Loads cached candles, enforces schema, trims to target rows.
    NEVER overwrites the cache file.
    """
    path = f"D:/Scarlet_Works/Scarlet/candle_cache_{sym}.csv"
    

    cols = ["timestamp", "open", "high", "low", "close", "volume"]

    if os.path.isfile(path):
        df = load_cached_candles(sym)
    else:
        return pd.DataFrame(columns=cols)

 
    if df.empty:
        return df


    df = df.reindex(columns=cols)

    df = df[df["timestamp"] > pd.Timestamp("2015-01-01", tz="UTC")]

    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    if len(df) > target:
        df = df.iloc[-target:].reset_index(drop=True)


    return df

def initialize_asset_cache(symbol, timeframe="15m", max_candles=50000, narrator=None):
    cache = load_full_cache(symbol)

   
    cache = backfill_asset_history(symbol, timeframe, cache, max_candles)
    save_asset_cache(symbol, cache)

   
    cache = fetch_incremental_candles(symbol, timeframe, cache, max_candles)
    save_asset_cache(symbol, cache)

    if narrator:
        narrator.narrate(f"🗄️ Cache[{symbol}] initialized → {len(cache)} candles")

    return cache




def compute_atr(high, low, close, window=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean().fillna(0)

def save_checkpoint(model, optimizer, epoch, path, narrator=None, phase="update", loss=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "timestamp": datetime.utcnow().isoformat(),
        "phase": phase,
        "loss": float(loss) if loss is not None else None
    }, path)
    if narrator:
        narrator.narrate(f"💾 Checkpoint saved → {path} (epoch={epoch}, phase={phase}, loss={loss})")

def save_micro_checkpoint(model, optimizer, loss, narrator, path=None):
    path = r"D:\Scarlet_Works\Scarlet\checkpoints\online_best.ckpt"

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=0,
        path=path,
        narrator=narrator,
        phase="microtrain",
        loss=loss
    )

    if narrator:
        narrator.narrate(
            f"💾 Online micro‑checkpoint saved → {path} "
            f"(phase=microtrain, loss={loss:.6f})"
        )




def compute_rsi(series, window=70):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
 
    return rsi.fillna(50)

def compute_bollinger_bands(series, window=70, k=2):
    mid = series.rolling(window).mean()
    std = series.rolling(window).std()

    upper = mid + k * std
    lower = mid - k * std


    mid = mid.where(mid != 0, series).fillna(series)
    upper = upper.fillna(series)
    lower = lower.fillna(series)

    return mid, upper, lower
def compute_macd(series, fast=12, slow=26, signal=9):
    """
    Computes MACD line, signal line, and histogram from a price series.
    Returns: macd_line, signal_line, macd_histogram
    """
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist
def compute_vwap(price_series, volume_series):
    """
    Computes VWAP (Volume Weighted Average Price) from price and volume Series.
    Returns a Series of VWAP values.
    """
    pv = price_series * volume_series
    cumulative_pv = pv.cumsum()
    cumulative_volume = volume_series.cumsum()
    vwap = cumulative_pv / (cumulative_volume + 1e-8)
    return vwap
def compute_slope(series, window):
    """
    Computes the slope (linear regression coefficient) over a rolling window.
    Returns a Series of slope values.
    """
    import numpy as np
    from scipy.stats import linregress

    slopes = [np.nan] * (window - 1)
    for i in range(window - 1, len(series)):
        y = series[i - window + 1:i + 1]
        x = np.arange(window)
        slope, _, _, _, _ = linregress(x, y)
        slopes.append(slope)
    return pd.Series(slopes, index=series.index)

def engineer_features_multi(merged_df: pd.DataFrame, narrator=None) -> pd.DataFrame:
    """
    Recompute ALL engineered features on the aligned dataframe.
    Ensures non-zero ATR, VWAP, MACD, slopes, RSI, Bollinger Bands,
    AND includes sentiment placeholders to match INPUT_FEATURES.
    """

    assets = ["solusd", "ethusd", "btcusd"]
    slope_windows = (10, 30, 40)

    for sym in assets:
        close = merged_df[f"close_{sym}"]
        high  = merged_df[f"high_{sym}"]
        low   = merged_df[f"low_{sym}"]
        vol   = merged_df[f"volume_{sym}"]


        merged_df[f"RSI_{sym}"] = compute_rsi(close).fillna(0.5)

        mid, upper, lower = compute_bollinger_bands(close)
        merged_df[f"BB_mid_{sym}"] = mid.fillna(close)
        merged_df[f"BB_upper_{sym}"] = upper.fillna(close)
        merged_df[f"BB_lower_{sym}"] = lower.fillna(close)
        merged_df[f"BB_width_{sym}"] = ((upper - lower) / (mid + 1e-8)).fillna(0.0)

        macd_line, signal_line, macd_hist = compute_macd(close)
        merged_df[f"macd_line_{sym}"] = macd_line.fillna(0.0)
        merged_df[f"signal_line_{sym}"] = signal_line.fillna(0.0)
        merged_df[f"MACD_hist_{sym}"] = macd_hist.fillna(0.0)


        vwap = compute_vwap(close, vol)
        merged_df[f"VWAP_{sym}"] = vwap.bfill().ffill()


        merged_df[f"ATR_{sym}"] = compute_atr(high, low, close).fillna(0.0)

 
        merged_df[f"volatility_{sym}"] = (
            close.pct_change().rolling(20).std().fillna(0.0)
        )


        for w in slope_windows:
            merged_df[f"slope_{w}_{sym}"] = close.diff(w).fillna(0.0)

        merged_df[f"reddit_sentiment_{sym}"] = 0.0
        merged_df[f"sentiment_slope_10_{sym}"] = 0.0
        merged_df[f"sentiment_slope_25_{sym}"] = 0.0
        merged_df[f"sentiment_slope_40_{sym}"] = 0.0
        merged_df[f"sentiment_slope_40L_{sym}"] = 0.0
        merged_df[f"sentiment_slope_120_{sym}"] = 0.0
        merged_df[f"sentiment_slope_250_{sym}"] = 0.0

    if narrator:
        narrator.narrate("🌐 Multi-asset feature engineering complete → SOL, ETH, BTC")

    return merged_df


MAX_CANDLES = 30_000

CACHE_DIR = r"D:\Scarlet_Works\Scarlet"
  
def get_cache_path(symbol):
    return os.path.join(CACHE_DIR, f"candle_cache_{symbol}.csv")




def _lookup_from_df(df, target_time):
    """Return nearest close price from a candle dataframe."""
    try:
        target_time = pd.to_datetime(target_time, utc=True, errors="coerce")
        if target_time is None:
            return get_market_price()

        idx = (df["timestamp"] - target_time).abs().idxmin()
        price = df.loc[idx, "close"]
        return float(price)
    except Exception:
        return get_market_price()







def fetch_latest_chunk(symbol, timeframe="15m", limit=1000):
    return fetch_historical_candles(symbol, timeframe, limit)




MAX_CANDLES = 30_000


def trim_cache(df, max_len=MAX_CANDLES):
    if len(df) <= max_len:
        return df
    return df.iloc[-max_len:].reset_index(drop=True)
















def fetch_and_align_assets(symbols, timeframe="15m"):
    """
    Load full cached candles for each asset, suffix columns, and return a single
    fully aligned multi‑asset DataFrame with forward/backward fill applied.

    Alignment is strict (inner join on timestamp) to guarantee identical rows
    across all assets for multi‑asset feature engineering and model input.
    """

    merged = None
    aligned_assets = []

    for sym in symbols:
        df = load_full_cache(sym)

        if df.empty:
            print(f"⚠️ Skipping {sym}: empty cache")
            continue

   
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])

   
        rename_map = {col: f"{col}_{sym}" for col in df.columns if col != "timestamp"}
        df = df.rename(columns=rename_map)

        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="timestamp", how="inner")

        aligned_assets.append(sym)
        print(f"✅ Aligned {sym} — {len(df)} candles")


    if merged is None:
        print("❌ No valid assets retrieved")
        return pd.DataFrame()

    merged = merged.sort_values("timestamp").reset_index(drop=True)
    merged = merged.ffill().bfill()

    print(f"📊 Final aligned rows: {len(merged)}")
    print(f"🧩 Assets aligned: {', '.join(aligned_assets)}")

    return merged




symbols = ["solusd", "ethusd", "btcusd"]


for sym in symbols:
    initialize_asset_cache(sym, timeframe="15m", max_candles=50000, narrator=narrator)


crypto_data = fetch_and_align_assets(symbols, timeframe="15m")




def update_asset_cache(symbol, timeframe="15m", narrator=None):
    """
    Fetches new candles, appends them to the cache, and saves the result.
    This is the missing link that makes the cache actually persist.
    """
 
    cache_df = load_full_cache(symbol)

    new_df = fetch_recent_window(symbol, timeframe=timeframe, limit=1000)

    combined = append_to_cache(cache_df, new_df)

    combined = combined.sort_values("timestamp").drop_duplicates("timestamp")

    save_asset_cache(symbol, combined)

    return combined



import time
from datetime import datetime, timedelta
from decimal import Decimal
import os, torch
from datetime import datetime












        
from decimal import Decimal
from sklearn.preprocessing import RobustScaler







def build_offline_dataset(
    symbols=None,
    timeframe="15m",
    target=50000,
    narrator=None,
):
    if symbols is None:
        symbols = ["solusd", "ethusd", "btcusd"]

    sol = load_full_cache("solusd", target=target)
    eth = load_full_cache("ethusd", target=target)
    btc = load_full_cache("btcusd", target=target)

    print(f"[Offline Training] sol={len(sol)}, eth={len(eth)}, btc={len(btc)}")

    merged_df = sol.merge(eth, on="timestamp", suffixes=("", "_eth"))
    merged_df = merged_df.merge(btc, on="timestamp", suffixes=("", "_btc"))
    merged_df = merged_df.sort_values("timestamp").reset_index(drop=True)

    col_map = {

        "open": "open_solusd", "high": "high_solusd", "low": "low_solusd",
        "close": "close_solusd", "volume": "volume_solusd",

        "open_eth": "open_ethusd", "high_eth": "high_ethusd", "low_eth": "low_ethusd",
        "close_eth": "close_ethusd", "volume_eth": "volume_ethusd",

        "open_btc": "open_btcusd", "high_btc": "high_btcusd", "low_btc": "low_btcusd",
        "close_btc": "close_btcusd", "volume_btc": "volume_btcusd",
    }
    merged_df = merged_df.rename(columns=col_map)

    if merged_df.empty:
        raise RuntimeError("❌ No candle data retrieved from cache for offline training")

    if narrator:
        narrator.narrate(
            f"📊 Offline dataset loaded → {len(merged_df)} aligned candles for {', '.join(symbols)}"
        )


    merged_df = engineer_features_multi(merged_df, narrator=narrator)

    sentiment_cols = [
        "reddit_sentiment",
        "sentiment_slope_10",
        "sentiment_slope_25",
        "sentiment_slope_40",
        "sentiment_slope_40L",
        "sentiment_slope_120",
        "sentiment_slope_250",
    ]

    for sym in symbols:
        for col in sentiment_cols:
            merged_df[f"{col}_{sym}"] = 0.0

    return merged_df


def load_or_build_offline_df(symbols=None, narrator=None, lookback=2000):

    merged_df = build_offline_dataset(
        symbols=symbols,
        narrator=narrator,
        target=lookback,
    )

    engineered = engineer_features_multi(merged_df, narrator=narrator)

    numeric_cols = INPUT_FEATURES

    scaler = RobustScaler().fit(engineered[numeric_cols].fillna(0.0))

    df_scaled = engineered.copy()
    df_scaled[numeric_cols] = scaler.transform(
        engineered[numeric_cols].fillna(0.0)
    )

    return engineered, df_scaled, scaler, numeric_cols





def build_window_dataset(merged_df, window_len=8):
    total_rows = len(merged_df)
    total_windows = max(0, total_rows - window_len)
    return [(i, i + window_len) for i in range(total_windows)]

class MultiAssetWindowDataset(Dataset):
    def __init__(self, df_raw: pd.DataFrame, df_scaled: pd.DataFrame,
                 window_size: int = 128, target_horizon: int = 1):

        self.df_raw = df_raw.drop(columns=["timestamp"], errors="ignore")
        self.df = df_scaled.drop(columns=["timestamp"], errors="ignore")

        self.window_size = window_size
        self.target_horizon = target_horizon

        self.features = self.df.values.astype("float32")
        self.num_rows, self.num_features = self.features.shape

        self.max_start = self.num_rows - window_size - target_horizon

        def pick(name):
            if name in self.df_raw.columns:
                return name
            raise KeyError(f"Missing column: {name}")

        self.close_cols = [
            pick("close_solusd"),
            pick("close_ethusd"),
            pick("close_btcusd"),
        ]

    def __len__(self):
        return max(0, self.max_start)

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.window_size]  

        now_row = self.df_raw.iloc[idx + self.window_size - 1]
        future_row = self.df_raw.iloc[idx + self.window_size + self.target_horizon - 1]

        now_prices = now_row[self.close_cols].values.astype("float32")
        future_prices = future_row[self.close_cols].values.astype("float32")

        returns = (future_prices - now_prices) / (now_prices + 1e-8)
        y = returns  

        return {
            "inputs": torch.from_numpy(x),      
            "targets": torch.from_numpy(y),    
        }


    def sample_random(self, device):
        idx = random.randint(0, len(self) - 1)
        sample = self[idx]

        x = sample["inputs"].to(device)
        y = sample["targets"].to(device)

     
        vol = torch.std(y).clamp(min=1e-6)
        vol = vol.expand(3)

      
        return {
            "inputs": x,     
            "targets": y,     
            "vol": vol,     
        }



def build_dataloader(df_raw, df_scaled, batch_size=256, window_size=128, target_horizon=1):
    dataset = MultiAssetWindowDataset(
        df_raw=df_raw,
        df_scaled=df_scaled,
        window_size=window_size,
        target_horizon=target_horizon
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    return loader


def compute_costbasis_loss_vectorized(
    model,
    batch_tensor,      
    target_deltas,     
    current_price,    
    policy_config,
    atr_value,         
    vwap_value,        
    macd_line,        
    signal_line,      
    slope_value,      
    device,
):
    """
    Fully vectorized offline/online loss.
    Uses model(...) which returns:
        {"forecast_deltas": {"multi": (B, 3, 6)}}
    We train ONLY on SOL (asset index 0), across 6 horizons.
    """

  
    batch_tensor = batch_tensor.to(device)
    target_deltas = target_deltas.to(device)
    current_price = current_price.to(device)
    atr_value     = atr_value.to(device)
    vwap_value    = vwap_value.to(device)
    macd_line     = macd_line.to(device)
    signal_line   = signal_line.to(device)
    slope_value   = slope_value.to(device)

    if target_deltas.dim() == 3:

        sol_idx = 0
        target_deltas = target_deltas[:, sol_idx, :]        

    elif target_deltas.dim() == 2:
      
        if target_deltas.size(1) == 1:
            target_deltas = target_deltas.squeeze(-1)       
     

    if target_deltas.dim() == 1:
        target_deltas = target_deltas.unsqueeze(-1).repeat(1, 6) 

    out = model(batch_tensor)

    sol_idx = 0
    try:
        pred_deltas = out["forecast_deltas"]["multi"][:, sol_idx, :]   
    except Exception as e:
        raise RuntimeError(f"Model output missing forecast_deltas['multi'] or bad shape: {e}")

    reward = online_costbasis_reward_from_verdict_batch(
        forecast_delta=pred_deltas,    
        realized_delta=target_deltas,   
        current_price=current_price,    
        policy_config=policy_config,
        atr_value=atr_value,           
        vwap_value=vwap_value,          
        macd_line=macd_line,            
        signal_line=signal_line,        
        slope_value=slope_value,         
    )  


    base_loss = -reward.mean()
    l2_term   = 0.0001 * (pred_deltas ** 2).mean()

    return base_loss + l2_term




def online_costbasis_reward_from_verdict_batch(
    forecast_delta,  
    realized_delta,   
    current_price,    
    policy_config,
    atr_value,        
    vwap_value,       
    macd_line,        
    signal_line,      
    slope_value,      
):
    """
    Multi-asset online reward.
    No SOL-only collapse; returns (B, 3).
    If forecast/realized are (B,3,H), we use the last horizon.
    """

    device = forecast_delta.device

    def to_B3(x):
        if x is None:
            return None
        if x.dim() == 1:
  
            return x.unsqueeze(-1).expand(-1, 3)
        if x.dim() == 2:
   
            if x.size(1) == 1:
                return x.expand(-1, 3)
            return x
        if x.dim() == 3:
 
            return x[:, :, -1]
        raise ValueError(f"Unexpected shape in to_B3: {x.shape}")

    forecast_delta = to_B3(forecast_delta)   
    realized_delta = to_B3(realized_delta)   
    current_price  = to_B3(current_price)   
    atr_value      = to_B3(atr_value)       
    vwap_value     = to_B3(vwap_value)       
    macd_line      = to_B3(macd_line)      
    signal_line    = to_B3(signal_line)     
    slope_value    = to_B3(slope_value)      


    atr_ratio = atr_value / (current_price + 1e-8)
    macd_hist = macd_line - signal_line

    cond_atr   = atr_ratio.abs() < 0.002
    cond_macd  = macd_hist.abs() < 0.0005
    cond_slope = slope_value.abs() < 0.0005

    flat_mask  = cond_atr & cond_macd & cond_slope
    flat_score = flat_mask.float()          # (B,3)


    forecast_mag = torch.abs(forecast_delta)

    buy_mask  = forecast_delta > 0
    sell_mask = forecast_delta < 0
    hold_mask = forecast_delta == 0

    future_price = current_price * (1.0 + realized_delta)
    roi = realized_delta

    direction_correctness = torch.where(
        forecast_delta * realized_delta > 0,
        torch.tensor(1.0, device=device),
        torch.tensor(-1.0, device=device),
    )

    slope_negative = slope_value < 0
    macd_bearish   = macd_line < signal_line
    vwap_below     = future_price < vwap_value
    atr_ok         = atr_value > 0.0

    stop_loss_threshold = float(policy_config.big_loss_threshold)

    sell_profitable     = roi >= 0.0
    sell_stop_loss      = roi <= stop_loss_threshold
    sell_indicator_exec = slope_negative & macd_bearish & vwap_below & atr_ok

    sell_should_exec = sell_mask & (sell_profitable | sell_stop_loss | sell_indicator_exec)

    buy_delta_threshold = float(policy_config.buy_delta_threshold)
    roi_threshold       = float(policy_config.min_roi_for_confident_buy)

    forecast_strong = forecast_mag >= buy_delta_threshold
    roi_ok          = roi >= roi_threshold

    macd_bullish = macd_line > signal_line
    vwap_above   = future_price > vwap_value
    slope_up     = slope_value > 0.0
    atr_ok_buy   = atr_value > 0.0

    buy_should_exec = (
        buy_mask
        & forecast_strong
        & roi_ok
        & macd_bullish
        & vwap_above
        & slope_up
        & atr_ok_buy
    )

    hold_should_exec = torch.zeros_like(hold_mask, dtype=torch.bool)

    should_execute = sell_should_exec | buy_should_exec | hold_should_exec

    flat_suppress = flat_score > 0.7
    roi_small     = roi.abs() < 0.005

    suppressed = flat_suppress & roi_small

    should_execute = torch.where(
        suppressed,
        torch.zeros_like(should_execute, dtype=torch.bool),
        should_execute,
    )

    reward = roi * direction_correctness

    reward = torch.where(
        should_execute,
        reward,
        reward * 0.25,
    )

    reward = reward * (1 - 0.5 * flat_score)

    return reward 





from typing import Dict



def offline_shaping_collate_fn(batch):
    """
    Collate function that:
    - stacks inputs → (B, T, F)
    - computes 12×15m horizon deltas
    - assembles shaping dict with (B, 3) tensors for all assets
    """

    inputs = torch.stack([item["inputs"] for item in batch], dim=0)  

    horizon_deltas = []
    current_prices = []
    atr_vals = []
    vwap_vals = []
    macd_lines = []
    signal_lines = []
    slopes = []

    for item in batch:
        closes = item["targets"].view(-1, 3) 
        start = closes[0]
        end = closes[-1]
        delta = (end - start) / torch.clamp(start, min=1e-8)
        horizon_deltas.append(delta)

        shp = item["shaping"]

     
        current_prices.append(torch.tensor([
            shp["current_price_solusd"],
            shp["current_price_ethusd"],
            shp["current_price_btcusd"],
        ], dtype=torch.float32))

        atr_vals.append(torch.tensor([
            shp["ATR_solusd"],
            shp["ATR_ethusd"],
            shp["ATR_btcusd"],
        ], dtype=torch.float32))

        vwap_vals.append(torch.tensor([
            shp["VWAP_solusd"],
            shp["VWAP_ethusd"],
            shp["VWAP_btcusd"],
        ], dtype=torch.float32))

        macd_lines.append(torch.tensor([
            shp["macd_line_solusd"],
            shp["macd_line_ethusd"],
            shp["macd_line_btcusd"],
        ], dtype=torch.float32))

        signal_lines.append(torch.tensor([
            shp["signal_line_solusd"],
            shp["signal_line_ethusd"],
            shp["signal_line_btcusd"],
        ], dtype=torch.float32))

        slopes.append(torch.tensor([
            shp["slope_10_solusd"],
            shp["slope_10_ethusd"],
            shp["slope_10_btcusd"],
        ], dtype=torch.float32))


    targets = torch.stack(horizon_deltas, dim=0)      
    current_price = torch.stack(current_prices, dim=0) 
    atr = torch.stack(atr_vals, dim=0)                
    vwap = torch.stack(vwap_vals, dim=0)               
    macd_line = torch.stack(macd_lines, dim=0)         
    signal_line = torch.stack(signal_lines, dim=0)     
    slope = torch.stack(slopes, dim=0)                 

    shaping = {
        "current_price": current_price,
        "atr": atr,
        "vwap": vwap,
        "macd_line": macd_line,
        "signal_line": signal_line,
        "slope": slope,
    }

    return {
        "inputs": inputs,
        "targets": targets,
        "shaping": shaping,
    }



def make_offline_dataloader(batch_size, device):
    symbols = ["solusd", "ethusd", "btcusd"]

    
    merged = fetch_and_align_assets(symbols)

 
    merged = add_engineered_features(
        merged,
        symbols=symbols,
        narrator=None
    )

  
    dataset = CandleDatasetV2(
        df=merged,
        INPUT_FEATURES=INPUT_FEATURES,
        output_features=["close_solusd", "close_ethusd", "close_btcusd"],
        input_seq_len=128,
        output_seq_len=48,
        return_diff=False,
        mode="offline",
    )


    train_size = int(len(dataset) * 0.9)
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=offline_shaping_collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=offline_shaping_collate_fn,
    )

    return train_loader, val_loader, INPUT_FEATURES, ["close_solusd", "close_ethusd", "close_btcusd"]

from tqdm import tqdm



class PassiveLearningScheduler:
    def __init__(
        self,
        start_weight=0.0,    
        max_weight=0.20,      
        warmup_steps=10,      
        ramp_steps=500,      
        error_clip=0.001,
    ):
        self.start_weight = start_weight
        self.max_weight = max_weight
        self.warmup_steps = warmup_steps
        self.ramp_steps = ramp_steps
        self.error_clip = error_clip

    def weight(self, step):
        if step < self.warmup_steps:
            return 0.0
        t = min(1.0, (step - self.warmup_steps) / max(self.ramp_steps, 1))
        return self.start_weight + t * (self.max_weight - self.start_weight)

    def filter_error(self, err):
        if abs(err) < self.error_clip:
            return 0.0
        return err



def passive_learning_update(
    forecast_error,
    features_tensor,
    model,
    optimizer,
    device,
    volatility=0.0,
    confidence=1.0,
    scheduler=None,
    global_step=0,
    narrator=None,
):
    """
    Passive learning from scalar forecast error.
    Multi-horizon safe: collapses any vector input to scalar.
    """

    if isinstance(forecast_error, (list, tuple, np.ndarray, torch.Tensor)):
        arr = np.array(forecast_error, dtype=float)
        forecast_error = float(arr.reshape(-1)[-1])
    else:
        forecast_error = float(forecast_error)

    if abs(forecast_error) > 5:
        if narrator:
            narrator.narrate(f"⚠️ Passive learning skipped — absurd forecast_error={forecast_error}.")
        return None


    if scheduler is not None:
        w = scheduler.weight(global_step)
        filtered_error = scheduler.filter_error(forecast_error)
    else:
        w = 1.0
        filtered_error = forecast_error

    if w <= 0:
        return None

    if filtered_error == 0:
        if narrator:
            narrator.narrate("🌫 Passive learning skipped — error too small.")
        return None


    vol_scale = 1.0 / (1.0 + float(volatility))
    conf_scale = max(0.1, float(confidence))

    effective_error = filtered_error * w * vol_scale * conf_scale

    if narrator:
        narrator.narrate(
            f"🧩 Passive learning factors → "
            f"raw={forecast_error:.5f}, filtered={filtered_error:.5f}, "
            f"w={w:.4f}, vol_scale={vol_scale:.4f}, conf_scale={conf_scale:.4f}, "
            f"effective={effective_error:.5f}"
        )

  
    if features_tensor is None:
        if narrator:
            narrator.narrate("⚠️ Passive learning skipped — features_tensor=None.")
        return None

    if features_tensor.ndim != 3:
        if narrator:
            narrator.narrate(f"❌ Passive learning aborted — expected [1, L, F], got {features_tensor.shape}.")
        return None

    input_tensor = features_tensor.detach().clone().to(device)

    target = torch.tensor([effective_error], dtype=torch.float32, device=device)

    try:
        out = model(input_tensor)
    except Exception as e:
        if narrator:
            narrator.narrate(f"❌ Passive learning forward pass failed: {e}")
        return None

    try:
        pred_vec = out["forecast_deltas"]["multi"].reshape(-1)
        pred = pred_vec[-1]  # last horizon scalar
    except Exception as e:
        if narrator:
            narrator.narrate(f"❌ Passive learning failed — cannot extract scalar pred: {e}")
        return None

    loss = (pred - target).pow(2).mean()

    optimizer.zero_grad(set_to_none=True)

    try:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    except Exception as e:
        if narrator:
            narrator.narrate(f"❌ Passive learning backward/update failed: {e}")
        return None

    if narrator:
        narrator.narrate(
            f"🧪 Passive learning → Δ={effective_error:.6f}, loss={loss.item():.6f}"
        )

    return float(loss.item())
from decimal import Decimal
from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
@dataclass
class AssetPosture:
    symbol: str
    exposure: Decimal = Decimal("0")
    avg_entry: Optional[Decimal] = None
    is_flat: bool = True

    recent_loss: Optional[Decimal] = None
    cooldown_cycles_remaining: int = 0

    def update(self, wallet_amount: Decimal, entry_info: dict | None,
               dust=Decimal("0.00001")):


        if wallet_amount <= dust:
            self.exposure = Decimal("0")
            self.avg_entry = None
            self.is_flat = True

            return

        prev_exposure = None
        if entry_info and "exposure" in entry_info:
            try:
                prev_exposure = Decimal(str(entry_info["exposure"]))
            except Exception:
                prev_exposure = None

        if prev_exposure is None or prev_exposure <= dust:
    
            self.exposure = wallet_amount
            self.avg_entry = None
        else:
       
            self.exposure = prev_exposure

        self.is_flat = (self.exposure <= Decimal("0"))

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict

@dataclass
class AssetPosture:
    symbol: str
    avg_entry: Decimal | None = None
    exposure: Decimal = Decimal("0")
    cooldown: int = 0


@dataclass
class PortfolioPosture:
    assets: Dict[str, AssetPosture] = field(default_factory=dict)


    anchor_recovered: Dict[str, bool] = field(default_factory=dict)

    def get(self, symbol: str) -> AssetPosture:
        if symbol not in self.assets:
            self.assets[symbol] = AssetPosture(symbol=symbol)
        return self.assets[symbol]

    def update_from_wallets(self, wallet_fn, entry_state: Dict[str, dict], dust=Decimal("0.00001")):
        for sym, state in entry_state.items():
            base = sym.replace("usd", "").upper()
            wallet_amount = Decimal(str(wallet_fn(base) or 0))
            entry_info = state.get("entry_info")
            self.get(sym).update(wallet_amount, entry_info, dust=dust)

class OnlineRLScheduler:
    def __init__(
        self,
        rl_start: float = 0.0,   
        rl_max: float = 0.10,     
        warmup_steps: int = 5,  
        ramp_steps: int = 1000,   
        reward_ramp: float = 0.05 
    ):
        self.rl_start = rl_start
        self.rl_max = rl_max
        self.warmup_steps = warmup_steps
        self.ramp_steps = ramp_steps
        self.reward_ramp = reward_ramp

    def rl_weight(self, global_step: int) -> float:

        if global_step < self.warmup_steps:
            return 0.0

        t = min(
            1.0,
            (global_step - self.warmup_steps) / max(self.ramp_steps, 1)
        )
        return self.rl_start + t * (self.rl_max - self.rl_start)

    def reward_multiplier(self, global_step: int) -> float:
      
        t = min(1.0, global_step / max(self.ramp_steps, 1))
        return 1.0 + self.reward_ramp * t





from decimal import Decimal



from pynput import keyboard

pressed_keys = set()

def on_press(key):
    pressed_keys.add(key)

def on_release(key):
    if key in pressed_keys:
        pressed_keys.remove(key)



def get_current_usd_balance():
    try:
        balances = get_gemini_balances()
        if not balances:
            return Decimal("0")
        return Decimal(str(balances.get("USD", Decimal("0"))))
    except Exception:
        return Decimal("0")

def get_live_price(symbol):
    """
    Lightweight wrapper around get_market_price().
    Returns Decimal price or Decimal('0') on failure.
    """
    try:
        price = get_market_price(symbol, return_history=False)
        if price is None:
            return Decimal("0")
        return Decimal(str(price))
    except Exception as e:
        print(f"❌ get_live_price error for {symbol}: {e}")
        return Decimal("0")

def execute_hotkey_sell_all(narrator=None):
    """
    Real‑time PANIC EXIT.
    Immediately sells all SOL, ETH, BTC using live balances and live prices.
    """
    try:
        balances = get_gemini_balances()
        if not balances:
            if narrator:
                narrator.narrate("🚨 Panic Exit aborted — could not fetch balances.")
            return
    except Exception as e:
        if narrator:
            narrator.narrate(f"🚨 Panic Exit failed — balance error: {e}")
        return

    assets = {
        "solusd": "SOL",
        "ethusd": "ETH",
        "btcusd": "BTC",
    }

    for sym, cur in assets.items():
        bal = balances.get(cur.upper(), Decimal("0"))
        if bal <= 0:
            continue

        price = get_live_price(sym)
        if price <= 0:
            if narrator:
                narrator.narrate(f"⚠️ Panic Exit skipped for {cur} — no live price.")
            continue

        if narrator:
            narrator.narrate(
                f"🚨 Real‑time PANIC EXIT → SELL ALL {cur} "
                f"({bal} @ {price})"
            )

        result = execute_trade_gemini(
            symbol=sym,
            amount=str(bal),
            action="sell",
            override=True,
        )

        if narrator:
            narrator.narrate(f"📬 Panic Exit result for {cur} → {result}")


def execute_hotkey_trade(symbol, narrator=None):
    from decimal import Decimal


    sym = symbol.lower() + "usd"

    usd_balance = get_current_usd_balance()
    price = get_live_price(sym)

    BUY_FRACTION = POLICY_CONFIG.buy_fraction.get(sym, Decimal("0.25"))
    MIN_NOTIONAL = POLICY_CONFIG.min_notional.get(sym, Decimal("1.00"))

    notional = usd_balance * BUY_FRACTION
    if notional < MIN_NOTIONAL:
        if narrator:
            narrator.narrate(f"⚠️ Hotkey BUY skipped — notional too small ({notional})")
        return

    amount = notional / price

    if narrator:
        narrator.narrate(
            f"⚡ Real‑time hotkey BUY → {symbol.upper()} "
            f"amount={amount} price={price}"
        )

    result = execute_trade_gemini(
        symbol=sym,
        amount=str(amount),
        action="buy",
        override=True,
    )

    if narrator:
        narrator.narrate(f"📬 Hotkey BUY result → {result}")

def key_pressed(combo):
    """
    combo examples:
      "ctrl+d"
      "ctrl+s"
      "ctrl+a"
      "ctrl+p"
    """
    combo = combo.lower()

    ctrl_down = (
        keyboard.Key.ctrl_l in pressed_keys or
        keyboard.Key.ctrl_r in pressed_keys
    )

    if combo.startswith("ctrl+"):
        key_char = combo.split("+")[1]
        return ctrl_down and keyboard.KeyCode.from_char(key_char) in pressed_keys

    return False
listener = None

def start_keyboard_listener():
    global listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()
    print("🔥 Global keyboard listener started")

def hotkey_listener(narrator):
    print("🔥 Hotkey listener thread started")

    while True:
        if key_pressed("ctrl+a"):
            narrator.narrate("⚡ Real‑time hotkey BUY for SOL")
            execute_hotkey_trade("sol", narrator)

        if key_pressed("ctrl+s"):
            narrator.narrate("⚡ Real‑time hotkey BUY for ETH")
            execute_hotkey_trade("eth", narrator)

        if key_pressed("ctrl+d"):
            narrator.narrate("⚡ Real‑time hotkey BUY for BTC")
            execute_hotkey_trade("btc", narrator)

        if key_pressed("ctrl+p"):
            narrator.narrate("🚨 Real‑time PANIC EXIT triggered")
            execute_hotkey_sell_all(narrator)

        time.sleep(0.05)



import threading
from decimal import Decimal

DRIFT_THRESHOLD = Decimal("0.2")
_microtrain_lock = threading.Lock()
_microtrain_running = False





def compute_rolling_vwap(df, sym, window=96):
    price_col = f"close_{sym}"
    vol_col = f"volume_{sym}"

    if price_col not in df.columns or vol_col not in df.columns:
        return None

    prices = df[price_col].astype(float)
    volumes = df[vol_col].astype(float)

    pv = prices * volumes
    pv_roll = pv.rolling(window=window, min_periods=1).sum()
    vol_roll = volumes.rolling(window=window, min_periods=1).sum()

    vwap = pv_roll / (vol_roll + 1e-9)
    return float(vwap.iloc[-1])




class FlatRegimeClassifier:
    """
    Detects volatility-compression / flat regimes.
    Returns a score in [0, 1] where:
        0.0 = trending / volatile
        1.0 = fully flat / compressed
    """

    def __init__(self,
                 atr_ratio_thresh=0.002,     
                 macd_thresh=0.0005,       
                 slope_thresh=0.0005,       
                 bb_width_thresh=0.003,       
                 recent_delta_thresh=0.001):  
        self.atr_ratio_thresh = atr_ratio_thresh
        self.macd_thresh = macd_thresh
        self.slope_thresh = slope_thresh
        self.bb_width_thresh = bb_width_thresh
        self.recent_delta_thresh = recent_delta_thresh

    def classify(self, features, recent_actuals):
        """
        features: dict with keys:
            atr, price, macd_hist, slope_10, slope_30, slope_40, bb_width
        recent_actuals: list of recent actual deltas (e.g., last 5)
        """

        atr_ratio = features["atr"] / features["price"]
        macd_flat = abs(features["macd_hist"]) < self.macd_thresh
        slope_flat = (
            abs(features["slope_10"]) < self.slope_thresh and
            abs(features["slope_30"]) < self.slope_thresh and
            abs(features["slope_40"]) < self.slope_thresh
        )
        bb_flat = features["bb_width"] < self.bb_width_thresh
        actuals_flat = all(abs(x) < self.recent_delta_thresh for x in recent_actuals)

      
        conditions = [atr_ratio < self.atr_ratio_thresh,
                      macd_flat,
                      slope_flat,
                      bb_flat,
                      actuals_flat]

        score = sum(conditions) / len(conditions)
        return score


classifier = FlatRegimeClassifier()



def compute_drift(
    pred_delta,
    true_delta,
    volatility=None,
    atr=None,
    rolling_std=None,
    regime=None,
    max_mag=0.05,
):
    """
    pred_delta:  (A) or (B, A)
    true_delta:  same shape
    volatility:  ATR/price or similar
    atr:         raw ATR values (optional)
    rolling_std: rolling std of returns (optional)
    regime:      string: "trend", "chop", "volatile", etc.
    """

    eps = 1e-6

    vols = []

    if volatility is not None:
        vols.append(volatility)

    if atr is not None:
        vols.append(atr)

    if rolling_std is not None:
        vols.append(rolling_std)

    if len(vols) == 0:
        base_vol = torch.ones_like(pred_delta)
    else:
        base_vol = torch.stack(vols, dim=0).mean(dim=0)

  
    base_vol = torch.clamp(base_vol, min=1e-3)

   
    dynamic_max_mag = torch.clamp(5 * base_vol, 0.005, max_mag)

    pred = torch.clamp(pred_delta, -dynamic_max_mag, dynamic_max_mag)

 
    mag_drift = (pred - true_delta).abs() / (base_vol + eps)
    mag_drift = mag_drift.mean()

  
    dir_drift = (torch.sign(pred) != torch.sign(true_delta)).float().mean()


    conf_drift = pred.abs().mean()


    if regime == "trend":
        w_mag, w_dir, w_conf = 0.6, 0.3, 0.1
    elif regime == "chop":
        w_mag, w_dir, w_conf = 0.4, 0.5, 0.1
    elif regime == "volatile":
        w_mag, w_dir, w_conf = 0.7, 0.2, 0.1
    else:
        w_mag, w_dir, w_conf = 0.7, 0.3, 0.0


    drift = (
        w_mag * mag_drift +
        w_dir * dir_drift +
        w_conf * conf_drift
    )

    return drift, {
        "mag_drift": mag_drift.item(),
        "dir_drift": dir_drift.item(),
        "conf_drift": conf_drift.item(),
        "volatility": base_vol.mean().item(),
        "dynamic_max_mag": dynamic_max_mag.mean().item(),
        "regime": regime,
    }

class OnlineReplayBuffer:
    def __init__(self, capacity=512, min_samples=64):
        self.capacity = capacity
        self.min_samples = min_samples
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def can_sample(self):
        return len(self.buffer) >= self.min_samples

    def add(self, sample):
        """
        Adds a sample only if it contains the required keys and has the correct shape.
        Prevents malformed samples from entering the buffer.
        NOTE: This class does NOT narrate — narration belongs in the caller.
        """
        if sample is None:
            return

    
        required = ("inputs", "targets", "vol")
        if not all(k in sample for k in required):
            print("⚠️ Skipping malformed sample (missing required keys)")
            return

        x = sample["inputs"]

      
        if x.ndim != 2 or x.shape[0] != 128:
            print(f"⚠️ Skipping malformed sample with shape {x.shape}")
            return

      
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)

        self.buffer.append(sample)

    def sample_batch_with_vol(self, batch_size, device):
        """
        Returns (inputs, targets, vol) as tensors on the given device.
        Assumes each stored sample is a dict with:
            'inputs': (128, F)
            'targets': (3,)
            'vol': (3,)
        """
        import random
        batch = random.sample(self.buffer, batch_size)

        for s in batch:
            if "vol" not in s:
                raise KeyError("❌ Sample missing 'vol' — buffer contains malformed entries.")

        inputs = torch.stack([s["inputs"] for s in batch]).to(device)
        targets = torch.stack([s["targets"] for s in batch]).to(device)
        vol = torch.stack([s["vol"] for s in batch]).to(device)

        return inputs, targets, vol


import torch
import torch.nn.functional as F


def compute_per_asset_reward(pred, target):
    """
    pred, target: (batch, 3) deltas for [SOL, ETH, BTC]
    Simple directional reward: positive if aligned with actual move.
    """
 
    return (pred * target)  


def compute_vol_weight(vol_tensor, min_scale=0.5, max_scale=2.0):
    """
    vol_tensor: (batch, 3) or (3,) volatility values.
    Map volatility into a [min_scale, max_scale] band.
    Higher vol → higher penalty / lower effective reward.
    """
    
    v = torch.clamp(vol_tensor, 1e-6, None)
    norm = v / v.median()
   
    inv = 1.0 / norm
    return torch.clamp(inv, min_scale, max_scale)


def compute_drift_tensor(pred, target, vol_tensor, eps=1e-6):
    """
    pred, target: (batch, 3)
    vol_tensor: (batch, 3) or (3,)
    """
    diff = torch.abs(pred - target)
    vol_safe = torch.clamp(vol_tensor, eps, None)
    return diff / vol_safe  


def cvar_loss(loss_vec, alpha=0.9):
    """
    loss_vec: (batch,) scalar losses
    CVaR over the worst alpha tail.
    """
    q = torch.quantile(loss_vec.detach(), alpha)
    tail = loss_vec[loss_vec >= q]
    if tail.numel() == 0:
        return loss_vec.mean()
    return tail.mean()


def entropy_from_logits(logits):
    """
    If you ever convert deltas to logits / probs, you can use this.
    For now, we’ll keep it as a placeholder.
    """
   
    probs = F.softmax(logits, dim=-1)
    logp = torch.log(torch.clamp(probs, 1e-8, 1.0))
    return -(probs * logp).sum(dim=-1).mean()




def run_online_micro_training_step(
    model,
    optimizer,
    buffer,
    device,
    narrator,
    drift_threshold,
    batch_size=8,
):
    if len(buffer) < batch_size:
        narrator.narrate(
            f"⏳ Online buffer too small for RL micro‑step → {len(buffer)}/{batch_size}"
        )
        return None

    model.train()

    # x:   (B, 128, F)
    # y:   (B, 3, 6)   multi-asset, multi-horizon realized deltas
    # vol: (B, 3) or (B, 3, 6) true volatility per asset (and optionally horizon)
    x, y, vol = buffer.sample_batch_with_vol(batch_size, device)

    x = x.to(device)
    y = y.to(device)          # (B, 3, 6)
    vol = vol.to(device)      # (B, 3) or (B, 3, 6)

    # If vol is per-asset only, broadcast to horizons
    if vol.dim() == 2:        # (B, 3) → (B, 3, 6)
        vol = vol.unsqueeze(-1).expand_as(y)

    optimizer.zero_grad()

    # ------------------------------------------------------------
    # 0. Forward pass through multi-horizon model
    # ------------------------------------------------------------
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]

    if not isinstance(out, dict):
        raise RuntimeError("Model must return a dict with 'forecast_deltas' for online training.")

    try:
        pred = out["forecast_deltas"]["multi"]   # (B, 3, 6)
    except Exception as e:
        raise RuntimeError(f"Missing forecast_deltas['multi'] in model output: {e}")

    # ------------------------------------------------------------
    # 1. Dynamic volatility-aware clamping (per asset, per horizon)
    # ------------------------------------------------------------
    base_vol = torch.clamp(vol, min=1e-3)                     # (B, 3, 6)
    dynamic_max_mag = torch.clamp(5 * base_vol, 0.005, 0.05)  # (B, 3, 6)
    pred = torch.clamp(pred, -dynamic_max_mag, dynamic_max_mag)

    # ------------------------------------------------------------
    # 2. Supervised loss (MSE over assets and horizons)
    # ------------------------------------------------------------
    mse = torch.nn.MSELoss(reduction="none")
    per_elem_mse = mse(pred, y)                 # (B, 3, 6)
    per_sample_mse = per_elem_mse.mean(dim=(1, 2))  # (B,)
    sup_loss = per_sample_mse.mean()

    # ------------------------------------------------------------
    # 3. RL reward shaping (vol-normalized, error-based, multi-horizon)
    # ------------------------------------------------------------
    vol_safe = base_vol + 1e-6
    per_elem_reward = -torch.abs(pred - y) / vol_safe   # (B, 3, 6)
    reward_scalar = per_elem_reward.mean(dim=(1, 2))    # (B,)

    temp = 0.05
    rl_weight = 0.10
    drift_weight = 0.30
    cvar_weight = 0.10

    weights = torch.exp(reward_scalar / temp).detach()
    weights = torch.clamp(weights, 0.1, 5.0)
    rwr_loss = (weights * per_sample_mse).mean()

    # ------------------------------------------------------------
    # 4. Drift penalty (vector-aware)
    # ------------------------------------------------------------
    # compute_drift must now accept (B, 3, 6) and aggregate internally
    drift, drift_info = compute_drift(
        pred_delta=pred,      # (B, 3, 6)
        true_delta=y,         # (B, 3, 6)
        volatility=base_vol,  # (B, 3, 6)
        max_mag=0.05,
        regime=None,
    )
    drift_pen = drift  # scalar

    # ------------------------------------------------------------
    # 5. CVaR penalty over per-sample MSE
    # ------------------------------------------------------------
    cvar_pen = cvar_loss(per_sample_mse, alpha=0.9)

    # ------------------------------------------------------------
    # 6. Final blended loss
    # ------------------------------------------------------------
    loss = (
        sup_loss
        + rl_weight * rwr_loss
        + drift_weight * drift_pen
        + cvar_weight * cvar_pen
    )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # ------------------------------------------------------------
    # 7. Narration
    # ------------------------------------------------------------
    narrator.narrate(
        f"🏋️ Online RL micro‑step → loss={loss.item():.6f}, "
        f"sup={sup_loss.item():.6f}, rwr={rwr_loss.item():.6f}, "
        f"drift_pen={float(drift_pen):.6f}, cvar={cvar_pen.item():.6f}"
    )

    return loss.item()




def pick_global_regime(regimes):
                if "BREAKOUT" in regimes:
                    return "volatile"      
                if "VOLATILE" in regimes:
                    return "volatile"
                if "UP" in regimes or "DOWN" in regimes:
                    return "trend"
                if "FLAT" in regimes:
                    return "chop"
                return "chop"





def compute_forecast_delta(forecast_price, market_price):
    fp = Decimal(str(forecast_price))
    mp = Decimal(str(market_price))
    if mp <= 0:
        return Decimal("0")
    return (fp - mp) / mp



def build_online_sample(
    seq_tensor,
    pred_deltas,
    actual_deltas,
    multi_vol,
    narrator=None,
):
    """
    Build a single online training sample in the SAME format
    your online buffer expects, but now fully multi‑horizon:

    {
        "inputs":  (128, F),
        "targets": (3, 6),
        "pred":    (3, 6),
        "vol":     (3, 6)
    }
    """


    if seq_tensor.ndim != 2:
        if narrator:
            narrator.narrate(f"❌ seq_tensor malformed → ndim={seq_tensor.ndim}")
        return None

    T, F = seq_tensor.shape

    if T < 128:
        if narrator:
            narrator.narrate(f"❌ seq_tensor too short → {T} timesteps (need 128)")
        return None

    if T > 128:
        seq_tensor = seq_tensor[-128:]

    x = seq_tensor.detach().clone()
    device = x.device


    if isinstance(pred_deltas, torch.Tensor):
        pred_deltas = pred_deltas.detach().clone().to(device)
    else:
        pred_deltas = torch.as_tensor(pred_deltas, dtype=torch.float32, device=device)

    if pred_deltas.dim() == 1:
        pred_deltas = pred_deltas.unsqueeze(-1).repeat(1, 6)
    elif pred_deltas.dim() == 2 and pred_deltas.size(1) != 6:
        N = pred_deltas.size(1)
        if N > 6:
            pred_deltas = pred_deltas[:, :6]
        else:
            pad = 6 - N
            pred_deltas = torch.cat(
                [pred_deltas, torch.zeros(3, pad, device=device)], dim=1
            )


    if isinstance(actual_deltas, torch.Tensor):
        actual_deltas = actual_deltas.detach().clone().to(device)
    else:
        actual_deltas = torch.as_tensor(actual_deltas, dtype=torch.float32, device=device)

    if actual_deltas.dim() == 1:
        actual_deltas = actual_deltas.unsqueeze(-1).repeat(1, 6)
    elif actual_deltas.dim() == 2 and actual_deltas.size(1) != 6:
        N = actual_deltas.size(1)
        if N > 6:
            actual_deltas = actual_deltas[:, :6]
        else:
            pad = 6 - N
            actual_deltas = torch.cat(
                [actual_deltas, torch.zeros(3, pad, device=device)], dim=1
            )

    if isinstance(multi_vol, torch.Tensor):
        multi_vol = multi_vol.detach().clone().to(device)
    else:
        multi_vol = torch.as_tensor(multi_vol, dtype=torch.float32, device=device)

    if multi_vol.dim() == 1:
        multi_vol = multi_vol.unsqueeze(-1).repeat(1, 6)
    elif multi_vol.dim() == 2 and multi_vol.size(1) != 6:
        N = multi_vol.size(1)
        if N > 6:
            multi_vol = multi_vol[:, :6]
        else:
            pad = 6 - N
            multi_vol = torch.cat(
                [multi_vol, torch.zeros(3, pad, device=device)], dim=1
            )


    return {
        "inputs": x,
        "targets": actual_deltas,
        "pred": pred_deltas,
        "vol": multi_vol,
    }
def warm_start_online_buffer(
    online_buffer,
    offline_dataset,
    device,
    narrator,
    n_samples=24,
):
    narrator.narrate(f"🔥 Warm‑starting online buffer with {n_samples} historical samples")

    added = 0

    for _ in range(n_samples):
  
        raw = offline_dataset.sample_random(device=device)

        seq_tensor = raw["inputs"]     
        actual = raw["targets"]           

 
        if actual.dim() == 1:
            actual = actual.unsqueeze(-1).repeat(1, 6)
        elif actual.size(1) != 6:
    
            N = actual.size(1)
            if N > 6:
                actual = actual[:, :6]
            else:
                pad = 6 - N
                actual = torch.cat([actual, torch.zeros(3, pad, device=device)], dim=1)


        vol = torch.std(actual, dim=1).clamp(min=1e-6)   
        vol = vol.unsqueeze(-1).repeat(1, 6)            

      
        pred = torch.zeros(3, 6, device=device)

    
        sample = build_online_sample(
            seq_tensor=seq_tensor,
            pred_deltas=pred,
            actual_deltas=actual,
            multi_vol=vol,
            narrator=narrator,
        )

        if sample is not None:
            online_buffer.add(sample)
            added += 1

    narrator.narrate(f"✅ Warm‑start complete → {added} samples added.")
    
    ONLINE_CKPT  = r"D:\Scarlet_Works\Scarlet\checkpoints\online_best.ckpt"
OFFLINE_CKPT = r"D:\Scarlet_Works\Scarlet\checkpoints\bestmodel.ckpt"
def load_scarlet_model(model, narrator=None):

    if os.path.exists(OFFLINE_CKPT):
        ckpt = torch.load(OFFLINE_CKPT, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        if narrator:
            narrator.narrate("📥 Loaded offline foundation model.")

    
    if os.path.exists(ONLINE_CKPT):
        ckpt = torch.load(ONLINE_CKPT, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if narrator:
            narrator.narrate("📥 Loaded online drift‑aligned updates.")

    return model




def main(model, optimizer, lr_scheduler, loss_scheduler, epoch, device, narrator, global_scaler, symbols):
    global model_lock
    global offline_training_lock
    online_step = 0
    warmup_cycles = 16
    cycle_count = 0
    
    passive_scheduler = PassiveLearningScheduler(
        start_weight=0.0,     
        max_weight=0.10,    
        warmup_steps=10,    
        ramp_steps=500,      
        error_clip=0.02,     
    )
    print("🔥 ENTERED main()")
    symbols = ["solusd", "ethusd", "btcusd"]
    
   
    narrator = Narrator()
    reward_shaper = DirectionalRewardShaper(narrator)
    writer = SummaryWriter(log_dir="runs/scarlet_drift")

    model_lock = threading.Lock()
    offline_training_lock = threading.Lock()
    nonce_mgr = SmartNonceManager()

    memory_path = r"D:\Scarlet_Works\Scarlet\scarlet_memory.csv"
    CHECKPOINT_PATH = r"D:\Scarlet_Works\Scarlet\checkpoints\bestmodel.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   
    print(f"✅ Model loaded on: {device}")

  
    if device.type == "cuda":
        torch.randn(1, device=device)
        torch.cuda.synchronize()
        narrator.narrate("🔥 CUDA warm‑up complete — GPU context stabilized.")

    merged_df, df_scaled, scaler, numeric_cols = load_or_build_offline_df(lookback=30000)

    offline_dataset = MultiAssetWindowDataset(
        df_raw=merged_df,
        df_scaled=df_scaled,
        window_size=128,
        target_horizon=1,   
    )

        
    narrator.narrate("🧵 Micro‑training scheduler thread started.")

    threading.Thread(
        target=sentiment_loop,
        daemon=True
    ).start()
    narrator.narrate("💓 Sentiment thread launched.")
    
    
    
    threading.Thread(
        target=start_keyboard_listener,
        daemon=True
    ).start()




    print("🔁 Entered trading loop...")

 
    def update_learning(model, optimizer, forecast_delta, actual_delta, reward_value, narrator):
        narrator.narrate(
            f"🎓 Live sample collected → "
            f"forecast_delta={forecast_delta}, "
            f"actual_delta={actual_delta}, "
            f"reward={reward_value:.6f}"
        )
        return torch.tensor([reward_value], dtype=torch.float32)
    import inspect



    narrator.narrate("🚀 Scarlet startup — initializing caches")
    per_asset = {}
    for sym in symbols:
        per_asset[sym] = {
            "action": "HOLD",
            "amount": 0,
            "price": None,
            "forecast_delta": 0.0,
            "actual_delta": 0.0,
            "atr_value": 0.0,
            "regime": None,         
            "flat_score": 0.0,     
            "volatility": 0.0,      
        }

        trade_result = None
        initialize_asset_cache(sym, narrator=narrator)

    narrator.narrate("📚 Cache initialization complete — entering runtime loop")



    def run_multi_asset_trading_cycle(
        mode: str,
        model,
        merged_df,
        global_scaler,
        portfolio_posture,
        narrator,
        device,
        nonce_mgr,
        symbols=None,
        classifier=None,
        online_buffer=None,
    ):
        global POLICY_CONFIG
        if symbols is None:
            symbols = ["solusd", "ethusd", "btcusd"]

        for s in symbols:
            if cooldown_cycles[s] > 0:
                cooldown_cycles[s] -= 1

        previous_posture = {s: portfolio_posture.get(s) for s in symbols}
        previous_regime = {s: last_regime.get(s, None) for s in symbols}  
        previous_cooldown = {s: cooldown_cycles[s] for s in symbols}

        actual_deltas = [0.0, 0.0, 0.0]
        vol_vec = [0.0, 0.0, 0.0]
        results = {}
        per_asset = {}

        if symbols is None:
            symbols = ["solusd", "ethusd", "btcusd"]

        if classifier is None:
            classifier = FlatRegimeClassifier()

        if device.type == "cuda":
            try:
                torch.cuda.synchronize()
            except Exception as e:
                narrator.narrate(f"⚠️ CUDA fault detected mid‑cycle → switching to CPU: {e}")
                device = torch.device("cpu")

        engineered_df = engineer_features_multi(merged_df, narrator=narrator)

        features_tensor, enriched_df = build_feature_tensor(
            df=engineered_df,
            device=device,
            scaler=global_scaler,
            timeframe="15m",
            limit=None,
            narrator=narrator,
        )
        any_posture_changed = False
        any_trade_closed = False
        cooldown_expired = False
        large_forecast_error = False
        regime_shift_detected = False


        for sym_idx, sym in enumerate(symbols):
            trade_result = None

            per_asset_entry = per_asset.setdefault(sym, {
                "action": "HOLD",
                "amount": 0,
                "price": None,
                "forecast_delta": 0.0,
                "forecast_vec": None,
                "actual_delta": 0.0,
                "atr_value": 0.0,
                "regime": None,
                "flat_score": 0.0,
                "volatility": 0.0,
            })


            per_asset_entry["actual_delta"] = 0.0  

            base = sym.replace("usd", "").upper()
            results[sym] = per_asset_entry  
    
            context = {}

            if forced_signal["sol"] and sym == "solusd":
                context["forced"] = True
                context["forced_symbol"] = "solusd"
                forced_signal["sol"] = False
                narrator.narrate("⚡ Hotkey override → FORCED BUY for SOL")

            elif forced_signal["eth"] and sym == "ethusd":
                context["forced"] = True
                context["forced_symbol"] = "ethusd"
                forced_signal["eth"] = False
                narrator.narrate("⚡ Hotkey override → FORCED BUY for ETH")

            elif forced_signal["btc"] and sym == "btcusd":
                context["forced"] = True
                context["forced_symbol"] = "btcusd"
                forced_signal["btc"] = False
                narrator.narrate("⚡ Hotkey override → FORCED BUY for BTC")

            if forced_signal["panic"]:
                context["panic_exit"] = True
                narrator.narrate("🚨 Hotkey override → PANIC EXIT triggered")

    
            quote = get_market_price(sym)
            fallback = merged_df[f"close_{sym}"].iloc[-1]
            market_price = resolve_market_price(quote, fallback_close=fallback)
            market_price_dec = Decimal(str(market_price))

            narrator.narrate(f"📈 {base} market lock → {market_price_dec:.4f}")

            (
                forecast_price,
                forecast_delta,      
                forecast_strength,
                forecast_np,
                seq_tensor,
                current_price_tensor,
                atr_tensor,
                vwap_tensor,
                macd_line_tensor,
                signal_line_tensor,
                slope_tensor,
                sentiment_tensor,
                sentiment_slopes_tensor,
            ) = demo_inference(
                model,
                enriched_df,
                device,
                None,
                None,
                global_scaler,
                sym,
            )

            if forecast_price is None:
                narrator.narrate(f"⚠️ Inference failed for {base} — defaulting actual_delta=0")
                per_asset_entry["actual_delta"] = 0.0
                per_asset_entry["policy_delta"] = 0.0  
                results[sym] = per_asset_entry       
                continue


            if isinstance(forecast_delta, (float, int)):
        
                forecast_vec = np.array([forecast_delta] * 6, dtype=float)
            else:
                forecast_vec = np.asarray(forecast_delta, dtype=float).reshape(-1)
            if forecast_vec.shape[0] != 6:
         
                if forecast_vec.shape[0] > 6:
                    forecast_vec = forecast_vec[:6]
                else:
                    pad = 6 - forecast_vec.shape[0]
                    forecast_vec = np.concatenate([forecast_vec, np.zeros(pad, dtype=float)])

        
            if isinstance(forecast_strength, dict):
                forecast_strength = forecast_strength.get(sym, 0)
            forecast_strength = Decimal(str(forecast_strength))

            atr_col = f"ATR_{sym}"
            vol_col = f"volatility_{sym}"

            atr_value = Decimal("0")
            vol_value = Decimal("0")

            if atr_col in engineered_df.columns:
                try:
                    atr_value = Decimal(str(engineered_df[atr_col].iloc[-1]))
                except Exception:
                    atr_value = Decimal("0")

            if vol_col in engineered_df.columns:
                try:
                    vol_value = Decimal(str(engineered_df[vol_col].iloc[-1]))
                except Exception:
                    vol_value = Decimal("0")
            else:
                narrator.narrate(f"⚠️ Volatility column missing for {sym} — defaulting to 0")

            atr_ratio = atr_value / market_price_dec if market_price_dec > 0 else Decimal("0")

      
            recent_closes = merged_df[f"close_{sym}"].iloc[-12:]

     
            recent_actuals = []
            for i in range(1, len(recent_closes)):
                prev = Decimal(str(recent_closes.iloc[i - 1]))
                curr = Decimal(str(recent_closes.iloc[i]))
                recent_actuals.append((curr - prev) / prev)

            if len(recent_closes) >= 2:
                start = Decimal(str(recent_closes.iloc[0]))
                end = Decimal(str(recent_closes.iloc[-1]))
                actual_delta = (end - start) / start
            else:
                start = end = Decimal("0")
                actual_delta = Decimal("0")

            per_asset_entry["actual_delta"] = float(actual_delta)

            narrator.narrate(
                f"[ActualCheck:{base}] start={float(start):.4f}, "
                f"end={float(end):.4f}, "
                f"horizon_delta={float(actual_delta):.6f}"
            )

            sym_idx_map = {"solusd": 0, "ethusd": 1, "btcusd": 2}
            sym_idx = sym_idx_map[sym]
            actual_deltas[sym_idx] = per_asset_entry["actual_delta"]
            vol_vec[sym_idx] = float(vol_value)

   
            features = {
                "price": market_price_dec,
                "atr": atr_value,
                "macd_hist": Decimal(str(enriched_df[f"MACD_hist_{sym}"].iloc[-1])),
                "slope_10": Decimal(str(enriched_df[f"slope_10_{sym}"].iloc[-1])),
                "slope_30": Decimal(str(enriched_df[f"slope_30_{sym}"].iloc[-1])),
                "slope_40": Decimal(str(enriched_df[f"slope_40_{sym}"].iloc[-1])),
                "bb_width": Decimal(str(enriched_df[f"BB_width_{sym}"].iloc[-1])),
            }

            slope_10 = features["slope_10"]
            slope_30 = features["slope_30"]
            slope_40 = features["slope_40"]
            bb_width = features["bb_width"]
            macd_hist = features["macd_hist"]

            flat_score = classifier.classify(features, recent_actuals)
            if narrator:
                narrator.narrate(f"[Flatness:{base}] flat_score={float(flat_score):.2f}")

            regime = classify_regime(
                flat_score=flat_score,
                slope_10=slope_10,
                slope_30=slope_30,
                slope_40=slope_40,
                atr=atr_value,
                volatility=float(vol_value),
                bb_width=bb_width,
                macd_hist=macd_hist,
                recent_delta=recent_actuals,
                narrator=narrator,
                base=base,
            )

            flat_scale = (1 - float(flat_score))
            forecast_vec = forecast_vec * flat_scale
            forecast_strength *= (Decimal("1") - Decimal(str(flat_score)))


            horizon_weights = np.array([0.4, 0.3, 0.2, 0.15, 0.5, 0.3], dtype=float)
            horizon_weights /= horizon_weights.sum()
            policy_delta = float(np.dot(forecast_vec, horizon_weights))
            per_asset_entry["price"] = float(market_price_dec)
            per_asset_entry["forecast_vec"] = forecast_vec.tolist()
            per_asset_entry["forecast_delta"] = float(policy_delta)
            per_asset_entry["policy_delta"] = float(policy_delta)
            per_asset_entry["forecast_strength"] = float(forecast_strength)
            per_asset_entry["atr_value"] = float(atr_value)
            per_asset_entry["volatility"] = float(vol_value)
            per_asset_entry["flat_score"] = float(flat_score)
            per_asset_entry["regime"] = regime

            current_posture = portfolio_posture.get(sym)
            if current_posture != previous_posture[sym]:
                any_posture_changed = True

        
            if previous_regime[sym] is not None and previous_regime[sym] != regime:
                regime_shift_detected = True
            last_regime[sym] = regime  

            if cooldown_cycles[sym] == 0 and previous_cooldown[sym] > 0:
                cooldown_expired = True

            forecast_error = abs(policy_delta - float(actual_delta))
            if forecast_error > float(POLICY_CONFIG.error_threshold):
                large_forecast_error = True

            
            if trade_result and trade_result.get("execute") and trade_result.get("side") == "SELL":
                any_trade_closed = True


    
            wallet_asset = get_wallet_balance(base, nonce_mgr)
           
            DUST = Decimal("0.00001")

            posture = portfolio_posture.get(sym)
            posture.is_flat = flat_score > Decimal("0.7")
            

            policy_delta_dec = Decimal(str(per_asset_entry["policy_delta"]))
            policy_forecast_price = market_price_dec * (Decimal("1") + policy_delta_dec)

            action_infer, forecast_delta_pa = predict_action(
                market_price=Decimal(str(market_price_dec)),
                model_forecast=Decimal(str(policy_forecast_price)),
                policy_config=POLICY_CONFIG,
                is_flat=posture.is_flat,
                symbol=sym, 
            )


            narrator.narrate(
                f"[Policy:{base}] raw_action={action_infer}, "
                f"policy_forecast={float(policy_forecast_price):.2f}, "
                f"market={float(market_price_dec):.2f}, "
                f"Δ_policy={float(policy_delta_dec):.4f}"
            )
            try:
                prices = enriched_df[f"close_{sym}"].tail(75).to_numpy(dtype=float)
                volumes = enriched_df[f"volume_{sym}"].tail(75).to_numpy(dtype=float)

                if volumes.sum() > 0:
                    vwap_value = Decimal(str((prices * volumes).sum() / volumes.sum()))
                else:
                    vwap_value = Decimal(str(prices.mean()))
            except Exception:
                vwap_value = None

            forecast_delta = compute_forecast_delta(
                forecast_price=policy_forecast_price,
                market_price=market_price_dec
            )



            if str(action_infer).upper() == "BUY":
                buy_ok = buy_ok = buy_ok = is_buy_allowed(
                    forecast_delta=forecast_delta,
                    atr_ratio=atr_ratio,
                    volatility=vol_value,
                    market_price=market_price_dec,
                    vwap=vwap_value,
                    narrator=narrator,
                )



                if not buy_ok:
                    narrator.narrate(f"[Policy:{base}] BUY suppressed → HOLD")
                    action_infer = "HOLD"

            action_upper = str(action_infer).upper()

            flat_by_wallet = (wallet_asset is None) or (Decimal(str(wallet_asset)) <= DUST)
            flat_by_exposure = (posture.exposure is None) or (posture.exposure <= Decimal("0"))

            if action_upper == "SELL" and (flat_by_wallet or flat_by_exposure):
                action_upper = "HOLD"
                narrator.narrate(f"🧯 {base} flat/dust → SELL→HOLD")

            usd_balance = get_wallet_balance("USD", nonce_mgr)
            sol_balance = get_wallet_balance("SOL", nonce_mgr)
            eth_balance = get_wallet_balance("ETH", nonce_mgr)
            btc_balance = get_wallet_balance("BTC", nonce_mgr)
            
            sym_key = sym.lower()
          
            if sym == "solusd":
                asset_balance = sol_balance
            elif sym == "ethusd":
                asset_balance = eth_balance
            elif sym == "btcusd":
                asset_balance = btc_balance
            else:
                asset_balance = Decimal("0")
            

            safe_amount, sizing_suppressors = resolve_trade_size(
                action=action_upper,
                forecast_decimal=forecast_strength,
                market_price=market_price_dec,
                usd_balance=usd_balance,
                sol_balance=sol_balance,         
                asset_balance=asset_balance,    
                override_triggered=False,
                volatility=vol_value,
                atr_value=atr_value,
                limit_price=None,
                hard_stop_loss=False,
            )

            try:
                prev_close = Decimal(str(merged_df[f"close_{sym}"].iloc[-2]))
            except Exception:
                prev_close = None
                narrator.narrate(f"⚠️ Failed to extract previous_close for {sym}")
            if cooldown_cycles[sym] > 0:
                narrator.narrate(
                    f"⏳ {sym.upper()} cooldown active → {cooldown_cycles[sym]} cycles remaining"
                )

                
                per_asset_entry["action"] = "HOLD"
                per_asset_entry["amount"] = 0
                per_asset_entry["reward"] = 0.0
                per_asset_entry["forecast_delta"] = float(policy_delta)

                per_asset[sym] = per_asset_entry    

         
                results[sym] = {
                    "execute": False,
                    "reason": "cooldown_active",
                    "side": "HOLD",
                    "forecast_delta": float(policy_delta),
                    "actual_delta": None,
                    "reward": 0.0,
                }

                continue



            if mode == "eval":
                trade_result = execute_trade(
                    model_forecast=float(policy_forecast_price),
                    market_price=float(market_price_dec),
                    action_upper=action_upper,
                    symbol=sym,
                    price_tensor=features_tensor,
                    avg_entry_price=posture.avg_entry,
                    crypto_df=merged_df,
                    scaler=global_scaler,
                    trade_nonce_mgr=nonce_mgr,
                    override_triggered=False,
                    suppressors=sizing_suppressors,
                    amount=safe_amount,
                    adaptive_threshold=Decimal("0.02"),
                    portfolio_posture=posture,
                    latest_volatility=vol_value,
                    profit_log="scarlet_profit_log.csv",
                    memory_log="scarlet_memory.csv",
                    forecast_np=forecast_np,
                    entry_info=None,
                    forecast_strength=forecast_strength,
                    price_series=merged_df[f"close_{sym}"],
                    roi_reason=None,
                    slope_10=None,
                    slope_30=None,
                    slope_40=None,
                    forecast_delta=policy_delta, 
                    position_state=None,
                    sol_balance=sol_balance,
                    usd_balance=usd_balance,
                    asset_balance=asset_balance,
                    mode=mode,
                    narrator=narrator,
                    previous_close=prev_close,
                    context=context,
                )
            else:
                trade_result = {
                    "execute": False,
                    "reason": "train_mode_no_trading",
                    "forecast_delta": float(policy_delta_dec),
                    "actual_delta": None,
                    "reward": 0.0,
                }

      
            actual_delta_sym = per_asset[sym]["actual_delta"]

            if actual_delta_sym is None:
                actual_delta_sym = 0.0
                forecast_error = 0.0
            else:
                forecast_error = actual_delta_sym - float(policy_delta_dec)

  
        
        significant_event = (
            any_posture_changed
            or any_trade_closed
            or cooldown_expired
            or large_forecast_error
            or regime_shift_detected
        )

        if significant_event:
            narrator.narrate("📚 Triggering passive learning cycle...")
            POLICY_CONFIG = run_passive_learning(
                memory_log_path,
                POLICY_CONFIG,
                narrator,
            )   

       

            
            slope_val = (
                0.6 * enriched_df[f"slope_10_{sym}"].iloc[-1] +
                0.3 * enriched_df[f"slope_30_{sym}"].iloc[-1] +
                0.1 * enriched_df[f"slope_40_{sym}"].iloc[-1]
            )

      
            vwap_val = compute_rolling_vwap(merged_df, sym, window=96)
            if vwap_val is None:
                vwap_val = float(market_price_dec)
                narrator.narrate(f"⚠️ VWAP missing for {sym} — using market price fallback")

            per_asset_entry["action"] = trade_result.get("action", action_upper)
            per_asset_entry["amount"] = float(trade_result.get("amount", safe_amount))
            per_asset_entry["price"] = float(market_price_dec)
            per_asset_entry["forecast_delta"] = float(policy_delta)
            per_asset_entry["reward"] = float(trade_result.get("reward", 0.0))

            per_asset_entry["atr_value"] = float(atr_value)
            per_asset_entry["vwap_value"] = vwap_val
            per_asset_entry["macd_line"] = float(enriched_df[f"macd_line_{sym}"].iloc[-1])
            per_asset_entry["signal_line"] = float(enriched_df[f"signal_line_{sym}"].iloc[-1])
            per_asset_entry["slope"] = float(slope_val)

            per_asset_entry["regime"] = trade_result.get("regime")
            results[sym] = per_asset_entry

            regime_map = {
                "FLAT": 0,
                "UP": 1,
                "DOWN": 2,
                "VOLATILE": 3,
                "BREAKOUT": 4,
                "NORMAL": 5,
                None: -1,
            }
            regime = per_asset_entry["regime"]
            regime_idx = regime_map.get(regime, -1)
            regime_tensor = torch.tensor([regime_idx], dtype=torch.int64)

            sym_idx = {"solusd": 0, "ethusd": 1, "btcusd": 2}[sym]
            actual_deltas[sym_idx] = per_asset_entry["actual_delta"]
            vol_vec[sym_idx] = float(vol_value)

        inputs_tensor = seq_tensor.squeeze(0)       
        pred_tensor = torch.tensor(forecast_np)       

        targets_tensor = torch.tensor(actual_deltas, dtype=torch.float32).unsqueeze(-1).repeat(1, 6)
        symbol_vol_tensor = torch.tensor(vol_vec, dtype=torch.float32) 

        forced_signal["panic"] = False

        return {
            "per_asset": results,
            "merged_df": merged_df,
            "features_tensor": features_tensor,
            "enriched_df": enriched_df,
            "device": device,
        }

        
    portfolio_posture = PortfolioPosture()
    symbols = ["solusd", "ethusd", "btcusd"]

    if not hasattr(model, "best_loss"):
        model.best_loss = float("inf")

    for sym in symbols:
        update_asset_cache(sym, timeframe="15m", narrator=narrator)

        last_micro_train = time.time()
        last_train_row_count = 0 

    
        merged_df = fetch_and_align_assets(symbols, timeframe="15m")
        engineered = engineer_features_multi(merged_df, narrator=narrator)
        
       
         
        BUFFER_SIZE = 512
        MIN_SAMPLES = 64

        online_buffer = OnlineReplayBuffer(capacity=BUFFER_SIZE, min_samples=MIN_SAMPLES)
        warm_start_online_buffer(
            online_buffer=online_buffer,
            offline_dataset=offline_dataset,
            device=device,
            narrator=narrator,
            n_samples=64,
        )



        first_cycle = True

        try:
            with open("online_step.txt", "r") as f:
                online_step = int(f.read().strip())
        except:
            online_step = 0

        loss_scheduler = OnlineRLScheduler(
            rl_start=0.02,
            rl_max=0.10,
            warmup_steps=100,
            ramp_steps=1000,
        )


        try:
            with open("online_step.txt", "r") as f:
                online_step = int(f.read().strip())
        except Exception:
            online_step = 0

        loss_scheduler = OnlineRLScheduler(
            rl_start=0.02,
            rl_max=0.10,
            warmup_steps=100,
            ramp_steps=1000,
        )
    def build_online_sample(
        seq_tensor,
        pred_deltas,
        actual_deltas,
        multi_vol,
        narrator=None,
    ):
        """
        Build a single online training sample in the SAME format
        your online buffer expects, but now fully multi‑horizon:

        {
            "inputs":  (128, F),
            "targets": (3, 6),
            "pred":    (3, 6),
            "vol":     (3, 6)
        }
        """

        if seq_tensor.ndim != 2:
            if narrator:
                narrator.narrate(f"❌ seq_tensor malformed → ndim={seq_tensor.ndim}")
            return None

        T, F = seq_tensor.shape

        if T < 128:
            if narrator:
                narrator.narrate(f"❌ seq_tensor too short → {T} timesteps (need 128)")
            return None

        if T > 128:
            seq_tensor = seq_tensor[-128:]

        x = seq_tensor.detach().clone()
        device = x.device


        if isinstance(pred_deltas, torch.Tensor):
            pred_deltas = pred_deltas.detach().clone().to(device)
        else:
            pred_deltas = torch.as_tensor(pred_deltas, dtype=torch.float32, device=device)

        if pred_deltas.dim() == 1:
            pred_deltas = pred_deltas.unsqueeze(-1).repeat(1, 6)
        elif pred_deltas.dim() == 2 and pred_deltas.size(1) != 6:
            N = pred_deltas.size(1)
            if N > 6:
                pred_deltas = pred_deltas[:, :6]
            else:
                pad = 6 - N
                pred_deltas = torch.cat(
                    [pred_deltas, torch.zeros(3, pad, device=device)], dim=1
                )


        if isinstance(actual_deltas, torch.Tensor):
            actual_deltas = actual_deltas.detach().clone().to(device)
        else:
            actual_deltas = torch.as_tensor(actual_deltas, dtype=torch.float32, device=device)

        if actual_deltas.dim() == 1:
            actual_deltas = actual_deltas.unsqueeze(-1).repeat(1, 6)
        elif actual_deltas.dim() == 2 and actual_deltas.size(1) != 6:
            N = actual_deltas.size(1)
            if N > 6:
                actual_deltas = actual_deltas[:, :6]
            else:
                pad = 6 - N
                actual_deltas = torch.cat(
                    [actual_deltas, torch.zeros(3, pad, device=device)], dim=1
                )

 
        if isinstance(multi_vol, torch.Tensor):
            multi_vol = multi_vol.detach().clone().to(device)
        else:
            multi_vol = torch.as_tensor(multi_vol, dtype=torch.float32, device=device)

        if multi_vol.dim() == 1:
            multi_vol = multi_vol.unsqueeze(-1).repeat(1, 6)
        elif multi_vol.dim() == 2 and multi_vol.size(1) != 6:
            N = multi_vol.size(1)
            if N > 6:
                multi_vol = multi_vol[:, :6]
            else:
                pad = 6 - N
                multi_vol = torch.cat(
                    [multi_vol, torch.zeros(3, pad, device=device)], dim=1
                )


        return {
            "inputs": x,
            "targets": actual_deltas,
            "pred": pred_deltas,
            "vol": multi_vol,
        }

    while True:
        narrator.narrate("🔍 Running unified multi‑asset trading cycle...")

        if device.type == "cuda":
            try:
                torch.cuda.synchronize()
            except Exception as e:
                narrator.narrate(...)
                device = torch.device("cpu")

        for sym in symbols:
            update_asset_cache(sym, timeframe="15m", narrator=narrator)

        merged_df = fetch_and_align_assets(symbols, timeframe="15m")
        if merged_df.empty:
            narrator.narrate("⚠️ No candle data retrieved — skipping cycle.")
            time.sleep(60)
            continue

        engineered = engineer_features_multi(merged_df, narrator=narrator)

        global_scaler = RobustScaler().fit(
            engineered[INPUT_FEATURES].fillna(0.0)
        )

        results = run_multi_asset_trading_cycle(
            mode="eval",
            model=model,
            merged_df=engineered,
            global_scaler=global_scaler,
            portfolio_posture=portfolio_posture,
            narrator=narrator,
            device=device,
            nonce_mgr=nonce_mgr,
            symbols=symbols,
            online_buffer=online_buffer,
        )

        per_asset = results["per_asset"] 
        
        device = results.get("device", device)
       

        sol = per_asset["solusd"]
        eth = per_asset["ethusd"]
        btc = per_asset["btcusd"]


        if cycle_count < warmup_cycles:
            narrator.narrate(
                f"🛡 Warm‑up cycle {cycle_count}/{warmup_cycles} — HOLD override active."
            )
            for sym in results["per_asset"]:
                results["per_asset"][sym]["action"] = "HOLD"
                results["per_asset"][sym]["amount"] = 0

        

        summary = {
            sym: {
                "action": per_asset[sym]["action"],
                "amount": per_asset[sym]["amount"],
                "price": per_asset[sym]["price"],
                "forecast_delta": per_asset[sym]["forecast_delta"],
            }
            for sym in per_asset
        }
        narrator.narrate(f"📒 Cycle complete → {summary}")


        crypto_df = results["merged_df"]

        def to_scalar(x):
            try:
                return float(x)
            except Exception:
                return 0.0

        sol = per_asset["solusd"]
        eth = per_asset["ethusd"]
        btc = per_asset["btcusd"]

 
        sol_price = to_scalar(sol.get("price"))
        eth_price = to_scalar(eth.get("price"))
        btc_price = to_scalar(btc.get("price"))

 
        sol_forecast = to_scalar(sol.get("forecast_delta"))
        eth_forecast = to_scalar(eth.get("forecast_delta"))
        btc_forecast = to_scalar(btc.get("forecast_delta"))

        sol_actual = float(per_asset["solusd"]["actual_delta"])
        eth_actual = float(per_asset["ethusd"]["actual_delta"])
        btc_actual = float(per_asset["btcusd"]["actual_delta"])

        

        sol_atr = to_scalar(sol.get("atr_value"))
        eth_atr = to_scalar(eth.get("atr_value"))
        btc_atr = to_scalar(btc.get("atr_value"))

   
        invalid = False
        if sol_actual is None or eth_actual is None or btc_actual is None:
            narrator.narrate("⚠️ Missing actual_delta — skipping training sample.")
            invalid = True

        if sol_atr is None or eth_atr is None or btc_atr is None:
            narrator.narrate("⚠️ Missing ATR — skipping training sample.")
            invalid = True

        engineered = crypto_df  
        if len(engineered) < 128:
            narrator.narrate("⚠️ Not enough candles for 128‑window — skipping training sample.")
            invalid = True

     
        seq_tensor = torch.tensor(
            engineered.iloc[-128:][INPUT_FEATURES].values,
            dtype=torch.float32,
            device=device,
        )

        
        sol_vol = (float(sol_atr) / float(sol_price)) if sol_atr else 0.0
        eth_vol = (float(eth_atr) / float(eth_price)) if eth_atr else 0.0
        btc_vol = (float(btc_atr) / float(btc_price)) if btc_atr else 0.0

        multi_vol_tensor = torch.tensor(
            [sol_vol, eth_vol, btc_vol],
            dtype=torch.float32,
            device=device,
        )


        try:
            if invalid:
                narrator.narrate("⏳ Skipping drift check — shaping signals not ready.")
            else:
              
                pred_deltas = torch.tensor(
                    [sol_forecast, eth_forecast, btc_forecast],
                    dtype=torch.float32,
                    device=device,
                )
                actual_deltas = torch.tensor(
                    [sol_actual, eth_actual, btc_actual],
                    dtype=torch.float32,
                    device=device,
                )

                regimes = [per_asset[s]["regime"] for s in ["solusd", "ethusd", "btcusd"]]
                global_regime = pick_global_regime(regimes)

                drift, drift_info = compute_drift(
                    pred_delta=pred_deltas,       
                    true_delta=actual_deltas,       
                    volatility=multi_vol_tensor,   
                    max_mag=0.05,
                    regime=global_regime,
                )

                if drift > float(DRIFT_THRESHOLD):
                    narrator.narrate(
                        f"📈 Drift {drift:.4f} > {DRIFT_THRESHOLD:.4f} → online micro‑training step."
                    )
                    narrator.narrate(
                        f"🔍 Drift breakdown → "
                        f"mag={drift_info['mag_drift']:.4f}, "
                        f"dir={drift_info['dir_drift']:.4f}, "
                        f"conf={drift_info['conf_drift']:.4f}, "
                        f"vol={drift_info['volatility']:.6f}, "
                        f"regime={drift_info['regime']}"
                    )

                  
                    sol_vec = per_asset["solusd"].get("forecast_vec", [sol_forecast] * 6)
                    eth_vec = per_asset["ethusd"].get("forecast_vec", [eth_forecast] * 6)
                    btc_vec = per_asset["btcusd"].get("forecast_vec", [btc_forecast] * 6)

                    sample = build_online_sample(
                        seq_tensor=seq_tensor,
                        pred_deltas=[sol_vec, eth_vec, btc_vec],         
                        actual_deltas=[sol_actual, eth_actual, btc_actual],  
                        multi_vol=[sol_vol, eth_vol, btc_vol],            
                        narrator=narrator,
                    )

                    if sample is not None:
                        online_buffer.add(sample)

                    if len(online_buffer) >= MIN_SAMPLES:
                        drift_value = float(drift)  
                        threshold_value = float(DRIFT_THRESHOLD)  

                        epochs = int(min(20, max(1, drift_value / threshold_value)))
                        narrator.narrate(f"📈 Drift-triggered → running {epochs} micro‑epochs")

                        best_run_loss = float("inf")
                        best_run_state = None

                        for _ in range(epochs):
                            loss = run_online_micro_training_step(
                                model=model,
                                optimizer=optimizer,
                                buffer=online_buffer,
                                device=device,
                                narrator=narrator,
                                drift_threshold=DRIFT_THRESHOLD,
                            )

                            if loss is not None and loss < best_run_loss:
                                best_run_loss = loss
                                best_run_state = {k: v.clone() for k, v in model.state_dict().items()}

                        ONLINE_CKPT  = r"D:\Scarlet_Works\Scarlet\checkpoints\online_best.ckpt"

                        if best_run_state is not None:
                            model.load_state_dict(best_run_state)
                            save_micro_checkpoint(model, optimizer, best_run_loss, narrator,
                                                  path=ONLINE_CKPT)
                            narrator.narrate("🏅 Online micro‑training improved model — online checkpoint updated")

                        if loss is not None:
                            narrator.narrate(f"🏋️ Online micro‑training complete → loss={loss:.6f}")
                    else:
                        narrator.narrate(
                            f"⏳ Buffer not ready → {len(online_buffer)}/{MIN_SAMPLES} samples."
                        )
                else:
                    narrator.narrate(f"🟢 Drift {drift:.4f} within tolerance.")

        except Exception as e:
            import traceback
            narrator.narrate(f"⚠️ Drift block error: {e}")
            narrator.narrate(traceback.format_exc())

        narrator.narrate("⏳ Sleeping 16 minutes before next cycle...")
        cycle_count += 1
        time.sleep(16 * 60)





if __name__ == "__main__":
   
    input_dim = len(INPUT_FEATURES)

    model, optimizer, lr_scheduler, loss_scheduler, epoch = build_or_load_model(
        device=device,
        input_dim=input_dim,
        narrator=narrator,
    )

    model.to(device)

    
    main(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_scheduler=loss_scheduler,
        epoch=epoch,
        device=device,
        narrator=narrator,
        global_scaler=None,   # main will build this
        symbols=["solusd", "ethusd", "btcusd"],
    )


