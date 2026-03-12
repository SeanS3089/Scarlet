import os
import sys
import time
import hmac
import hashlib
import base64
import json
import re
from decimal import Decimal, ROUND_DOWN
from typing import Optional

import requests
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset
from dotenv import load_dotenv
from datetime import datetime


import importlib.util

gemini_utils_path = r"D:\Scarlet_Works\Scarlet\gemini_time_utils.py"
spec = importlib.util.spec_from_file_location("gemini_time_utils", gemini_utils_path)
gemini_time_utils = importlib.util.module_from_spec(spec)
sys.modules["gemini_time_utils"] = gemini_time_utils
spec.loader.exec_module(gemini_time_utils)

SmartNonceManager = gemini_time_utils.SmartNonceManager
trade_nonce_mgr = SmartNonceManager(nonce_file="trade_nonce.txt")
balance_nonce_mgr = SmartNonceManager(nonce_file="balance_nonce.txt")


load_dotenv("C:/Users/seans_1ymf/OneDrive/Documents/OneDrive/Documents/Desktop/Scarlet/.env")

API_URL = "https://api.gemini.com"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_SECRET = os.getenv("GEMINI_API_SECRET").encode()


from hybrid_forecaster import HybridForecasterD

class Narrator:
    def narrate(self, message: str):
        print(f"[Narration] {message}")

narrator = Narrator()
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


class CostBasisRecovery:
    """
    Thin wrapper around the unified cost‑basis engine (get_entry_info).
    Preserves the public API but delegates all logic to the new system.
    """

    def __init__(self, *paths: str, dust_threshold: float = 1e-6):
        self.paths = paths
        self.recovered_price: Optional[Decimal] = None
        self.recovery_note: str = "none_found"
        self.source_path: Optional[str] = None
        self.position_exited: bool = False
        self.dust_threshold = dust_threshold

    def set_position_exited(self, exited: bool):
        self.position_exited = exited

    def recover(self) -> Optional[Decimal]:
        if self.position_exited:
            print("🧹 Position exited — skipping cost basis recovery")
            self.recovery_note = "position_exited"
            return None

        print(f"🧪 Recovery paths: {self.paths}")

        from .get_entry_info import get_entry_info

        for path in self.paths:
            print(f"📁 Attempting to load: {path}")

            if not os.path.isfile(path):
                print(f"🚫 File not found: {path}")
                continue

            try:
                df = pd.read_csv(path, on_bad_lines="skip")

                if df.empty:
                    print(f"🚫 Empty memory file: {path}")
                    continue

                df.columns = [c.strip().lower() for c in df.columns]

                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(
                        df["timestamp"], errors="coerce", utc=True
                    )

                result = get_entry_info(df)

                avg = result.get("avg_entry")
                if avg is not None:
                    self.recovered_price = Decimal(str(avg))
                    self.recovery_note = result.get("commentary", "ok")
                    self.source_path = path

                    print(f"💡 Recovered avg_entry_price: {self.recovered_price}")
                    print(f"📝 Note: {self.recovery_note}")
                    return self.recovered_price

                print(f"🚫 No valid BUY entries in: {path}")

            except Exception as e:
                print(f"❌ Error reading {path}: {e}")

        self.recovery_note = "no_valid_entry_found"
        return None

    def has_buys_after_reset(self) -> bool:
        try:
            from .get_entry_info import get_entry_info

            df = pd.read_csv(self.paths[0], on_bad_lines="skip")
            df.columns = [c.strip().lower() for c in df.columns]
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

            result = get_entry_info(df)
            return result.get("valid_anchor", False)

        except Exception:
            return False

    def narrate(self) -> str:
        if self.recovered_price is not None:
            return (
                f"📈 Recovered avg_entry_price = {self.recovered_price} "
                f"from {self.source_path} ({self.recovery_note})"
            )
        else:
            return f"🚫 Recovery failed — avg_entry_price remains None ({self.recovery_note})"


def get_market_data(symbol="solusd", retries=3, delay=1):
    url = f"{API_URL}/v1/book/{symbol}"

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                print(f"⚠️ Gemini API returned status {response.status_code} on attempt {attempt+1}")
                time.sleep(delay * (attempt + 1))
                continue

            if not response.text.strip():
                print(f"⚠️ Empty response from Gemini API on attempt {attempt+1}")
                time.sleep(delay * (attempt + 1))
                continue

            book = response.json()

            market_data = {
                "bid": book["bids"][0]["price"],
                "ask": book["asks"][0]["price"],
                "bid_qty": book["bids"][0]["amount"],
                "ask_qty": book["asks"][0]["amount"],
                "timestamp": time.time(),
            }

            print(f"✅ Market data retrieved successfully on attempt {attempt+1}")
            return market_data

        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed on attempt {attempt+1}: {e}")
        except (KeyError, IndexError, ValueError) as e:
            print(f"❌ Data parsing error on attempt {attempt+1}: {e}")

        time.sleep(delay * (attempt + 1))

    print("🚫 All attempts to fetch market data failed")
    return None


def validate_avg_entry_price(avg_entry_price, override, recovery: CostBasisRecovery):
    if avg_entry_price is None and not override:
        print("⚠️ Suppressed trade due to missing avg_entry_price")
        return {
            "status": "error",
            "reason": "invalid_avg_entry_price",
            "note": recovery.recovery_note,
            "narration": recovery.narrate(),
        }
    elif avg_entry_price is None and override:
        print("⚠️ Proceeding with override despite missing avg_entry_price")
        return {"status": "override", "avg_entry_price": Decimal("0.0")}
    else:
        return {"status": "ok", "avg_entry_price": Decimal(str(avg_entry_price))}




def execute_trade_gemini(
    symbol="solusd",
    amount="0.002",
    action="buy",
    nonce_mgr=None,
    override=False,
):
    from decimal import Decimal, ROUND_DOWN

    action_lower = action.lower()
    print(f"🧪 Received action: {action} | amount: {amount} | override: {override}")

    if action_lower == "hold":
        print("⏸ HOLD action received → skipping Gemini order.")
        return {"status": "skipped", "reason": "hold_action_no_order"}


    response = requests.get(f"{API_URL}/v1/book/{symbol}")
    if response.status_code == 200 and response.text.strip():
        try:
            book = response.json()
            bid = Decimal(book["bids"][0]["price"])
            ask = Decimal(book["asks"][0]["price"])
            limit_price = bid if action_lower == "sell" else ask
        except (KeyError, IndexError, ValueError) as e:
            print(f"⚠️ Unexpected response structure: {e}")
            return {"status": "error", "reason": "gemini_response_structure_error"}
    else:
        print(f"❌ Gemini returned {response.status_code}: {response.text}")
        return {"status": "error", "reason": "gemini_empty_response"}

    endpoint = "/v1/order/new"
    url = API_URL + endpoint

    if nonce_mgr is None:
        nonce_mgr = SmartNonceManager(nonce_file="trade_nonce.txt")

    print(f"🧪 Interpreted action_lower: {action_lower}")


    dynamic_amt = Decimal(str(amount))
    if action_lower == "buy":
        print(f"🧠 BUY mode — using posture-aligned amount: {dynamic_amt}")

    if dynamic_amt <= 0:
        return {"status": "error", "reason": "invalid_trade_size"}


    sym_upper = symbol.upper()

    EXCHANGE_MIN = {
        "SOLUSD": Decimal("0.01"),
        "ETHUSD": Decimal("0.001"),
        "BTCUSD": Decimal("0.00001"),
    }.get(sym_upper, Decimal("0"))

    if sym_upper == "BTCUSD":
        precision = Decimal("0.00000001")
        fmt = "{:.8f}"
    elif sym_upper == "ETHUSD":
        precision = Decimal("0.00001")
        fmt = "{:.7f}"
    else:
        precision = Decimal("0.0001")
        fmt = "{:.4f}"


    dynamic_amt = dynamic_amt.quantize(precision, rounding=ROUND_DOWN)
    amount_str = fmt.format(dynamic_amt)

    if dynamic_amt < EXCHANGE_MIN:
        required_usd = EXCHANGE_MIN * limit_price

        try:
            wallet = get_wallet_snapshot(nonce_mgr)
            usd_balance = wallet.get("USD", Decimal("0"))

        except Exception:
            usd_balance = None

        if usd_balance is not None and usd_balance >= required_usd:
            print(
                f"🔧 Adjusting BUY amount from {dynamic_amt} to "
                f"exchange minimum {EXCHANGE_MIN} ({required_usd} USD required)"
            )
            dynamic_amt = EXCHANGE_MIN
            amount_str = fmt.format(dynamic_amt)
        else:
            print(
                f"🛑 Final BUY amount {dynamic_amt} below exchange minimum "
                f"{EXCHANGE_MIN} and insufficient USD → skipping execution."
            )
            return {
                "status": "error",
                "reason": "below_exchange_minimum",
                "final_amount": str(dynamic_amt),
            }

    last_error_response = None


    for nonce in nonce_mgr.get_safe_nonce_with_fallback(max_retries=3, delay_sec=3):
        payload = {
            "request": endpoint,
            "nonce": str(nonce),
            "symbol": sym_upper,
            "amount": amount_str,
            "price": f"{limit_price:.6f}",
            "side": action_lower,
            "type": "exchange limit",
            "account": "primary",
            "options": ["immediate-or-cancel"],
        }

        clean_payload = json.dumps(payload, separators=(",", ":"))
        print("📜 Encoded Payload:", clean_payload)

        b64 = base64.b64encode(clean_payload.encode())
        signature = hmac.new(GEMINI_API_SECRET, b64, hashlib.sha384).hexdigest()

        headers = {
            "X-GEMINI-APIKEY": GEMINI_API_KEY,
            "X-GEMINI-PAYLOAD": b64.decode(),
            "X-GEMINI-SIGNATURE": signature,
            "Content-Type": "text/plain",
        }

        print("📦 Final Payload:", payload)
        response = requests.post(url, headers=headers)
        print(f"📬 {response.status_code} → {response.text}")
        last_error_response = response.text

        if response.status_code == 200:
            print("✅ Trade successful:", response.json())
            nonce_mgr._save_nonce(int(nonce))
            return response.json()

        if "InvalidNonce" in response.text:
            print("⚠️ Invalid nonce detected.")
            match = re.search(r"Nonce must be greater than (\d+)", response.text)
            if match:
                required = int(match.group(1)) + 1
                print(f"🔁 Advancing nonce to: {required}")
                nonce_mgr._save_nonce(required)
            else:
                fallback = int(time.time() * 1000) + 10_000
                print(f"🔁 Forcing fallback nonce: {fallback}")
                nonce_mgr._save_nonce(fallback)
            continue

    recovery = int(time.time() * 1000) + 5_000
    print(f"🚫 Retries failed. Saving recovery nonce: {recovery}")
    nonce_mgr._save_nonce(recovery)

    if last_error_response:
        print("🧭 Last Gemini response:", last_error_response)

    return {"status": "error", "reason": "nonce_failure"}

def fetch_order_book(symbol="solusd", retries=3, delay=5):
    url = f"{API_URL}/v1/book/{symbol}"
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            if "application/json" not in response.headers.get("Content-Type", ""):
                raise ValueError("❌ Unexpected content type — not JSON.")
            return response.json()
        except (requests.exceptions.RequestException, ValueError, json.JSONDecodeError) as e:
            print(f"⚠️ Attempt {attempt+1} failed: {e}")
            time.sleep(delay)
    print(f"❌ All attempts to fetch order book from {url} failed.")
    return None


def create_orderbook_dataframe(order_book):
    data_rows = []

    def process(orders, side):
        for row in orders:
            try:
                if isinstance(row, dict):
                    price = float(row.get("price") or row.get("px") or row.get("0"))
                    size = float(row.get("amount") or row.get("qty") or row.get("1"))
                else:
                    price, size = float(row[0]), float(row[1])
                data_rows.append({"price": price, "amount": size, "side": side})
            except Exception as e:
                print(f"⚠️ Skipping malformed row in {side}s:", row, "| Error:", e)

    process(order_book.get("bids", []), "bid")
    process(order_book.get("asks", []), "ask")

    df_new = pd.DataFrame(data_rows)

    if df_new.empty:
        print("⚠️ Gemini returned an empty order book.")
    else:
        df_new = df_new.astype({"price": float, "amount": float, "side": str})
        df_new.sort_values(by=["side", "price"], ascending=[True, False], inplace=True)

    return df_new


FALLBACK_COLUMNS = ["price", "amount", "side"]


def safe_load_csv(csv_path, retries=3, delay=30):
    for attempt in range(retries):
        try:
            return pd.read_csv(csv_path, encoding="utf-8")
        except (pd.errors.EmptyDataError, PermissionError) as e:
            print(f"⚠️ Attempt {attempt+1} to load {csv_path} failed: {e}")
            time.sleep(delay)
    print(f"❌ All attempts to load {csv_path} failed. Falling back to empty DataFrame.")
    return pd.DataFrame(columns=FALLBACK_COLUMNS)


def update_crypto_data_file(symbol="SOLUSD", limit=1000, csv_path="crypto_data.csv"):
    if os.path.isfile(csv_path):
        try:
            df_existing = pd.read_csv(csv_path, encoding="utf-8")
        except (pd.errors.EmptyDataError, PermissionError):
            df_existing = pd.DataFrame(columns=["price", "amount", "side"])
    else:
        df_existing = pd.DataFrame(columns=["price", "amount", "side"])

    order_book = fetch_order_book(symbol=symbol)
    if not order_book:
        print("⚠️ No order book data fetched.")
        return df_existing

    df_new = create_orderbook_dataframe(order_book)
    if df_new.empty:
        print("⚠️ No new rows to append.")
        return df_existing

    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    if len(df_combined) > 21000:
        df_combined = df_combined.iloc[-21000:]

    tmp_path = csv_path + ".tmp"
    df_combined.to_csv(tmp_path, index=False, columns=["price", "amount", "side"])
    os.replace(tmp_path, csv_path)

    return df_combined