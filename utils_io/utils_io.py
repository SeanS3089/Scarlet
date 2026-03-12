import os
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from collections.abc import Mapping
from torch.utils.data import Dataset

def update_crypto_data_file(order_book):
    """
    Fetch live order book data and save it to crypto_data.csv.
    Uses create_orderbook_dataframe() instead of random data.
    """
    df = create_orderbook_dataframe(order_book)
    
    if df.empty:
        print("⚠ Warning: Order book data is empty!")
    else:
        df.to_csv("crypto_data.csv", index=False)
        print("✓ crypto_data.csv updated with real order book data.")

    return True


def create_orderbook_dataframe(order_book, snapshot_time=None):
    """
    Convert the Level 3 order book response into a DataFrame.
    Supports different formats: dicts, objects with a pricebook attribute, etc.
    Returns a DataFrame with columns: timestamp, side, price, size, order_id.
    """
    if snapshot_time is None:
        snapshot_time = datetime.now().timestamp()
    
    bids = []
    asks = []
    
    if hasattr(order_book, "pricebook"):
        pb = order_book.pricebook
        try:
            from collections.abc import Mapping
            if isinstance(pb, Mapping):
                bids = pb.get("bids", [])
                asks = pb.get("asks", [])
                print("Using pricebook as mapping: bids count =", len(bids), "asks count =", len(asks))
            elif hasattr(pb, "__dict__"):
                pb_dict = pb.__dict__
                bids = pb_dict.get("bids", [])
                asks = pb_dict.get("asks", [])
                print("Using pricebook.__dict__: bids count =", len(bids), "asks count =", len(asks))
            else:
                print("pricebook exists but is not dict-like; no bids or asks extracted.")
        except Exception as e:
            print("Error processing pricebook:", e)
    elif isinstance(order_book, dict) and "pricebook" in order_book:
        pricebook_data = order_book["pricebook"]
        bids = pricebook_data.get("bids", [])
        asks = pricebook_data.get("asks", [])
        print("Using raw dict with 'pricebook' key: bids count =", len(bids), "asks count =", len(asks))
    elif hasattr(order_book, "get"):
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])
        print("Using dict interface on order_book: bids count =", len(bids), "asks count =", len(asks))
    else:
        bids = getattr(order_book, "bids", [])
        asks = getattr(order_book, "asks", [])
        print("Using attribute access on order_book: bids count =", len(bids), "asks count =", len(asks))
    
    data_rows = []
    def process_order_list(order_list, side):
        print(f"Processing {side} list with {len(order_list)} orders")
        for order in order_list:
            if isinstance(order, dict):
                try:
                    price = float(order.get("price", 0))
                    size = float(order.get("size", 0))
                except (TypeError, ValueError):
                    price, size = 0.0, 0.0
                order_id = order.get("order_id", None)
            elif isinstance(order, list):
                try:
                    price = float(order[0])
                    size = float(order[1])
                except (TypeError, ValueError):
                    price, size = 0.0, 0.0
                order_id = order[2] if len(order) > 2 else None
            elif hasattr(order, "price") and hasattr(order, "size"):
                try:
                    price = float(order.price)
                    size = float(order.size)
                except (TypeError, ValueError):
                    price, size = 0.0, 0.0
                order_id = getattr(order, "order_id", None)
            else:
                print("Skipping order of unexpected type:", type(order))
                continue
            data_rows.append({
                "timestamp": snapshot_time,
                "side": side,
                "price": price,
                "size": size,
                "order_id": order_id
            })
    process_order_list(bids, "bid")
    process_order_list(asks, "ask")
    print("Total orders processed:", len(data_rows))
    return pd.DataFrame(data_rows)


