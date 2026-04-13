import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

INPUT_SEQ_LEN = 75
OUTPUT_SEQ_LEN = 12

np.random.seed(42)

class CandleDatasetV2(Dataset):
    """
    Minimal patch: precompute sequences, targets, and shaping once.
    __getitem__ becomes O(1).
    """

    def __init__(self, df, INPUT_FEATURES, output_features,
                 input_seq_len, output_seq_len, return_diff=False, mode=None,
                 device="cuda"):

        self.device = device
        self.seq_len = input_seq_len
        self.out_len = output_seq_len

        # Convert DF to numpy once
        X_np = df[INPUT_FEATURES].to_numpy(dtype=np.float32)
        Y_np = df[output_features].to_numpy(dtype=np.float32)

        N = len(df) - input_seq_len - output_seq_len

        # ----------------------------------------------------
        # 1. Precompute all input sequences [N, T, F]
        # ----------------------------------------------------
        seqs = np.lib.stride_tricks.sliding_window_view(
            X_np,
            window_shape=(input_seq_len, X_np.shape[1])
        )
        seqs = seqs[:N, 0]  # shape: [N, T, F]

        # ----------------------------------------------------
        # 2. Precompute all targets [N, A]
        # ----------------------------------------------------
        now  = Y_np[input_seq_len - 1 : input_seq_len - 1 + N]
        fut  = Y_np[input_seq_len - 1 + output_seq_len : input_seq_len - 1 + output_seq_len + N]
        targets = (fut - now) / (now + 1e-8)

        # ----------------------------------------------------
        # 3. Precompute shaping [N, 3]
        # ----------------------------------------------------
        shaping_keys = {
            "current_price": ["close_solusd", "close_ethusd", "close_btcusd"],
            "atr":           ["ATR_solusd", "ATR_ethusd", "ATR_btcusd"],
            "vwap":          ["VWAP_solusd", "VWAP_ethusd", "VWAP_btcusd"],
            "macd_line":     ["macd_line_solusd", "macd_line_ethusd", "macd_line_btcusd"],
            "signal_line":   ["signal_line_solusd", "signal_line_ethusd", "signal_line_btcusd"],
            "slope":         ["slope_10_solusd", "slope_10_ethusd", "slope_10_btcusd"],
        }

        shaping = {}
        for key, cols in shaping_keys.items():
            arr = df[cols].to_numpy(dtype=np.float32)
            shaping[key] = arr[input_seq_len - 1 : input_seq_len - 1 + N]

        # ----------------------------------------------------
        # Move everything to GPU once
        # ----------------------------------------------------
        self.inputs  = torch.tensor(seqs,    dtype=torch.float32, device=device)
        self.targets = torch.tensor(targets, dtype=torch.float32, device=device)

        self.shaping = {
            k: torch.tensor(v, dtype=torch.float32, device=device)
            for k, v in shaping.items()
        }

        self.N = N

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # O(1) indexing — no slicing, no CPU work
        return {
            "inputs":  self.inputs[idx],
            "targets": self.targets[idx],
            "shaping": {k: v[idx] for k, v in self.shaping.items()},
        }


CandleDataset = CandleDatasetV2