import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

INPUT_SEQ_LEN = 75
OUTPUT_SEQ_LEN = 12

np.random.seed(42)

class CandleDatasetV2(Dataset):
    def __init__(self, df, INPUT_FEATURES, output_features,
                 input_seq_len, output_seq_len, return_diff=False, mode=None):

        # Convert entire DF to a single float32 tensor once
        self.data = torch.tensor(df[INPUT_FEATURES].values, dtype=torch.float32)
        self.raw = torch.tensor(df[output_features].values, dtype=torch.float32)

        # Precompute shaping tensors (all rows)
        self.current_price = torch.tensor(df[[
            "close_solusd", "close_ethusd", "close_btcusd"
        ]].values, dtype=torch.float32)

        self.atr = torch.tensor(df[[
            "ATR_solusd", "ATR_ethusd", "ATR_btcusd"
        ]].values, dtype=torch.float32)

        self.vwap = torch.tensor(df[[
            "VWAP_solusd", "VWAP_ethusd", "VWAP_btcusd"
        ]].values, dtype=torch.float32)

        self.macd_line = torch.tensor(df[[
            "macd_line_solusd", "macd_line_ethusd", "macd_line_btcusd"
        ]].values, dtype=torch.float32)

        self.signal_line = torch.tensor(df[[
            "signal_line_solusd", "signal_line_ethusd", "signal_line_btcusd"
        ]].values, dtype=torch.float32)

        self.slope = torch.tensor(df[[
            "slope_10_solusd", "slope_10_ethusd", "slope_10_btcusd"
        ]].values, dtype=torch.float32)

        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.return_diff = return_diff

    def __len__(self):
        return len(self.data) - self.input_seq_len - self.output_seq_len

    def __getitem__(self, idx):
        i0 = idx
        i1 = idx + self.input_seq_len
        i2 = i1 + self.output_seq_len

        # Fast tensor slicing
        x = self.data[i0:i1]

        now = self.raw[i1 - 1]
        future = self.raw[i2 - 1]
        y = (future - now) / (now + 1e-8)

        shaping = {
            "current_price": self.current_price[i1 - 1],
            "atr": self.atr[i1 - 1],
            "vwap": self.vwap[i1 - 1],
            "macd_line": self.macd_line[i1 - 1],
            "signal_line": self.signal_line[i1 - 1],
            "slope": self.slope[i1 - 1],
        }

        return {"inputs": x, "targets": y, "shaping": shaping}   
CandleDataset = CandleDatasetV2