import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

INPUT_SEQ_LEN = 75
OUTPUT_SEQ_LEN = 12

np.random.seed(42)

class CandleDatasetV2(Dataset):
    def __init__(
        self,
        df,
        INPUT_FEATURES,
        output_features,
        input_seq_len,
        output_seq_len,
        return_diff=False,
        mode=None,
    ):
        self.df_raw = df.copy()
        self.df = df.copy()

        for col in INPUT_FEATURES + output_features:
            if self.df[col].dtype == object:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
            if self.df_raw[col].dtype == object:
                self.df_raw[col] = pd.to_numeric(self.df_raw[col], errors="coerce")

        self.df = self.df.fillna(0.0)
        self.df_raw = self.df_raw.fillna(0.0)

        self.input_features = INPUT_FEATURES
        self.output_features = output_features
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.return_diff = return_diff
        self.external_mode = mode

    def __len__(self):
        return max(0, len(self.df) - self.input_seq_len - self.output_seq_len)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds")

        x = self.df[self.input_features].iloc[
            idx : idx + self.input_seq_len
        ].values.astype("float32")

        now_row = self.df_raw[self.output_features].iloc[
            idx + self.input_seq_len - 1
        ].values.astype("float32")

        future_row = self.df_raw[self.output_features].iloc[
            idx + self.input_seq_len + self.output_seq_len - 1
        ].values.astype("float32")

        returns = (future_row - now_row) / (now_row + 1e-8)
        y = returns.astype("float32")  

        if self.return_diff:
            x_diff = np.diff(x, axis=0)
            x = np.vstack([np.zeros((1, x.shape[1])), x_diff])

        last_row = self.df_raw.iloc[idx + self.input_seq_len - 1]

        def safe(col):
            return float(last_row[col]) if col in last_row else 0.0

        shaping = {
 
            "current_price_solusd": safe("close_solusd"),
            "current_price_ethusd": safe("close_ethusd"),
            "current_price_btcusd": safe("close_btcusd"),

  
            "ATR_solusd": safe("ATR_solusd"),
            "ATR_ethusd": safe("ATR_ethusd"),
            "ATR_btcusd": safe("ATR_btcusd"),

   
            "VWAP_solusd": safe("VWAP_solusd"),
            "VWAP_ethusd": safe("VWAP_ethusd"),
            "VWAP_btcusd": safe("VWAP_btcusd"),

     
            "macd_line_solusd": safe("macd_line_solusd"),
            "macd_line_ethusd": safe("macd_line_ethusd"),
            "macd_line_btcusd": safe("macd_line_btcusd"),


            "signal_line_solusd": safe("signal_line_solusd"),
            "signal_line_ethusd": safe("signal_line_ethusd"),
            "signal_line_btcusd": safe("signal_line_btcusd"),

       
            "MACD_hist_solusd": safe("MACD_hist_solusd"),
            "MACD_hist_ethusd": safe("MACD_hist_ethusd"),
            "MACD_hist_btcusd": safe("MACD_hist_btcusd"),

     
            "slope_10_solusd": safe("slope_10_solusd"),
            "slope_10_ethusd": safe("slope_10_ethusd"),
            "slope_10_btcusd": safe("slope_10_btcusd"),
        }


        

        return {
            "inputs": torch.tensor(x, dtype=torch.float32),  
            "targets": torch.tensor(y, dtype=torch.float32),  
            "shaping": shaping,
        }
   
CandleDataset = CandleDatasetV2