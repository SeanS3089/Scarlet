#!/usr/bin/env python
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from collections import Counter
import torch.utils.checkpoint as checkpoint
import torch
import torch.nn as nn
import pytorch_lightning as pl
bnb = None
from decimal import Decimal

import torch
import torch.nn as nn
import pytorch_lightning as pl
from collections import Counter
from datetime import datetime
import random
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
class Narrator:
    def __init__(self, log_file="scarlet_narration_log.csv"):
        self.log_file = log_file

    def narrate(self, message):
        # Silent: no stdout, no interference with progress bar
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now()}, {message}\n")




class DirectionalRewardShaper:
    def __init__(
        self,
        narrator,
        max_reward=0.05,
        min_reward=-0.05,

        magnitude_threshold=0.005,
        gating_threshold=0.01,

        direction_penalty=0.01,
        direction_bonus_scale=0.01,

        scale=0.02,
        alpha=0.5,

        mag_weight=0.02,
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
        """
        Multi‑horizon version.
        forecast_delta: (H,) or (B,H)
        actual_delta:   (H,) or (B,H)
        volatility:     optional (H,) or (B,H)
        """

        
        forecast_delta = forecast_delta.float()
        actual_delta   = actual_delta.float()

        
        if volatility is None:
            volatility = actual_delta.abs() + self.eps
        else:
            volatility = volatility.float()

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


class BlockSparseAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        block_size=128,
        global_every=24,
        dropout=0.0,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.block_size = block_size
        self.global_every = global_every

    def make_block_sparse_mask(self, seq_len, device):
        """
        Returns an additive attention mask of shape [seq_len, seq_len]
        where 0 = allowed, -inf = disallowed.
        """
        mask = torch.full(
            (seq_len, seq_len),
            float("-inf"),
            device=device,
            dtype=torch.float32,   
        )

        num_blocks = (seq_len + self.block_size - 1) // self.block_size

        for b in range(num_blocks):
            start = b * self.block_size
            end = min((b + 1) * self.block_size, seq_len)

 
            mask[start:end, start:end] = 0.0


            if b % self.global_every == 0:
                mask[start:end, :] = 0.0
                mask[:, start:end] = 0.0

        return mask

    def forward(self, x):
        x = x.contiguous()

        """
        x: [batch, seq_len, embed_dim]
        """
        B, L, D = x.shape
        device = x.device

        attn_mask = self.make_block_sparse_mask(L, device=device)

        out, _ = self.attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            need_weights=False,
        )
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden=1024, dropout=0.1):
        super().__init__()


        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = BlockSparseAttention(
            embed_dim,
            num_heads,
            block_size=128,
            global_every=24,
        )
        self.attn_dropout = nn.Dropout(dropout)


        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x.contiguous()


        attn_in = self.norm1(x)
        attn_out = self.attn(attn_in)
        x = x + self.attn_dropout(attn_out)

        ff_in = self.norm2(x)
        ff_out = self.ff(ff_in)
        x = x + ff_out

        return x


from collections import deque

class LossScheduler:
    def __init__(
        self,
        rl_weight_start=0.5,
        rl_weight_max=1.0,
        rl_weight_growth=1.03,
        sched_sampling_start=0.3,
        sched_sampling_end=0.1,
        sched_sampling_decay=0.995,
        supervised_batches_per_epoch=10,
        min_supervised_batches=5,
        supervised_decay=0.995,
        reward_ramp=0.02,
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
        return ""

    def check_mid_epoch_balance(self, batch_idx: int, total_batches: int):
        halfway = total_batches // 2
        if batch_idx == halfway and not self.rl_ratio_met():
            return self.narrate_batch_balance()
        return None

def compute_delta_targets(current_close, future_close, horizons):


    device = future_close.device
    B = future_close.size(0)
    T = future_close.size(1)

    idx = torch.tensor([h - 1 for h in horizons], device=device)
    idx = torch.clamp(idx, 0, T - 1)

    if future_close.dim() == 2:
        fp = future_close[:, idx]         
    else:
        fp = future_close[:, idx, :]       

    if current_close.dim() == 1:
        delta = (fp - current_close.unsqueeze(1)) / current_close.unsqueeze(1)
    else:
        delta = (fp - current_close.unsqueeze(1)) / current_close.unsqueeze(1)

    return delta


import torch
import torch.nn as nn

class AssetEmbedder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 256,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed = nn.Linear(input_dim, embed_dim)

        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, embed_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.proj = nn.Linear(hidden_size, embed_dim)

        self.gate = nn.Linear(embed_dim * 2, embed_dim)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            [batch, embed_dim]
        """

        x = self.embed(x)  

        for layer in self.ff_layers:
            x = x + layer(x)

        lstm_out, (h, _) = self.lstm(x)
        h_last = h[-1] 
        lstm_embed = self.proj(h_last) 

        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)  

        weights = torch.softmax(attn_out.mean(dim=-1), dim=-1)  
        pooled = (attn_out * weights.unsqueeze(-1)).sum(dim=1)  

        fusion_input = torch.cat([lstm_embed, pooled], dim=-1) 
        gate = torch.sigmoid(self.gate(fusion_input))           
        combined = gate * lstm_embed + (1 - gate) * pooled      

        return self.norm(combined)



def shape_reward_tensor(
    base_reward,          
    action,               
    rsi,                  
    volatility,           
    rsi_zone,            
    signal_present,      
    cfg
):
 
    device = base_reward.device
    B, A = base_reward.shape


    rsi_buy_th  = cfg.get("rsi_buy_threshold", 30)
    rsi_sell_th = cfg.get("rsi_sell_threshold", 70)

    buy_bonus   = cfg.get("buy_bonus", 0.5)
    sell_bonus  = cfg.get("sell_bonus", 0.5)
    hold_bonus  = cfg.get("hold_bonus", 0.5)

    mistimed_penalty      = cfg.get("mistimed_penalty", 0.2)
    missed_signal_penalty = cfg.get("missed_signal_penalty", 0.1)
    hold_vol_max          = cfg.get("hold_volatility_max", 0.15)

    is_hold = (action == 0)
    is_sell = (action == 1)
    is_buy  = (action == 2)

    is_oversold   = rsi_zone["oversold"]    
    is_overbought = rsi_zone["overbought"]   

    bonus = torch.zeros_like(base_reward)

    buy_strength = torch.clamp((rsi_buy_th - rsi) / 100.0, min=0.0)
    bonus += is_buy * is_oversold * (buy_bonus * buy_strength)

    sell_strength = torch.clamp((rsi - rsi_sell_th) / 100.0, min=0.0)
    bonus += is_sell * is_overbought * (sell_bonus * sell_strength)

    vol_factor = torch.clamp(1 - (volatility / hold_vol_max), min=0.0)
    bonus += is_hold * (~signal_present) * hold_bonus * vol_factor

    overshoot  = torch.clamp((rsi - rsi_sell_th) / 100.0, min=0.0)
    undershoot = torch.clamp((rsi_buy_th - rsi) / 100.0, min=0.0)

    bonus -= is_buy * is_overbought * mistimed_penalty * overshoot

    bonus -= is_sell * is_oversold * mistimed_penalty * undershoot

    bonus -= is_hold * signal_present * missed_signal_penalty


    shaped_reward = base_reward + bonus

    shaped_reward = torch.clamp(shaped_reward, -1.0, 1.0)

    return shaped_reward, bonus
import torch

def adaptive_scale(
    base_rewards,          
    shaped_rewards,        
    min_scale=0.2,         
    max_scale=0.8,
    low=0.002,             
    high=0.02,
    clamp_output=True,
):
    if base_rewards.shape != shaped_rewards.shape:
        raise ValueError(
            f"Shape mismatch: base={base_rewards.shape}, shaped={shaped_rewards.shape}"
        )

    device = base_rewards.device

    base_std = base_rewards.std(unbiased=False)
    base_std = torch.clamp(base_std, min=1e-8)

    t = (base_std - low) / (high - low)
    t = torch.clamp(t, 0.0, 1.0)

    scale_factor = max_scale - t * (max_scale - min_scale)
    scale_factor = scale_factor.detach().clone()

    blended = base_rewards + (shaped_rewards - base_rewards) * scale_factor

    if clamp_output:
        blended = torch.clamp(blended, -1.0, 1.0)

    return blended, scale_factor



def compute_reward_with_adaptive_scaling(
    base_rewards,
    shaped_rewards,
    logger=None,
    narrator=None,
    clamp_output=True,
):

    blended, scale_factor = adaptive_scale(
        base_rewards,
        shaped_rewards,
        clamp_output=clamp_output,
    )

    if logger is not None:
        logger("rl/base_reward_mean", base_rewards.mean())
        logger("rl/shaped_reward_mean", shaped_rewards.mean())
        logger("rl/blended_reward_mean", blended.mean())
        logger("rl/adaptive_scale_factor", scale_factor)

    
        logger("rl/base_reward_std", base_rewards.std(unbiased=False))
        logger("rl/shaped_reward_std", shaped_rewards.std(unbiased=False))

    pass

    return blended, scale_factor



class HybridForecasterD(pl.LightningModule):
    def __init__(
        self,
        price_dim: int,
        hidden_size: int = 512,
        num_layers: int = 2,
        output_seq_len: int = 12,
        dropout: float = 0.2,
        lr: float = 1e-5,
        close_idx: int = 0,
        use_gradient_checkpointing: bool = True,
        narrator=None,
    ):
        super().__init__()
        self.price_dim = price_dim
        self.hidden_size = hidden_size
        self.output_seq_len = output_seq_len
        self.lr = lr
        self.close_idx = close_idx
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.narrator = narrator or Narrator()
        self.num_layers = num_layers
        self.dropout = dropout

        self.price_lstm = nn.LSTM(
            input_size=price_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=hidden_size,
                num_heads=4,
                ff_hidden=hidden_size * 4,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.decoder_input_projection = nn.Linear(price_dim, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.feature_transform = nn.Linear(hidden_size, hidden_size // 2)
        self.feature_dim = self.feature_transform.out_features
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.output_dropout = nn.Dropout(dropout)

        self.num_assets = 3  

        self.asset_embeddings = nn.Parameter(
            torch.randn(self.num_assets, self.feature_dim)
        )

        self.cross_asset_attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        self.price_head = nn.Linear(self.feature_dim, 1)
        self.delta_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 2, 6),
            nn.Tanh()
        )
        self.max_delta = 0.02


        self.strength_head = nn.Linear(self.feature_dim, 1)

        self.policy_head = nn.Linear(self.feature_dim, 3)
        self.softmax = nn.Softmax(dim=-1)
        self.best_loss = float("inf")

        self.encoder_dropout = nn.Dropout(p=0.10)    
        self.decoder_dropout = nn.Dropout(p=0.15)   
        self.head_dropout = nn.Dropout(p=0.10) 


    import os
    
    def generate_forecast(self, current_time, market_data):
        """
        Multi‑asset forecast generator for HybridForecaster‑D.
        Produces per‑asset forecasts and stores them in forecast_buffer.
        """

        target_time = current_time + self.forecast_horizon


        features = self.extract_features(market_data)       
        features = features.unsqueeze(0)                    

        pred, _ = self.forward(features)     

        out_np = pred.squeeze(0).detach().cpu().numpy()  

        ASSETS = ["solusd", "ethusd", "btcusd"]
        asset_forecasts = {}

        for i, sym in enumerate(ASSETS):
            price, delta, strength = out_np[i]
            asset_forecasts[sym] = {
                "price": float(price),
                "delta": float(delta),
                "strength": float(strength),
            }

        self.forecast_buffer.append({
            "timestamp": current_time,
            "forecast": asset_forecasts,
            "target_time": target_time,
        })
        sol = asset_forecasts["solusd"]["delta"]
        eth = asset_forecasts["ethusd"]["delta"]
        btc = asset_forecasts["btcusd"]["delta"]

        self.narrator.narrate(
            f"🔮 Forecasts for {target_time.strftime('%H:%M:%S')} → "
            f"SOL Δ={sol:+.4f}, ETH Δ={eth:+.4f}, BTC Δ={btc:+.4f}"
        )

        return asset_forecasts   

    def recover_supervised_signal(self):
        """
        Multi‑asset recovery logic.
        Forces full supervision when portfolio‑level delta correlation collapses.
        """

        recent_corrs = getattr(self, "epoch_delta_corrs", [])

        if not recent_corrs:
            return

        avg_corr = sum(recent_corrs) / len(recent_corrs)

        threshold = self.reward_cfg.get("delta_corr_recovery_threshold", 0.03)

        if avg_corr < threshold:

            if hasattr(self, "supervised_mask"):
                self.supervised_mask[:] = True

            self.delta_corr_last_portfolio = avg_corr

            


    def forward(self, x: torch.Tensor, decode_steps=12):
        """
        Produces: [B, num_assets, 3]  (price, delta, strength)
        """
        MAX_DELTA = 0.02

        # Always start with a contiguous input
        x = x.contiguous()

        B, T, _ = x.shape

        # --- PRICE LSTM ENCODER ---
        # Limit how much history the LSTM sees (e.g., last 2048 steps)
        max_lstm_len = 2048
        seq_len = min(T, max_lstm_len)

        price_input = x[:, -seq_len:, :self.price_dim].contiguous()


        


        if self.training and self.use_gradient_checkpointing:
            def lstm_forward(inp):
                out, _ = self.price_lstm(inp)
                return out

            enc_out = checkpoint.checkpoint(lstm_forward, price_input, use_reentrant=False)
            _, (h_price, _) = self.price_lstm(price_input)
        else:
            enc_out, (h_price, _) = self.price_lstm(price_input)

        # --- TRANSFORMER BLOCKS ---
        if self.training and self.use_gradient_checkpointing:
            enc_out.requires_grad_(True)
            for block in self.transformer_blocks:
                enc_out = checkpoint.checkpoint(block, enc_out, use_reentrant=False)
        else:
            for block in self.transformer_blocks:
                enc_out = block(enc_out)

        enc_out = self.encoder_dropout(enc_out)

        # --- SUMMARY VECTOR ---
        enc_summary = enc_out[:, -1, :]

        # --- DECODER ---
        features = self.feature_transform(enc_summary)
        features = self.output_dropout(features)

        dec_input = self.decoder_input_projection(x[:, -1, :]).unsqueeze(1)
        dec_input = self.decoder_dropout(dec_input)

        dec_hidden = h_price
        dec_outputs = []
        for _ in range(decode_steps):
            dec_out, dec_hidden = self.gru(dec_input, dec_hidden)
            dec_outputs.append(dec_out)
            dec_input = self.decoder_dropout(dec_out)

        dec_outputs = torch.cat(dec_outputs, dim=1)

        # --- FUSION ---
        fused = dec_outputs[:, -1, :] + enc_summary
        fused = self.layer_norm(fused)

        fused_features = self.feature_transform(fused)
        fused_features = self.head_dropout(fused_features)

        # --- CROSS-ASSET ATTENTION ---
        asset_feats = fused_features.unsqueeze(1) + self.asset_embeddings.unsqueeze(0)
        attn_out, _ = self.cross_asset_attention(asset_feats, asset_feats, asset_feats)

        # --- HEADS ---
        prices = self.price_head(attn_out).squeeze(-1)
        strengths = self.strength_head(attn_out).squeeze(-1)
        raw_deltas = self.delta_head(attn_out)
        deltas = raw_deltas * MAX_DELTA

        policy_logits = self.policy_head(attn_out)
        policy_probs = self.softmax(policy_logits)

        return {
            "prices": prices,
            "strengths": strengths,
            "deltas": deltas,
            "forecast_deltas": {"multi": deltas},
            "policy_probs": policy_probs,
        }

    def forward_returns(self, x):
        """
        Micro‑training forward path.
        Produces [B, 3] fractional returns for SOL, ETH, BTC,
        wrapped in a dict for the loss function.
        """

        MAX_DELTA = 0.02  

        price_input = x[..., :self.price_dim]
        enc_out, _ = self.price_lstm(price_input)

        for block in self.transformer_blocks:
            enc_out = block(enc_out)

        enc_summary = enc_out[:, -1, :]
        features = self.feature_transform(enc_summary)

        asset_feats = features.unsqueeze(1) + self.asset_embeddings.unsqueeze(0)
        asset_feats, _ = self.cross_asset_attention(asset_feats, asset_feats, asset_feats)

       
        raw_deltas = self.delta_head(asset_feats).squeeze(-1)  

     
        deltas = raw_deltas * MAX_DELTA

        return {
            "forecast_deltas": {
                "h10": deltas  
            }
        }



    def compute_diagnostic_reward(self, true_delta, actions):


        device = true_delta.device


        eps = 1e-6
        true_sign = torch.where(
            true_delta.abs() < eps,
            torch.zeros_like(true_delta),
            torch.sign(true_delta)
        )

        rewards = torch.zeros_like(true_delta)


        buy_mask = (actions == 2)
        rewards[buy_mask] = torch.where(
            true_sign[buy_mask] > 0,
            torch.ones_like(true_sign[buy_mask], device=device),
            -torch.ones_like(true_sign[buy_mask], device=device),
        )


        sell_mask = (actions == 1)
        rewards[sell_mask] = torch.where(
            true_sign[sell_mask] < 0,
            torch.ones_like(true_sign[sell_mask], device=device),
            -torch.ones_like(true_sign[sell_mask], device=device),
        )

        return rewards

    def compute_hybrid_reward(self, delta_pred, true_delta, action):


        delta_pred = delta_pred.to(true_delta.device)
        action     = action.to(true_delta.device)

        eps = 1e-6
        pred_sign = torch.where(
            delta_pred.abs() < eps,
            torch.zeros_like(delta_pred),
            torch.sign(delta_pred)
        )
        true_sign = torch.where(
            true_delta.abs() < eps,
            torch.zeros_like(true_delta),
            torch.sign(true_delta)
        )

        match_pred = (
            (action == 2) * (pred_sign > 0) +   
            (action == 1) * (pred_sign < 0) +  
            (action == 0) * (pred_sign == 0)  
        )

        match_true = (
            (action == 2) * (true_sign > 0) +
            (action == 1) * (true_sign < 0) +
            (action == 0) * (true_sign == 0)
        )

        reward = 0.5 * match_true.float() + 0.5 * match_pred.float()
        return reward    
    
    
    
    def compute_rsi_single(self, closes, period: int = 5):

        delta = closes[:, 1:] - closes[:, :-1]
        gain = torch.clamp(delta, min=0)
        loss = -torch.clamp(delta, max=0)

        if gain.size(1) < period:
            return torch.zeros(gain.size(0), device=gain.device)

        avg_gain = torch.nn.functional.avg_pool1d(gain.unsqueeze(1), period, stride=1).squeeze(1)
        avg_loss = torch.nn.functional.avg_pool1d(loss.unsqueeze(1), period, stride=1).squeeze(1)

        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi[:, -1]

    def compute_volatility_single(self, closes, period: int = 5):

        returns = closes[:, 1:] - closes[:, :-1]

        if returns.size(1) >= period:
            vols = returns.unfold(1, period, 1).std(dim=-1)
            return vols[:, -1]
        else:
            return returns.std(dim=1)
    
    def compute_slope_single(self, closes, window=None):

        B, T = closes.shape
        slopes = []

        eps = 1e-6 

        for i in range(B):
            y = closes[i]
            if window is not None and T >= window:
                y = y[-window:]

            y = y.float()
            N = len(y)

            if N < 2 or (y.max() - y.min()) < eps:
                slopes.append(torch.tensor(0.0, device=y.device))
                continue

            x = torch.arange(N, device=y.device).float()
            A = torch.stack([x, torch.ones_like(x)], dim=1)

            sol = torch.linalg.lstsq(A, y.unsqueeze(1)).solution
            slopes.append(sol[0].squeeze())

        return torch.stack(slopes, dim=0)

    def compute_reward_multi(self, actual_prices, actions, delta_pred, asset_idx, close_idx):

        closes = actual_prices[:, :, close_idx]  

        current_close = closes[:, 0]
        future_close  = closes[:, -1]
        true_delta = (future_close - current_close) / current_close.clamp(min=1e-8)

        rsi_vals  = self.compute_rsi_single(closes)
        vol_vals  = self.compute_volatility_single(closes)
        slope10   = self.compute_slope_single(closes, window=10)
        slope30   = self.compute_slope_single(closes, window=30)
        slope40   = self.compute_slope_single(closes, window=40)

        slope_mags = {
            10: slope10.abs(),
            30: slope30.abs(),
            40: slope40.abs(),
        }

        rsi_zone = {
            "oversold":   rsi_vals < 30,
            "overbought": rsi_vals > 70,
        }

        signal_present = (slope10.abs() > 1e-6)

        base_reward = true_delta * torch.where(actions == 2, 1,
                              torch.where(actions == 1, -1, 0))

        shaped_reward, _ = self.shape_reward_tensor(
            base_reward.unsqueeze(-1),      
            actions.unsqueeze(-1),          
            rsi_vals.unsqueeze(-1),         
            vol_vals.unsqueeze(-1),        
            rsi_zone,
            signal_present.unsqueeze(-1),   
            self.reward_cfg
        )

        return shaped_reward.squeeze(-1)


    def training_step(self, batch, batch_idx):

        is_supervised = self.loss_scheduler.is_supervised_batch(self.epoch_batch_index)
        if is_supervised:
            self.loss_scheduler.record_supervised_batch()
        else:
            self.loss_scheduler.record_rl_batch()

        self.epoch_batch_index += 1
        self.global_batch_index += 1

        x = batch["inputs"].to(self.device)   
        y = batch["targets"].to(self.device)  

        B, T, F = y.shape

    
        pred, policy_probs = self(x)          

        price_pred    = pred[:, :, 0]         
        delta_pred    = pred[:, :, 1]      
        strength_pred = pred[:, :, 2]         

      
        price_loss = 0.0

        for asset_idx, sym in enumerate(ASSETS):
            close_idx = self.input_features.index(f"close_{sym}")
            target_close = y[:, -1, close_idx]          
            pred_close   = price_pred[:, asset_idx]    
            price_loss += F.smooth_l1_loss(pred_close, target_close)

        price_loss /= len(ASSETS)

   
        flat_policy = policy_probs.view(B * 3, 3)      
        flat_actions = torch.multinomial(flat_policy, 1).squeeze(-1)  
        actions = flat_actions.view(B, 3)           


        rewards_all_assets = []

        for asset_idx, sym in enumerate(ASSETS):
            close_idx = self.input_features.index(f"close_{sym}")

            rewards_asset = self.compute_reward_multi(
                actual_prices=y,                
                actions=actions[:, asset_idx],   
                delta_pred=delta_pred[:, asset_idx],
                asset_idx=asset_idx,
                close_idx=close_idx,
            ) 

            rewards_all_assets.append(rewards_asset)

        rewards_all_assets = torch.stack(rewards_all_assets, dim=1) 
        rewards_shaped = rewards_all_assets.mean(dim=1)             

 
        if not hasattr(self, "reward_baseline"):
            self.reward_baseline = rewards_shaped.mean().detach()
        else:
            self.reward_baseline = (
                0.99 * self.reward_baseline + 0.01 * rewards_shaped.mean().detach()
            )

        advantages = rewards_shaped - self.reward_baseline 



        chosen_action_probs = policy_probs.gather(
            2, actions.unsqueeze(-1)
        ).squeeze(-1)  


        chosen_action_probs = chosen_action_probs.mean(dim=1)  

        rl_loss = -(advantages.detach() * torch.log(chosen_action_probs.clamp(min=1e-8))).mean()

        sup_weight = 1.0 if is_supervised else 0.2
        rl_weight  = self.rl_weight if not is_supervised else 0.2 * self.rl_weight

        total_loss = sup_weight * price_loss + rl_weight * rl_loss

        self.log("sup/price_loss", price_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("rl/loss", rl_loss, on_step=True, on_epoch=True)
        self.log("rl/avg_reward", rewards_shaped.mean(), prog_bar=True, on_step=True, on_epoch=True)

        return total_loss


    def validation_step(self, batch, batch_idx):
        x = batch["inputs"].to(self.device)  
        y = batch["targets"].to(self.device)   

        pred, policy_probs = self(x)           

        price_pred    = pred[:, :, 0]         
        delta_pred    = pred[:, :, 1]          
        strength_pred = pred[:, :, 2]      


        price_loss = 0.0
        for asset_idx, sym in enumerate(ASSETS):
            close_idx = self.input_features.index(f"close_{sym}")
            target_close = y[:, -1, close_idx]          
            pred_close   = price_pred[:, asset_idx]    
            price_loss += F.smooth_l1_loss(pred_close, target_close)

        price_loss /= len(ASSETS)

        actions = torch.argmax(policy_probs, dim=-1)    

        rewards_all_assets = []

        for asset_idx, sym in enumerate(ASSETS):
            close_idx = self.input_features.index(f"close_{sym}")

            rewards_asset = self.compute_reward_multi(
                actual_prices=y,                   
                actions=actions[:, asset_idx],      
                delta_pred=delta_pred[:, asset_idx], 
                asset_idx=asset_idx,
                close_idx=close_idx,
            )  

            rewards_all_assets.append(rewards_asset)

        rewards_all_assets = torch.stack(rewards_all_assets, dim=1) 
        rewards_portfolio = rewards_all_assets.mean(dim=1)          

        chosen_action_probs = policy_probs.gather(
            2, actions.unsqueeze(-1)
        ).squeeze(-1) 

        chosen_action_probs = chosen_action_probs.mean(dim=1) 

        val_rl_loss = -(rewards_portfolio * torch.log(chosen_action_probs.clamp(min=1e-8))).mean()

        if rewards_portfolio.numel() > 1:
            std = rewards_portfolio.std(unbiased=False).clamp(min=1e-8)
            val_avg_reward_norm = (rewards_portfolio.mean() / std).detach()
        else:
            val_avg_reward_norm = rewards_portfolio.mean().detach()


        self.log("val/sup/price_loss", price_loss, prog_bar=True, on_epoch=True)
        self.log("val/rl/loss", val_rl_loss, on_epoch=True)
        self.log("val/rl/avg_reward", rewards_portfolio.mean(), prog_bar=True, on_epoch=True)
        self.log("val/rl/avg_reward_norm", val_avg_reward_norm, prog_bar=True, on_epoch=True)


        for i, sym in enumerate(ASSETS):
            self.log(f"val/rl/reward_{sym}", rewards_all_assets[:, i].mean(), on_epoch=True)


        self.log("val/rl/action_hold", (actions == 0).float().mean(), on_epoch=True)
        self.log("val/rl/action_sell", (actions == 1).float().mean(), on_epoch=True)
        self.log("val/rl/action_buy",  (actions == 2).float().mean(), on_epoch=True)

        return {
            "val_loss": price_loss.detach(),
            "val_avg_reward_norm": val_avg_reward_norm,
            "val_avg_reward_raw": rewards_portfolio.mean().detach().item(),
        }



    def on_validation_epoch_end(self):


        if not hasattr(self, "validation_step_outputs") or len(self.validation_step_outputs) == 0:
            return

        val_losses = []
        val_reward_raw = []
        val_reward_norm = []

        for out in self.validation_step_outputs:
            if "val_loss" in out:
                val_losses.append(out["val_loss"])
            if "val_avg_reward_raw" in out:
                val_reward_raw.append(out["val_avg_reward_raw"])
            if "val_avg_reward_norm" in out:
                val_reward_norm.append(out["val_avg_reward_norm"])

        if len(val_reward_raw) == 0:
            self.validation_step_outputs.clear()
            return

        val_loss_mean = torch.tensor(val_losses).mean().item() if len(val_losses) else 0.0
        reward_raw_mean = torch.tensor(val_reward_raw).mean().item()
        reward_norm_mean = torch.tensor(val_reward_norm).mean().item()

        self.log("val/epoch_loss", val_loss_mean, on_epoch=True)
        self.log("val/epoch_reward_raw", reward_raw_mean, on_epoch=True)
        self.log("val/epoch_reward_norm", reward_norm_mean, on_epoch=True)


        if not hasattr(self, "best_val_reward"):
            self.best_val_reward = reward_raw_mean
            self.epochs_since_improve = 0
        else:
            if reward_raw_mean > self.best_val_reward + 1e-4:
                self.best_val_reward = reward_raw_mean
                self.epochs_since_improve = 0
                
            else:
                self.epochs_since_improve += 1


        plateau_patience = 5
        if self.epochs_since_improve >= plateau_patience:
            if not hasattr(self, "buy_sell_bonus"):
                self.buy_sell_bonus = 1.0

            self.buy_sell_bonus *= 1.25
            
            self.log("rl/buy_sell_bonus", self.buy_sell_bonus, on_epoch=True)

            self.epochs_since_improve = 0


        if hasattr(self, "scheduler"):
            rl_weight, sched_sampling_prob, supervised_batches = self.scheduler.update(
                epoch=self.current_epoch,
                val_loss=reward_raw_mean
            )

            self.narrator.narrate(self.scheduler.narrate_shaping(
                self.current_epoch, rl_weight, sched_sampling_prob, supervised_batches
            ))
            self.narrator.narrate(self.scheduler.narrate_phase(self.current_epoch))

            if hasattr(self, "optimizer"):
                floor_msg = self.scheduler.enforce_lr_floor(self.optimizer)
                if floor_msg:
                    self.narrator.narrate(floor_msg)

            self.log("shaping/rl_weight", rl_weight, on_epoch=True)
            self.log("shaping/sched_sampling_prob", sched_sampling_prob, on_epoch=True)
            self.log("shaping/supervised_batches", supervised_batches, on_epoch=True)


        self.validation_step_outputs.clear()
        

    def configure_optimizers(self):

        optimizer = bnb.optim.AdamW8bit(
            self.parameters(),
            lr=5e-6,
            weight_decay=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=3
        )

        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2
        )

        decay_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.98
        )

        

        return [optimizer], [
            {
                "scheduler": warmup_scheduler,
                "interval": "epoch",
                "frequency": 1
            },
            {
                "scheduler": plateau_scheduler,
                "monitor": "val/epoch_reward_raw", 
                "interval": "epoch",
                "frequency": 1
            },
            {
                "scheduler": decay_scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        ]

    def on_train_start(self):


        self.global_batch_index = 0
        self.epoch_batch_index = 0

        if hasattr(self, "validation_step_outputs"):
            self.validation_step_outputs.clear()

        if hasattr(self, "epoch_delta_corrs"):
            self.epoch_delta_corrs = []

        train_loader = self.trainer.datamodule.train_dataloader()
        self.num_batches = len(train_loader)

        if hasattr(self, "narrator"):
            self.narrator.narrate(
                f"🚀 Beginning training — {self.num_batches} batches per epoch."
            )

def on_train_epoch_start(self):
    import numpy as np

    self.supervised_batch_count = 0
    self.rl_batch_count = 0
    self.epoch_batch_index = 0

    val_reward_raw = self.trainer.callback_metrics.get("val/epoch_reward_raw", None)
    if val_reward_raw is not None:
        val_reward_raw = float(val_reward_raw)

    prev_val_reward = getattr(self, "prev_val_reward", None)

    rl_weight, sched_prob, sup_batches = self.loss_scheduler.update(
        self.current_epoch,
        val_loss=val_reward_raw
    )

    delta_corr = getattr(self, "delta_corr_last_portfolio", None)

    if delta_corr is not None:
        if delta_corr < 0.05:
            rl_weight *= 0.25
        elif delta_corr > 0.10:
            rl_weight *= 1.25

    self.rl_weight = rl_weight
    self.sched_sampling_prob = sched_prob

    train_loader = self.trainer.datamodule.train_dataloader()
    num_batches = len(train_loader)
    self.num_batches = num_batches

    num_supervised = int(np.clip(sup_batches, 1, max(1, num_batches - 1)))
    chosen = np.random.choice(num_batches, num_supervised, replace=False)
    mask = np.zeros(num_batches, dtype=bool)
    mask[chosen] = True

    cycle_length = 6
    phase = self.current_epoch % cycle_length
    phase_name = "mixed"

    if phase in [0, 1]:
        mask[:] = True
        phase_name = "compass"

    elif phase in [2, 3, 4]:
        phase_name = "mixed"

    else:
        if delta_corr is not None and delta_corr > 0.1:
            num_supervised = int(self.num_batches * 0.3)
            chosen = np.random.choice(self.num_batches, num_supervised, replace=False)
            mask = np.zeros(self.num_batches, dtype=bool)
            mask[chosen] = True
            phase_name = "rl_burst"
        else:
            mask[:] = True
            phase_name = "compass_recovery"

    self.supervised_mask = mask
    self.supervised_batches_per_epoch = mask.sum()


    recovery_active = False

    if getattr(self, "in_recovery", False):
        self.supervised_mask[:] = True
        self.in_recovery = False
        recovery_active = True

    elif delta_corr is not None and delta_corr < 0.05:
        self.supervised_mask[:] = True
        self.in_recovery = True
        recovery_active = True


    MIN_SUP_RATIO = self.reward_cfg.get("min_supervised_ratio", 0.10)
    min_sup_batches = max(1, int(self.num_batches * MIN_SUP_RATIO))

    if self.supervised_batches_per_epoch < min_sup_batches:
        additional = min_sup_batches - self.supervised_batches_per_epoch
        available = np.where(~self.supervised_mask)[0]

        if len(available) > 0:
            chosen_extra = np.random.choice(available, additional, replace=False)
            self.supervised_mask[chosen_extra] = True
            self.supervised_batches_per_epoch = self.supervised_mask.sum()

    self.log("rl_weight", rl_weight, prog_bar=True)
    self.log("sched_sampling_prob", sched_prob, prog_bar=True)
    self.log("supervised_batches_target", sup_batches)
    self.log("supervised_batches_planned", self.supervised_batches_per_epoch)
    self.log("rl/batches_planned", num_batches - self.supervised_batches_per_epoch)
    self.log("rl/recovery_active", float(recovery_active))

    phase_map = {"compass": 0, "mixed": 1, "rl_burst": 2, "compass_recovery": 3}
    self.log("train/phase_code", phase_map[phase_name])
    self.log("train/is_compass", float(phase_name == "compass"))
    self.log("train/is_mixed", float(phase_name == "mixed"))
    self.log("train/is_rl_burst", float(phase_name == "rl_burst"))
    self.log("train/is_recovery", float(phase_name == "compass_recovery"))

    self.prev_val_reward = val_reward_raw
    self.phase_name = phase_name
    self.phase_code = phase_map[phase_name]
    def on_train_epoch_end(self):
        sup_count = getattr(self, "supervised_batch_count", 0)
        rl_count  = getattr(self, "rl_batch_count", 0)
        total     = sup_count + rl_count


        if total == 0 or getattr(self.trainer, "sanity_checking", False):
            return

        sup_pct = sup_count / total * 100.0
        rl_pct  = rl_count  / total * 100.0

        hold_count = getattr(self, "epoch_hold_count", 0)
        sell_count = getattr(self, "epoch_sell_count", 0)
        buy_count  = getattr(self, "epoch_buy_count", 0)

        action_total = hold_count + sell_count + buy_count
        if action_total > 0:
            hold_pct = hold_count / action_total * 100.0
            sell_pct = sell_count / action_total * 100.0
            buy_pct  = buy_count  / action_total * 100.0
        else:
            hold_pct = sell_pct = buy_pct = 0.0

        val_reward_raw = self.trainer.callback_metrics.get("val/epoch_reward_raw")
        if val_reward_raw is not None:
            
            pass

        if hasattr(self, "epoch_delta_corrs") and self.epoch_delta_corrs:
            avg_corr = sum(self.epoch_delta_corrs) / len(self.epoch_delta_corrs)
            self.delta_corr_last_portfolio = avg_corr

            self.epoch_delta_corrs = []


        scheds = self.lr_schedulers()

        if isinstance(scheds, list):
            for sched in scheds:
                if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    metric = float(val_reward_raw) if val_reward_raw is not None else None
                    if metric is not None:
                        sched.step(metric)
                else:
                    sched.step()

            current_lr = scheds[0].optimizer.param_groups[0]["lr"]
        else:

            if isinstance(scheds, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric = float(val_reward_raw) if val_reward_raw is not None else None
                if metric is not None:
                    scheds.step(metric)
            else:
                scheds.step()

            current_lr = scheds.optimizer.param_groups[0]["lr"]

        
        self.log("lr", current_lr, prog_bar=True, on_epoch=True)


        self.supervised_batch_count = 0
        self.rl_batch_count = 0
        self.epoch_hold_count = 0
        self.epoch_sell_count = 0
        self.epoch_buy_count  = 0
