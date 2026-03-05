import torch
from Scarlet_Core import (
    make_offline_dataloader,
    Narrator,
    LossScheduler,
    INPUT_FEATURES,
    POLICY_CONFIG,
)
from tqdm import tqdm
from decimal import Decimal
from hybrid_forecaster.hybrid_forecaster import HybridForecasterD
import os
from datetime import datetime
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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



class OfflinePolicyConfig:
    def __init__(self, buy_delta_threshold, min_roi_for_confident_buy, big_loss_threshold):
        self.buy_delta_threshold = buy_delta_threshold
        self.min_roi_for_confident_buy = min_roi_for_confident_buy
        self.big_loss_threshold = big_loss_threshold

offline_policy = OfflinePolicyConfig(
        buy_delta_threshold=Decimal("0.0012"),
        min_roi_for_confident_buy=Decimal("0.0008"),
        big_loss_threshold=Decimal("-0.0002"),
    )


def reward_offline_multiasset(
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
    device = forecast_delta.device
    B, A, H = forecast_delta.shape

    
    if realized_delta.dim() == 2:
        realized_delta = realized_delta.unsqueeze(-1).expand(-1, -1, H)
    elif realized_delta.dim() == 3 and realized_delta.size(2) != H:
        K = realized_delta.size(2)
        if K > H:
            realized_delta = realized_delta[:, :, :H]
        else:
            pad = H - K
            realized_delta = torch.cat(
                [realized_delta, torch.zeros(B, A, pad, device=device)],
                dim=2
            )

    
    def expand(x):
        return x.unsqueeze(-1).expand(-1, -1, H)

    current_price = expand(current_price)
    atr_value     = expand(atr_value)
    vwap_value    = expand(vwap_value)
    macd_line     = expand(macd_line)
    signal_line   = expand(signal_line)
    slope_value   = expand(slope_value)

   
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

    sell_mask = forecast_delta < 0
    sell_should_exec = sell_mask & (sell_profitable | sell_stop_loss | sell_indicator_exec)

  
    buy_delta_threshold = float(policy_config.buy_delta_threshold)
    roi_threshold       = float(policy_config.min_roi_for_confident_buy)

    forecast_mag = torch.abs(forecast_delta)

    forecast_strong = forecast_mag >= buy_delta_threshold
    roi_ok          = roi >= roi_threshold

    macd_bullish = macd_line > signal_line
    vwap_above   = future_price > vwap_value
    slope_up     = slope_value > 0.0
    atr_ok_buy   = atr_value > 0.0

    buy_mask = forecast_delta > 0
    buy_should_exec = (
        buy_mask
        & forecast_strong
        & roi_ok
        & macd_bullish
        & vwap_above
        & slope_up
        & atr_ok_buy
    )

    hold_should_exec = torch.zeros_like(buy_mask, dtype=torch.bool)

    should_execute = sell_should_exec | buy_should_exec | hold_should_exec

    reward = roi * direction_correctness

    reward = torch.where(
        should_execute,
        reward,
        reward * 0.25,
    )

    return reward




def compute_costbasis_loss_vectorized_offline(
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
    batch_tensor = batch_tensor.to(device)
    target_deltas = target_deltas.to(device)
    current_price = current_price.to(device)
    atr_value     = atr_value.to(device)
    vwap_value    = vwap_value.to(device)
    macd_line     = macd_line.to(device)
    signal_line   = signal_line.to(device)
    slope_value   = slope_value.to(device)

    out = model(batch_tensor)

    pred_deltas = out["forecast_deltas"]["multi"]  

    reward = reward_offline_multiasset(
        forecast_delta=pred_deltas,
        realized_delta=target_deltas,
        current_price=current_price,
        policy_config=offline_policy,
        atr_value=atr_value,
        vwap_value=vwap_value,
        macd_line=macd_line,
        signal_line=signal_line,
        slope_value=slope_value,
    )

    base_loss = -reward.mean()
    l2_term = 0.0001 * (pred_deltas ** 2).mean()

    return base_loss + l2_term



def run_offline_training(
    model,
    optimizer,
    writer,
    device,
    epochs=500,
    batch_size=256,
    scale=1.0,
    checkpoint_path=r"D:\Scarlet_Works\Scarlet\checkpoints\bestmodel.ckpt",
    narrator=None,
    make_dataloader=make_offline_dataloader,
    scheduler=None,
):
    """
    Unified offline training loop with:
    - LossScheduler support
    - CostBasisVerdict loss
    - Validation pass
    - Checkpointing on BEST VALIDATION LOSS

    Assumes:
        batch["inputs"]  -> (B, T, F)
        batch["targets"] -> (B, 3, 6)  # 3 assets × 6 horizons
    """

    assert make_dataloader is not None, "make_dataloader must be provided"


    train_loader, val_loader, input_features, output_features = make_dataloader(
        batch_size=batch_size,
        device=device,
    )

    model.train()
    best_loss = float("inf")
    last_epoch_loss = None
    prev_val_loss = None

  
    for epoch_idx in range(epochs):
        epoch_loss_acc = 0.0
        step_count = 0

  
        if scheduler:
            rl_weight, sampling_prob, supervised_batches = scheduler.update(
                epoch=epoch_idx,
                val_loss=prev_val_loss,
                prev_val_loss=prev_val_loss,
            )
        else:
            rl_weight = 1.0
            sampling_prob = 0.0
            supervised_batches = 999999

 
        progress = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch_idx+1}/{epochs}",
            ncols=100,
            leave=False,
        )

        for batch_idx, batch in progress:

            if scheduler and not scheduler.is_supervised_batch(batch_idx):
                continue

            seq_batch = batch["inputs"].to(device)     
            delta_batch = batch["targets"].to(device)  
            shaping = batch["shaping"]

            optimizer.zero_grad(set_to_none=True)

            supervised_loss = compute_costbasis_loss_vectorized_offline(
                model=model,
                batch_tensor=seq_batch,
                target_deltas=delta_batch,               
                current_price=shaping["current_price"],
                policy_config=offline_policy,
                atr_value=shaping["atr"],
                vwap_value=shaping["vwap"],
                macd_line=shaping["macd_line"],
                signal_line=shaping["signal_line"],
                slope_value=shaping["slope"],
                device=device,
            )

            
            if scheduler:
                reward_mult = scheduler.reward_multiplier(
                    epoch_idx,
                    val_loss=prev_val_loss,
                    prev_val_loss=prev_val_loss,
                )
                rl_loss = supervised_loss * reward_mult
                loss = supervised_loss + rl_weight * rl_loss
            else:
                reward_mult = 1.0
                rl_loss = torch.zeros_like(supervised_loss)
                loss = supervised_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss_acc += loss.item()
            step_count += 1

            progress.set_postfix(
                loss=f"{loss.item():.6e}",
                rl_w=f"{rl_weight:.3f}",
            )

        avg_train_loss = epoch_loss_acc / max(step_count, 1)
        last_epoch_loss = avg_train_loss

        print(
            f"[Epoch {epoch_idx+1}/{epochs}] "
            f"steps={step_count}, train_loss={avg_train_loss:.6e}"
        )

        if writer:
            writer.add_scalar("offline/train_loss", avg_train_loss, epoch_idx)

 
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss_acc = 0.0
            val_steps = 0

            with torch.no_grad():
                for batch in val_loader:
                    seq_batch = batch["inputs"].to(device)
                    delta_batch = batch["targets"].to(device)   # (B, 3, 6)
                    shaping = batch["shaping"]

                    vloss = compute_costbasis_loss_vectorized_offline(
                        model=model,
                        batch_tensor=seq_batch,
                        target_deltas=delta_batch,
                        current_price=shaping["current_price"],
                        policy_config=offline_policy,
                        atr_value=shaping["atr"],
                        vwap_value=shaping["vwap"],
                        macd_line=shaping["macd_line"],
                        signal_line=shaping["signal_line"],
                        slope_value=shaping["slope"],
                        device=device,
                    )

                    val_loss_acc += vloss.item()
                    val_steps += 1

            val_loss = val_loss_acc / max(val_steps, 1)
            prev_val_loss = val_loss
            model.train()

            print(f"🔍 Validation loss: {val_loss:.6e}")

            if writer:
                writer.add_scalar("offline/val_loss", val_loss, epoch_idx)


        metric = val_loss if val_loss is not None else avg_train_loss

        if metric < best_loss:
            best_loss = metric
            save_checkpoint(
                model,
                optimizer,
                epoch_idx + 1,
                checkpoint_path,
                narrator=None,
                phase="offline_epoch",
                loss=metric,
            )
            print(
                f"💾 Checkpoint saved → {checkpoint_path} "
                f"(epoch={epoch_idx+1}, best_val_loss={metric:.6e})"
            )

    print(
        f"✅ Offline training complete → epochs={epochs}, "
        f"last_train_loss={last_epoch_loss:.6e}, best_val_loss={best_loss:.6e}"
    )

    return last_epoch_loss


def build_model(input_dim: int):
    return HybridForecasterD(
        price_dim=input_dim,
        hidden_size=512,
        num_layers=2,
        output_seq_len=12,
        dropout=0.2,
        lr=1e-5,
        close_idx=0,
        use_gradient_checkpointing=False,
    ).to(DEVICE)



def train_offline_entrypoint():
    narrator = Narrator()
    offline_scheduler = LossScheduler()

 
    model = build_model(input_dim=len(INPUT_FEATURES))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    writer = None 

    

    run_offline_training(
        model=model,
        optimizer=optimizer,
        writer=writer,
        device=DEVICE,
        epochs=1000,      #1000 epochs seems to be the sweet spot       
        batch_size=256,
        narrator=narrator,
        scheduler=offline_scheduler,
        make_dataloader=make_offline_dataloader, 
    )



if __name__ == "__main__":
    train_offline_entrypoint()