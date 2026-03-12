from .hybrid_forecaster import HybridForecasterD
import pytorch_lightning as pl


class HybridBalancer(pl.Callback):
    def __init__(self, rl_weight_start=0.01, rl_weight_max=0.5,
                 supervised_min_ratio=0.2, patience=3, decay_factor=0.7):
        super().__init__()
        self.rl_weight = rl_weight_start
        self.rl_weight_max = rl_weight_max
        self.supervised_min_ratio = supervised_min_ratio
        self.patience = patience
        self.decay_factor = decay_factor
        self.best_val = None
        self.bad_epochs = 0

    def on_validation_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is None:
            return

        if self.best_val is None or val_loss < self.best_val:
            self.best_val = val_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        if self.bad_epochs >= self.patience:
            old_weight = getattr(pl_module, "rl_weight", self.rl_weight)
            new_weight = max(old_weight * self.decay_factor, 0.01)
            pl_module.rl_weight = new_weight
            self.rl_weight = new_weight
            self.bad_epochs = 0
            print(f"[Balancer] ⚖️ RL weight adjusted {old_weight:.4f} → {new_weight:.4f}")

        if hasattr(pl_module, "supervised_ratio"):
            if pl_module.supervised_ratio < self.supervised_min_ratio:
                print(f"[Balancer] 🧩 Supervised ratio raised to floor {self.supervised_min_ratio}")
                pl_module.supervised_ratio = self.supervised_min_ratio


