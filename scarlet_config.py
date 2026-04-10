from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List


@dataclass
class ScarletPolicyConfig:
    """
    Fully 4‑asset dynamic policy configuration.
    All per‑asset fields are auto‑generated from `symbols`,
    so adding/removing assets is a one‑line change.
    """

    # 🔥 The single source of truth for assets
    symbols: List[str] = field(default_factory=lambda: [
        "solusd", "ethusd", "btcusd", "xrpusd"
    ])

    # These will be filled dynamically in __post_init__
    buy_delta_threshold: Dict[str, Decimal] = field(init=False)
    sell_delta_threshold: Dict[str, Decimal] = field(init=False)
    hold_zone: Dict[str, Decimal] = field(init=False)

    micro_scalp_factor: Dict[str, Decimal] = field(init=False)
    micro_scalp_dust: Dict[str, Decimal] = field(init=False)
    min_trade_size: Dict[str, Decimal] = field(init=False)
    min_notional: Dict[str, Decimal] = field(init=False)
    buy_fraction: Dict[str, Decimal] = field(init=False)
    symbolic_micro_scalp_size: Dict[str, Decimal] = field(init=False)
    dust_thresholds: Dict[str, Decimal] = field(init=False)
    max_exposure: Dict[str, Decimal] = field(init=False)
    min_roi_for_confident_buy: Dict[str, Decimal] = field(init=False)
    min_roi_for_sell: Dict[str, Decimal] = field(init=False)
    stop_loss_threshold: Dict[str, Decimal] = field(init=False)
    exchange_minimums: Dict[str, Decimal] = field(init=False)

    # Global (non‑per‑asset) parameters
    micro_scalp_min_medium_slope: Decimal = Decimal("-0.01")
    cooldown_cycles_after_loss: int = 4
    cooldown_cycles_after_big_loss: int = 12
    cooldown_cycles_after_sell: int = 5
    big_loss_threshold: Decimal = Decimal("-0.02")

    atr_min_ratio: Decimal = Decimal("0.0020")
    atr_max_ratio: Decimal = Decimal("0.0500")
    atr_scaling_enabled: bool = True

    min_confidence: Decimal = Decimal("0.0")
    max_confidence: Decimal = Decimal("1.0")

    error_threshold: Decimal = Decimal("0.01")

    # ---------------------------------------------------------
    # 🔥 Auto‑populate all per‑asset dicts dynamically
    # ---------------------------------------------------------
    def __post_init__(self):

        # Core delta thresholds
        self.buy_delta_threshold = {s: Decimal("0.0012") for s in self.symbols}
        self.sell_delta_threshold = {s: Decimal("0.0008") for s in self.symbols}
        self.hold_zone = {s: Decimal("0.0002") for s in self.symbols}

        # Micro‑scalp behavior
        self.micro_scalp_factor = {s: Decimal("0.0100") for s in self.symbols}

        # Dust thresholds (asset‑specific tuning preserved)
        self.micro_scalp_dust = {
            "solusd": Decimal("0.0020"),
            "ethusd": Decimal("0.00005"),
            "btcusd": Decimal("0.0000005"),
            "xrpusd": Decimal("0.00002"),
        }

        # Minimum trade sizes
        self.min_trade_size = {
            "solusd": Decimal("0.0010"),
            "ethusd": Decimal("0.0002"),
            "btcusd": Decimal("0.0000007"),
            "xrpusd": Decimal("1.0"),   # XRP trades in whole-ish units
        }

        # Minimum notional (exchange requirement)
        self.min_notional = {s: Decimal("1.00") for s in self.symbols}

        # Buy fraction (position sizing)
        self.buy_fraction = {
            "solusd": Decimal("0.25"),
            "ethusd": Decimal("0.20"),
            "btcusd": Decimal("0.010"),
            "xrpusd": Decimal("0.25"),
        }

        # Symbolic micro‑scalp size
        self.symbolic_micro_scalp_size = {
            "solusd": Decimal("0.0010"),
            "ethusd": Decimal("0.0002"),
            "btcusd": Decimal("0.000002"),
            "xrpusd": Decimal("1.0"),
        }

        # Dust thresholds
        self.dust_thresholds = {
            "solusd": Decimal("0.0001"),
            "ethusd": Decimal("0.00005"),
            "btcusd": Decimal("0.0000005"),
            "xrpusd": Decimal("0.5"),
        }

        # Max exposure
        self.max_exposure = {
            "solusd": Decimal("0.2100"),
            "ethusd": Decimal("0.0200"),
            "btcusd": Decimal("0.0020"),
            "xrpusd": Decimal("200.0"),   # XRP is cheap
        }

        # ROI thresholds
        self.min_roi_for_confident_buy = {s: Decimal("0.0030") for s in self.symbols}
        self.min_roi_for_sell = {s: Decimal("0.05") for s in self.symbols}

        # Stop‑loss
        self.stop_loss_threshold = {s: Decimal("-0.02") for s in self.symbols}

        # Exchange minimums
        self.exchange_minimums = {
            "solusd": Decimal("0.01"),
            "ethusd": Decimal("0.001"),
            "btcusd": Decimal("0.00001"),
            "xrpusd": Decimal("1.0"),
        }
