from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List


@dataclass
class ScarletPolicyConfig:
    """
    Modern, symbol‑aware configuration for Scarlet.
    Every field that affects trading behavior is now per‑asset,
    enabling passive learning and multi‑asset adaptation.
    """

 
    symbols: List[str] = field(default_factory=lambda: ["solusd", "ethusd", "btcusd"])


    buy_delta_threshold: Dict[str, Decimal] = field(default_factory=lambda: {
        "solusd": Decimal("0.0012"),
        "ethusd": Decimal("0.0012"),
        "btcusd": Decimal("0.0012"),
    })

    sell_delta_threshold: Dict[str, Decimal] = field(default_factory=lambda: {
        "solusd": Decimal("0.0008"),
        "ethusd": Decimal("0.0008"),
        "btcusd": Decimal("0.0008"),
    })

    hold_zone: Dict[str, Decimal] = field(default_factory=lambda: {
        "solusd": Decimal("0.0002"),
        "ethusd": Decimal("0.0002"),
        "btcusd": Decimal("0.0002"),
    })


    micro_scalp_factor: Dict[str, Decimal] = field(default_factory=lambda: {
        "solusd": Decimal("0.0100"),
        "ethusd": Decimal("0.0100"),
        "btcusd": Decimal("0.0100"),
    })

    micro_scalp_min_medium_slope: Decimal = Decimal("-0.01")


    micro_scalp_dust: Dict[str, Decimal] = field(default_factory=lambda: {
        "solusd": Decimal("0.0020"),
        "ethusd": Decimal("0.00005"),
        "btcusd": Decimal("0.0000005"),
    })

   
    min_trade_size: Dict[str, Decimal] = field(default_factory=lambda: {
        "solusd": Decimal("0.0010"),
        "ethusd": Decimal("0.0002"),
        "btcusd": Decimal("0.0000007"),
    })

    min_notional: Dict[str, Decimal] = field(default_factory=lambda: {
        "solusd": Decimal("1.00"),
        "ethusd": Decimal("1.00"),
        "btcusd": Decimal("1.00"),
    })

    buy_fraction: Dict[str, Decimal] = field(default_factory=lambda: {
        "solusd": Decimal("0.25"),
        "ethusd": Decimal("0.20"),
        "btcusd": Decimal("0.010"),
    })

    symbolic_micro_scalp_size: Dict[str, Decimal] = field(default_factory=lambda: {
        "solusd": Decimal("0.0010"),
        "ethusd": Decimal("0.0002"),
        "btcusd": Decimal("0.000002"),
    })

    dust_thresholds: Dict[str, Decimal] = field(default_factory=lambda: {
        "solusd": Decimal("0.0001"),
        "ethusd": Decimal("0.00005"),
        "btcusd": Decimal("0.0000005"),
    })

    max_exposure: Dict[str, Decimal] = field(default_factory=lambda: {
        "solusd": Decimal("0.2100"),
        "ethusd": Decimal("0.0200"),
        "btcusd": Decimal("0.0020"),
    })

    min_roi_for_confident_buy: Dict[str, Decimal] = field(default_factory=lambda: {
        "solusd": Decimal("0.0030"),
        "ethusd": Decimal("0.0030"),
        "btcusd": Decimal("0.0030"),
    })

    min_roi_for_sell: Dict[str, Decimal] = field(default_factory=lambda: {
        "solusd": Decimal("0.0112"),
        "ethusd": Decimal("0.0112"),
        "btcusd": Decimal("0.0112"),
    })

    cooldown_cycles_after_loss: int = 4
    cooldown_cycles_after_big_loss: int = 12
    cooldown_cycles_after_sell: int = 5
    big_loss_threshold: Decimal = Decimal("-0.02")

    atr_min_ratio: Decimal = Decimal("0.0020")
    atr_max_ratio: Decimal = Decimal("0.0500")
    atr_scaling_enabled: bool = True

    min_confidence: Decimal = Decimal("0.0")
    max_confidence: Decimal = Decimal("1.0")

    stop_loss_threshold: Dict[str, Decimal] = field(default_factory=lambda: {
        "solusd": Decimal("-0.02"),
        "ethusd": Decimal("-0.02"),
        "btcusd": Decimal("-0.02"),
    })

    exchange_minimums: Dict[str, Decimal] = field(default_factory=lambda: {
        "solusd": Decimal("0.01"),
        "ethusd": Decimal("0.001"),
        "btcusd": Decimal("0.00001"),
    })

    error_threshold: Decimal = Decimal("0.01")