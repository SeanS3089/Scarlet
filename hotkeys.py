# hotkeys.py
import keyboard
import threading

# Shared signal dictionary
forced_signal = {
    "sol": False,
    "eth": False,
    "btc": False,
    "panic": False,
}

_lock = threading.Lock()

def _set_force(asset):
    with _lock:
        forced_signal[asset] = True
    print(f"[HOTKEY] Forced BUY triggered for {asset.upper()}")

def _set_panic():
    with _lock:
        forced_signal["panic"] = True
    print("[HOTKEY] PANIC EXIT triggered — SELL EVERYTHING")

def setup_hotkeys():
    """
    Registers global hotkeys.
    Call this once at Scarlet startup.
    """
    keyboard.add_hotkey("ctrl+a", lambda: _set_force("sol"))
    keyboard.add_hotkey("ctrl+s", lambda: _set_force("eth"))
    keyboard.add_hotkey("ctrl+d", lambda: _set_force("btc"))
    keyboard.add_hotkey("ctrl+p", _set_panic)

    print("[HOTKEY] Hotkeys active:")
    print("  Ctrl+A → Force SOL BUY")
    print("  Ctrl+S → Force ETH BUY")
    print("  Ctrl+D → Force BTC BUY")
    print("  Ctrl+P → PANIC EXIT (SELL ALL)")
