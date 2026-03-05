import os
import time
import requests
import re
import time
from time import time

import time
import os

class SmartNonceManager:
    def __init__(self, nonce_file="last_nonce.txt"):
        self.nonce_file = nonce_file

    def _load_last_nonce(self):
        if os.path.exists(self.nonce_file):
            try:
                with open(self.nonce_file, "r") as f:
                    contents = f.read().strip()
                    return int(contents) if contents else 0
            except Exception as e:
                print(f"⚠️ Failed to load nonce: {e}")
        return 0

    def _save_nonce(self, nonce):
        with open(self.nonce_file, "w") as f:
            f.write(str(nonce))

    def get_safe_nonce_with_fallback(self, max_retries=3, delay_sec=3):
        for attempt in range(max_retries):
            try:
                last_nonce = self._load_last_nonce()
                current_time_ms = int(time.time() * 1000) + 1000
                next_nonce = max(current_time_ms, last_nonce + 1)

                with open("nonce_debug.log", "a") as log:
                    log.write(f"[RETRY-{attempt+1}] Nonce: {next_nonce} | LocalTime: {current_time_ms}\n")
                print(f"🧾 Yielding nonce: {next_nonce}")

                yield str(next_nonce)
                break  # Let the caller decide if it should be saved

            except Exception as e:
                print(f"⚠️ Failed to generate nonce: {e}. Retrying in {delay_sec}s...")
                time.sleep(delay_sec)

        else:
            raise RuntimeError("❌ Failed to generate nonce after retries.")

       


        

       

