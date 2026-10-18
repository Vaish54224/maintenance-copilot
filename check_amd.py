"""Smoke test: one real request to the AMD endpoint. Prints status, latency, tokens (never the key)."""
import sys

import amd_client
import config

s = config.amd_settings()
print("endpoint:", s["base_url"] or "(not set)")
print("model   :", s["model"] or "(not set)")
print("key set :", bool(s["api_key"]))
try:
    r = amd_client.chat([{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=20)
except amd_client.AMDError as e:
    print("FAILED:", e)
    sys.exit(1)
print("reply   :", r["text"].strip()[:80])
print(f"latency : {r['latency_s']} s, tokens: {r['total_tokens']}")
print("OK")
