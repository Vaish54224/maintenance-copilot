"""Thin client for the AMD Token Factory endpoint (OpenAI-compatible chat completions).

All real model inference in this project goes through chat(). It records latency and
token usage so the UI can show the "Powered by AMD" panel.
"""
import json
import re
import time

import requests

import config

CALL_LOG: list[dict] = []  # per-process log of calls (no secrets)


class AMDError(Exception):
    pass


def chat(messages: list[dict], temperature: float = 0.2, max_tokens: int = 900, retries: int = 3) -> dict:
    s = config.amd_settings()
    if not (s["api_key"] and s["base_url"] and s["model"]):
        raise AMDError("AMD_API_KEY, AMD_BASE_URL and AMD_MODEL must all be set (see .env.example).")
    url = s["base_url"] + "/chat/completions"
    headers = {"Authorization": f"Bearer {s['api_key']}", "Content-Type": "application/json"}
    body = {"model": s["model"], "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    last = None
    merged = False
    for attempt in range(retries):
        t0 = time.perf_counter()
        try:
            r = requests.post(url, headers=headers, json=body, timeout=90)
        except requests.RequestException as e:
            last = f"network error: {type(e).__name__}"
            time.sleep(1.5 * (attempt + 1))
            continue
        latency = time.perf_counter() - t0
        if r.status_code == 429 or r.status_code >= 500:  # shared rate limits: back off
            last = f"HTTP {r.status_code}"
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 400 and not merged and any(m["role"] == "system" for m in body["messages"]):
            # Some Gemma chat templates reject a system role: fold it into the first user message and retry.
            merged = True
            sys_txt = "\n".join(m["content"] for m in body["messages"] if m["role"] == "system")
            rest = [m for m in body["messages"] if m["role"] != "system"]
            if rest and rest[0]["role"] == "user":
                rest[0] = {"role": "user", "content": sys_txt + "\n\n" + rest[0]["content"]}
            body["messages"] = rest
            continue
        if r.status_code != 200:
            raise AMDError(f"HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError):
            raise AMDError("Unexpected response shape from endpoint.")
        usage = data.get("usage", {}) or {}
        rec = {
            "model": s["model"],
            "endpoint": s["base_url"],
            "latency_s": round(latency, 2),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        CALL_LOG.append(rec)
        return {"text": text, **rec}
    raise AMDError(f"Endpoint unavailable after {retries} attempts ({last}).")


def parse_json(text: str) -> dict | None:
    """Extract a JSON object from model output; return None if malformed."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    candidates = [fence.group(1)] if fence else []
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None
