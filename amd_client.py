"""Thin client for the AMD Token Factory endpoint (OpenAI-compatible chat completions).

All real model inference in this project goes through chat(). It records latency and
token usage so the UI can show the "Powered by AMD" panel.
"""
import json
import re
import time

import requests

import config

try:  # use the OS trust store (works behind corporate/antivirus TLS inspection; verification stays on)
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

CALL_LOG: list[dict] = []  # per-process log of calls (no secrets)


class AMDError(Exception):
    pass


class AMDRateLimited(AMDError):
    """429 after all retries: the key's limits are shared, so ask the user to wait."""


def _retry_after(r: requests.Response, attempt: int) -> float:
    try:
        return min(20.0, max(1.0, float(r.headers.get("Retry-After", ""))))
    except ValueError:
        return 2.0 * (attempt + 1)


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
            if attempt < retries - 1:
                time.sleep(_retry_after(r, attempt))
            continue
        if r.status_code in (401, 403):
            raise AMDError(f"HTTP {r.status_code}: the AMD API key was rejected. Check AMD_API_KEY.")
        if r.status_code == 404:
            raise AMDError("HTTP 404: check AMD_BASE_URL (it should end in /v1) and AMD_MODEL.")
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
        try:
            data = r.json()
            text = data["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError):
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
    if last == "HTTP 429":
        raise AMDRateLimited("The AMD endpoint is rate-limiting this key (HTTP 429, limits are shared). Wait ~30 s and try again.")
    raise AMDError(f"Endpoint unavailable after {retries} attempts ({last}).")


def parse_json(text: str) -> dict | None:
    """Extract a JSON object from model output; return None if malformed."""
    if not text:
        return None
    text = re.sub(r"<(think|thinking|reasoning)>.*?</\1>", "", text, flags=re.S | re.I)
    text = text.replace("“", '"').replace("”", '"').strip()
    candidates = [m.group(1) for m in re.finditer(r"```(?:json|JSON)?\s*(.*?)```", text, flags=re.S)]
    candidates.append(text)
    dec = json.JSONDecoder()
    found = []
    for c in candidates:
        for variant in (c, re.sub(r",\s*([}\]])", r"\1", c)):  # also try without trailing commas
            for m in re.finditer(r"\{", variant):
                try:
                    obj, _ = dec.raw_decode(variant, m.start())
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    found.append(obj)
            if found:
                # prefer the largest object (the answer, not a nested fragment or an example in reasoning)
                return max(reversed(found), key=lambda o: len(o))  # ties -> the last one (final answer)
    return None
