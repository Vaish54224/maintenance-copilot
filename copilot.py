"""LLM layer: grounded root-cause analysis (structured JSON) and technician chat.

Every call goes through the AMD endpoint via amd_client.chat().
"""
import amd_client
import detect
import retrieval

URGENCIES = ["low", "medium", "high", "critical"]
CONF_REVIEW_THRESHOLD = 0.6

SYSTEM_ANALYSIS = """You are a maintenance reliability engineer assistant for a factory.
Use ONLY the sensor evidence and the numbered reference excerpts provided. If the evidence does not support a conclusion, say so and lower your confidence.
Reply with ONE JSON object and nothing else, with exactly these keys:
{"cause": string, "confidence": number 0-1, "urgency": "low"|"medium"|"high"|"critical",
 "action": string (concrete next steps with timeframe), "safety_note": string,
 "sources": [integers = reference numbers you relied on], "uncertainty_note": string}"""

SYSTEM_CHAT = """You are a technician assistant. Answer ONLY from the numbered reference excerpts and sensor summary given.
If the answer is not in them, say you do not know and recommend asking a senior technician. Be concise (max 6 sentences). Cite references like [1], [2]."""


def sensor_evidence(status: dict) -> str:
    g = status["frame"]
    first = g.iloc[: detect.BASELINE_HOURS]
    last = g.iloc[-24:]
    lines = [f"Machine: {status['machine']}. Data window ends {g['timestamp'].iloc[-1]}."]
    for m in detect.METRICS:
        b, l = first[m].mean(), last[m].mean()
        if b == 0:
            continue
        pct = (l - b) / abs(b) * 100
        flag = status["drifted"].get(m)
        extra = f" DRIFT first flagged {flag['first_flag']}, latest z={flag['latest_z']}" if flag else ""
        lines.append(f"- {m}: baseline mean {b:.2f}, last-24h mean {l:.2f} ({pct:+.1f}%){extra}")
    return "\n".join(lines)


def _refs_block(hits: list[dict]) -> str:
    return "\n".join(f"[{i + 1}] ({h['source']}) {h['text']}" for i, h in enumerate(hits))


def normalize(obj: dict | None, hits: list[dict]) -> dict:
    """Validate/clean the model JSON. Malformed output becomes a safe human-review card."""
    if not obj or "cause" not in obj:
        return {
            "cause": "The model response could not be parsed.",
            "confidence": 0.0,
            "urgency": "medium",
            "action": "Review the sensor trends and sources below manually.",
            "safety_note": "Follow the lock-out/tag-out procedure before any work.",
            "sources": [h["source"] for h in hits],
            "uncertainty_note": "Malformed model output, treat as unverified.",
            "human_review": True,
            "parse_ok": False,
        }
    try:
        conf = max(0.0, min(1.0, float(obj.get("confidence", 0))))
    except (TypeError, ValueError):
        conf = 0.0
    urg = str(obj.get("urgency", "medium")).lower()
    urg = urg if urg in URGENCIES else "medium"
    srcs = []
    for s in obj.get("sources", []) or []:
        try:
            i = int(s) - 1
            if 0 <= i < len(hits):
                srcs.append(hits[i]["source"])
        except (TypeError, ValueError):
            if isinstance(s, str):
                srcs.append(s)
    return {
        "cause": str(obj.get("cause", "")),
        "confidence": conf,
        "urgency": urg,
        "action": str(obj.get("action", "")),
        "safety_note": str(obj.get("safety_note", "")),
        "sources": srcs or [h["source"] for h in hits[:2]],
        "uncertainty_note": str(obj.get("uncertainty_note", "")),
        "human_review": conf < CONF_REVIEW_THRESHOLD or urg in ("high", "critical"),
        "parse_ok": True,
    }


def analyze(status: dict, index: retrieval.Index, k: int = 5) -> dict:
    evidence = sensor_evidence(status)
    drift_terms = " ".join(status["drifted"].keys()).replace("_", " ")
    hits = index.search(f"{status['machine']} {drift_terms} rising fault cause action", k)
    user = f"SENSOR EVIDENCE:\n{evidence}\n\nREFERENCES:\n{_refs_block(hits)}\n\nDiagnose and recommend."
    resp = amd_client.chat(
        [{"role": "system", "content": SYSTEM_ANALYSIS}, {"role": "user", "content": user}],
        temperature=0.1,
        max_tokens=1200,
    )
    result = normalize(amd_client.parse_json(resp["text"]), hits)
    result["raw"] = resp["text"]
    result["call"] = {k2: resp[k2] for k2 in ("model", "endpoint", "latency_s", "prompt_tokens", "completion_tokens", "total_tokens")}
    result["evidence"] = evidence
    result["hits"] = hits
    return result


def ask(status: dict, question: str, index: retrieval.Index, k: int = 4) -> dict:
    evidence = sensor_evidence(status)
    hits = index.search(f"{status['machine']} {question}", k)
    user = f"SENSOR SUMMARY:\n{evidence}\n\nREFERENCES:\n{_refs_block(hits)}\n\nQUESTION: {question}"
    resp = amd_client.chat(
        [{"role": "system", "content": SYSTEM_CHAT}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=500,
    )
    return {"answer": resp["text"].strip(), "hits": hits, "call": {k2: resp[k2] for k2 in ("latency_s", "total_tokens")}}
