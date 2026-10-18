"""LLM layer: grounded root-cause analysis (structured JSON) and technician chat.

Every call goes through the AMD endpoint via amd_client.chat().
"""
import re

import amd_client
import detect
import retrieval

URGENCIES = ["low", "medium", "high", "critical"]
CONF_REVIEW_THRESHOLD = 0.6

SYSTEM_ANALYSIS = """You are a maintenance reliability engineer assistant for a factory.
Use ONLY the sensor evidence and the numbered reference excerpts provided. If the evidence does not support a conclusion, say so and lower your confidence.
Reply with ONE JSON object and nothing else: no reasoning, no markdown, no code fences. Use exactly these keys:
{"cause": string, "confidence": number 0-1, "urgency": "low"|"medium"|"high"|"critical",
 "action": string (concrete next steps with timeframe),
 "safety_note": string (specific precautions from the references: lock-out/tag-out, PPE, stop/escalation limits),
 "sources": [integers = reference numbers you relied on], "uncertainty_note": string}
Start your reply with { and end it with }."""

REPAIR = "Your previous reply was not valid JSON. Reply again with ONLY the JSON object with the required keys, starting with { and ending with }."

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
    raw_srcs = obj.get("sources", []) or []
    for s in raw_srcs if isinstance(raw_srcs, list) else [raw_srcs]:
        if isinstance(s, str) and re.fullmatch(r"\s*\[?\s*\d+\s*\]?\s*", s):
            s = re.sub(r"\D", "", s)  # "[2]" -> 2
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
    # always ground the safety note in the manual's safety section
    for h in index.search("safety lock out LOTO protection escalate stop", 3):
        if "safety" in h["source"].lower() and h["source"] not in {x["source"] for x in hits}:
            hits.append(h)
            break
    user = f"SENSOR EVIDENCE:\n{evidence}\n\nREFERENCES:\n{_refs_block(hits)}\n\nDiagnose and recommend."
    msgs = [{"role": "system", "content": SYSTEM_ANALYSIS}, {"role": "user", "content": user}]
    resp = amd_client.chat(msgs, temperature=0.1, max_tokens=1200)
    obj = amd_client.parse_json(resp["text"])
    raw = resp["text"]
    call = {k2: resp[k2] for k2 in ("model", "endpoint", "latency_s", "prompt_tokens", "completion_tokens", "total_tokens")}
    if not obj or "cause" not in obj:  # one repair attempt before falling back to the human-review card
        try:
            fix = amd_client.chat(
                msgs + [{"role": "assistant", "content": resp["text"]}, {"role": "user", "content": REPAIR}],
                temperature=0.0,
                max_tokens=1200,
            )
            obj, raw = amd_client.parse_json(fix["text"]), fix["text"]
            call["latency_s"] = round(call["latency_s"] + fix["latency_s"], 2)
            for t in ("prompt_tokens", "completion_tokens", "total_tokens"):
                call[t] += fix[t]
        except amd_client.AMDError:
            pass
    result = normalize(obj, hits)
    result["raw"] = raw
    result["call"] = call
    result["evidence"] = evidence
    result["hits"] = hits
    return result


def cited_sources(answer: str, hits: list[dict]) -> list[str]:
    """Sources the answer actually cites as [n]; falls back to all retrieved hits."""
    nums = []
    for m in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer):
        nums += [int(n) for n in m.group(1).split(",")]
    out = []
    for n in nums:
        if 1 <= n <= len(hits) and hits[n - 1]["source"] not in out:
            out.append(hits[n - 1]["source"])
    return out or [h["source"] for h in hits]


def ask(status: dict, question: str, index: retrieval.Index, k: int = 5) -> dict:
    evidence = sensor_evidence(status)
    q = question + (" safety LOTO" if re.search(r"safe|danger|hazard|risk", question, re.I) else "")
    hits = index.search(f"{status['machine']} {q}", k)
    user = f"SENSOR SUMMARY:\n{evidence}\n\nREFERENCES:\n{_refs_block(hits)}\n\nQUESTION: {question}"
    resp = amd_client.chat(
        [{"role": "system", "content": SYSTEM_CHAT}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=500,
    )
    answer = resp["text"].strip()
    return {
        "answer": answer,
        "hits": hits,
        "sources": cited_sources(answer, hits),
        "call": {k2: resp[k2] for k2 in ("latency_s", "total_tokens")},
    }
