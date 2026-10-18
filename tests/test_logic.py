"""Offline tests for the non-LLM logic. Run: python -m pytest -q   (or python tests/test_logic.py)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import amd_client
import copilot
import detect
import generate_data
import retrieval


def setup_module(_=None):
    generate_data.main()


def test_only_m3_flagged_and_earlier_than_fixed_alarm():
    dfz = detect.add_zscores(detect.load_sensors())
    for m in ["M1-Press", "M2-Conveyor", "M4-Compressor", "M5-Mixer"]:
        assert not detect.machine_status(dfz, m)["drifted"]
    assert detect.machine_status(dfz, "M3-Pump")["drifted"]
    cmp_ = detect.baseline_vs_ai(dfz)
    assert cmp_["hours_earlier"] > 12


def test_retrieval_finds_bearing_content():
    idx = retrieval.Index(retrieval.load_chunks())
    top = idx.search("bearing vibration temperature rising", 3)
    assert top and any("Bearing" in h["source"] or "M3-Pump" in h["source"] for h in top)


def test_json_parsing_variants():
    assert amd_client.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert amd_client.parse_json('<think>x</think> text {"a": 2} tail') == {"a": 2}
    assert amd_client.parse_json("not json at all") is None
    assert amd_client.parse_json('{"a": ') is None


def test_normalize_malformed_and_review_rules():
    hits = [{"source": "Manual: X", "text": "t"}]
    bad = copilot.normalize(None, hits)
    assert bad["human_review"] and not bad["parse_ok"]
    ok = copilot.normalize({"cause": "c", "confidence": 0.9, "urgency": "low", "sources": [1]}, hits)
    assert not ok["human_review"] and ok["sources"] == ["Manual: X"]
    hi = copilot.normalize({"cause": "c", "confidence": 0.9, "urgency": "critical"}, hits)
    assert hi["human_review"]
    lo = copilot.normalize({"cause": "c", "confidence": 0.3, "urgency": "low"}, hits)
    assert lo["human_review"]
    weird = copilot.normalize({"cause": "c", "confidence": "abc", "urgency": "???"}, hits)
    assert weird["confidence"] == 0.0 and weird["urgency"] == "medium"


if __name__ == "__main__":
    setup_module()
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
