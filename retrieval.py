"""Tiny TF-IDF retrieval over the manual and maintenance log (no heavy dependencies)."""
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data"
STOP = set("the a an and or of to in is are for with on at by be as it this that from if than then not no".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9\-\.]+", text.lower()) if t not in STOP and len(t) > 1]


def load_chunks() -> list[dict]:
    chunks = []
    manual = (DATA / "pump_manual.md").read_text(encoding="utf-8")
    section, title = [], "Manual"
    for line in manual.splitlines():
        if line.startswith("## "):
            if section:
                chunks.append({"source": f"Manual: {title}", "text": " ".join(section)})
            title, section = line[3:].strip(), []
        elif line.strip() and not line.startswith("# "):
            section.append(line.strip())
    if section:
        chunks.append({"source": f"Manual: {title}", "text": " ".join(section)})
    log = pd.read_csv(DATA / "maintenance_log.csv")
    for _, r in log.iterrows():
        chunks.append(
            {
                "source": f"Log {r['date']} {r['machine']}",
                "text": f"{r['machine']} {r['date']} {r['category']}: {r['note']} (downtime {r['downtime_hours']} h)",
            }
        )
    return chunks


class Index:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.docs = [Counter(tokenize(c["text"] + " " + c["source"])) for c in chunks]
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log((1 + n) / (1 + f)) + 1 for t, f in df.items()}

    def search(self, query: str, k: int = 4) -> list[dict]:
        q = Counter(tokenize(query))
        scored = []
        for chunk, doc in zip(self.chunks, self.docs):
            norm = math.sqrt(sum((c * self.idf.get(t, 0)) ** 2 for t, c in doc.items())) or 1.0
            score = sum(q[t] * self.idf.get(t, 0) * doc[t] * self.idf.get(t, 0) for t in q if t in doc) / norm
            if score > 0:
                scored.append({**chunk, "score": round(score, 3)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]
