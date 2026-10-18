# 🛠️ Maintenance Copilot

AMD Developer Hackathon: ACT III · Track: Intelligent Industry

## Problem
Factories produce huge amounts of sensor data, maintenance logs and manuals, but fixed-threshold alarms only fire when a machine is already close to failing. Technicians then lose hours finding the right manual page or past work order. Unplanned downtime commonly costs thousands of dollars per hour.

## User
Maintenance engineers and shift supervisors on a manufacturing floor.

## What it does
1. **Detects drift early.** Z-score drift against each machine's own baseline flags a developing fault days before a fixed alarm.
2. **Explains it.** An AMD-hosted LLM, grounded by retrieval over the pump manual and maintenance log, returns a structured diagnosis: cause, confidence, urgency, action, safety note, sources.
3. **Shows its uncertainty.** Low confidence, high urgency or malformed output triggers a visible "human review recommended" banner.
4. **Answers technician questions** (chat tab), grounded in the same manual and log with cited sources.
5. **Quantifies the result.** Headline: hours of earlier warning vs a fixed-threshold alarm × configurable $/hour = downtime cost avoided.

On the bundled demo data, drift detection flags the M3-Pump bearing fault **39 hours** before a fixed 4.5 mm/s vibration alarm fires.

## Architecture
```
sensors.csv ──► detect.py (z-score drift, fixed-alarm baseline) ─┐
manual + log ─► retrieval.py (TF-IDF chunks, top-k) ─────────────┼─► copilot.py ─► amd_client.py ─► AMD Token Factory
                                                                 │      (JSON prompt,   (OpenAI-compatible,
                                                                 │       validation)     latency + token tracking)
                                                                 └─► app.py (Streamlit cards, chat, AMD panel)
```

## How AMD is used
All model inference (diagnosis JSON and technician chat) calls the AMD Token Factory chat-completions endpoint through `amd_client.py`. The UI shows a "Powered by AMD" panel with the endpoint, model, per-call latency and token usage. No other LLM provider is used.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill AMD_API_KEY, AMD_BASE_URL, AMD_MODEL
python generate_data.py     # (data/ is also committed)
python check_amd.py         # one real request to verify endpoint, model and key
streamlit run app.py
```
Docker: `docker build -t maintenance-copilot . && docker run -p 8501:8501 --env-file .env maintenance-copilot`

Tests (no network needed): `python tests/test_logic.py`

Secrets are read from `st.secrets` first, then environment / `.env`. `.env` is git-ignored.

## Demo steps
1. Open the app: headline shows downtime cost avoided and hours earlier vs the fixed alarm.
2. *Fleet & alerts*: only M3-Pump is flagged; vibration crosses the drift line long before the fixed alarm.
3. *AI diagnosis* → **Diagnose with AMD**: cards show cause, action, safety note, sources, confidence and the human-review banner.
4. *Ask about this machine*: "How do I replace the drive-end bearing safely?"
5. Sidebar: AMD panel with model, latency and tokens. Change $/hour to see the headline update.

## Limitations
- Synthetic data; the demo is a proof of concept, not validated on a real plant.
- Downtime avoided is an estimate that assumes the early warning turns an unplanned stop into planned maintenance.
- TF-IDF retrieval is keyword-based; embeddings would improve recall on large document sets.
- The LLM can be wrong; the app recommends human review for low-confidence or high-urgency cases and never actuates equipment.
- Photo/defect-image analysis is not included (CSV upload and chat are the second use case).
