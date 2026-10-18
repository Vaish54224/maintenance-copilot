# Maintenance Copilot: Catch Failures Early

## Problem
Factories already collect sensor data, maintenance reports and manuals. The hard part is knowing what matters and what to do next. Fixed-threshold alarms fire late, and technicians lose time searching manuals and past work orders while a machine is already degrading. Unplanned downtime often costs thousands of dollars per hour.

## Solution
Maintenance Copilot is a working tool for maintenance engineers and shift supervisors.
- **Early drift detection:** each machine is compared with its own baseline (z-scores, sustained for several hours). On the demo data the M3-Pump bearing fault is flagged **39 hours before** a fixed 4.5 mm/s vibration alarm.
- **Grounded diagnosis:** a retrieval step pulls the most relevant manual sections and past maintenance-log entries, and an AMD-hosted LLM returns a structured result: cause, confidence, urgency, action, safety note and cited sources, rendered as clear cards.
- **Honest uncertainty:** low confidence, high urgency or malformed model output triggers a visible "human review recommended" banner.
- **Technician chat:** ask "how do I replace the drive-end bearing safely?" and get a cited answer from the manual and log.
- **Business number up front:** hours of early warning × configurable cost per hour = estimated downtime cost avoided (about $195,000 at $5,000/h in the demo).

## How AMD is used
Every model call (diagnosis and chat) goes to the AMD Token Factory chat-completions endpoint, running **Gemma-4-31B-IT**. Measured in the app: a full diagnosis takes about 6–9 s and ~1,000 tokens, a technician chat answer about 7 s and ~870 tokens. The app displays a "Powered by AMD" panel with the endpoint, model, latency per call and token usage, so judges can see the integration live. The structured-JSON prompt, validation and fallback logic are built around the model's output, so inference is central to the product rather than decoration.

## Impact
- Earlier, planned maintenance instead of emergency stops.
- Faster, safer repairs: every recommendation carries the relevant safety steps (lock-out/tag-out, cool-down) from the manual.
- Less tribal knowledge: new technicians get cited answers from the plant's own documents.

## Tech stack
Python, Streamlit, pandas/numpy, a TF-IDF retriever written from scratch, AMD Token Factory (OpenAI-compatible API) with Gemma-4-31B-IT, Docker, Streamlit Community Cloud.

## What's next
Live sensor streams (MQTT/OPC-UA), embedding-based retrieval, vision models on AMD for defect photos, work-order creation in a CMMS, and validation on real plant data.
