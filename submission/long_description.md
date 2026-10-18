# Maintenance Copilot: Catch Failures Early

## Problem
Factories already collect sensor data, maintenance reports and manuals. The hard part is knowing what matters and what to do next. Fixed-threshold alarms fire late, when a machine is already close to failure, and technicians lose time searching manuals and past work orders while it keeps degrading. Unplanned downtime commonly costs thousands of dollars per hour.

## Solution
Maintenance Copilot is a working tool for maintenance engineers and shift supervisors. It runs as a web app, with one screen for the whole fleet and three steps: detect, diagnose, ask.
- **Early drift detection:** each machine is compared with its own 7-day baseline (z-scores, sustained for 6 hours to ignore noise). On the demo plant, the M3-Pump bearing fault is flagged **39 hours before** a fixed 4.5 mm/s vibration alarm fires. The chart marks both moments and the gap between them.
- **Grounded AI diagnosis:** one click retrieves the most relevant manual sections and past work orders. An AMD-hosted LLM returns a structured diagnosis: likely cause, urgency, confidence, numbered actions, a safety callout and the exact sources it used.
- **Honest uncertainty:** low confidence, high urgency or output that fails validation triggers a "human review recommended" banner. The copilot advises; people decide.
- **Technician chat:** ask "How do I replace the drive-end bearing safely?" and get an answer drawn only from the plant's own documents, with the cited sources shown.
- **Business number up front:** hours of early warning × configurable cost per hour = estimated downtime cost avoided (about $195,000 at $5,000/h on the demo data).

## Application of technology: how AMD is used
- All inference runs on **AMD Token Factory** with **Gemma-4-31B-IT**, through its OpenAI-compatible chat-completions API. No other LLM provider is used.
- The model is central to the product, not decoration. It turns sensor statistics plus retrieved documents into a structured JSON decision. The app validates every field, maps cited reference numbers back to real documents, and retries once with a repair prompt if the JSON is malformed. If the output still can't be validated, the app shows a safe human-review card instead of guessing.
- Retrieval always includes the manual's safety section, so every recommendation carries real precautions (lock-out/tag-out, PPE, stop limits).
- Built for a shared endpoint: backoff on HTTP 429 (Retry-After), friendly error states, and a live "Powered by AMD" panel showing endpoint, model, per-call latency and token usage. Measured: a diagnosis takes about 6–9 s and ~1,000 tokens; a chat answer about 7 s and ~870 tokens.

## Business value
- Converts unplanned stops into planned maintenance: 39 hours of extra planning time on the demo case.
- Faster, safer repairs: cited, safety-aware actions in seconds instead of manual searching.
- Less dependence on tribal knowledge: new technicians get answers from the plant's own manual and work orders.
- Works on any plant's data: upload a sensor CSV (validated, with clear errors) and the fleet view, alerts and diagnosis adapt.

## Originality
- Puts a measurable comparison of AI drift detection against the classic fixed alarm on screen, and translates it straight into dollars.
- Combines statistical early warning with a grounded LLM decision, instead of a free-form chatbot, and shows its uncertainty, sources and costs openly.
- Demo-safe: an optional Demo mode can show a cached sample answer if the live endpoint is unavailable, and it is always labelled "cached sample, not live".

## Tech stack
Python, Streamlit, Altair, pandas/numpy, a from-scratch TF-IDF retriever, AMD Token Factory (Gemma-4-31B-IT, OpenAI-compatible API), Docker, Streamlit Community Cloud.

## What's next
Live sensor streams (MQTT/OPC-UA), embedding-based retrieval, vision models on AMD for defect photos, work-order creation in a CMMS, and a pilot on real plant data.
