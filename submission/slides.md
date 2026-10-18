# Slides outline (7)

## 1. Problem
**Factories have data, not decisions**
- Fixed-threshold alarms fire when a machine is already close to failure
- Technicians search manuals and old work orders by hand
- Unplanned downtime costs thousands of dollars per hour

## 2. User
**Maintenance engineers and shift supervisors**
- Need: know early, know why, know what to do safely
- Today: alarms, spreadsheets, PDF manuals, tribal knowledge

## 3. Solution
**Maintenance Copilot**
- Drift detection against each machine's own baseline
- Grounded diagnosis: cause, confidence, urgency, action, safety note, sources
- Technician chat over the manual and maintenance log
- Human-review flag when the model is unsure or urgency is high

## 4. Live demo
**M3-Pump bearing fault** (screenshot: `docs/home.jpg`)
- Fleet cards: M3-Pump flagged red; chart shows "AI flags drift" 39 h before "Fixed alarm fires"
- One click: AMD model returns a cited diagnosis with urgency, confidence and a safety callout (`docs/diagnosis.jpg`)
- Ask: "How do I replace the drive-end bearing safely?" with source chips (`docs/chat.jpg`)

## 5. How AMD powers it
**All inference runs on AMD Token Factory: Gemma-4-31B-IT**
- OpenAI-compatible chat completions, structured JSON output
- Measured: diagnosis ≈ 6–9 s / ~1,000 tokens; chat answer ≈ 7 s / ~870 tokens
- Retrieval-grounded prompts (manual + log, TF-IDF top-k)
- Live panel: endpoint, model, latency per call, tokens used
- Validation + graceful fallback for malformed output

## 6. Measurable impact
**39 hours earlier warning than a fixed vibration alarm** (demo data)
- ≈ $195,000 of unplanned downtime avoidable at $5,000/h (configurable)
- Cited, safety-aware actions in seconds instead of manual search

## 7. Next steps
- Live sensor streams (MQTT / OPC-UA)
- Embedding retrieval and vision models for defect photos on AMD
- CMMS work-order creation
- Pilot on real plant data
