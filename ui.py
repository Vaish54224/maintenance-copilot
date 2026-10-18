"""Presentation helpers for app.py: CSS, small HTML components, sparkline SVG and the trend chart.

No business logic lives here; everything is computed by detect/copilot and only rendered.
"""
import html
import re
from pathlib import Path
from urllib.parse import urlparse

import altair as alt
import pandas as pd
import streamlit as st

import detect

TEAL, RED, AMBER, GREEN, MUTED, TEXT = "#35e0c2", "#ff4b4b", "#f5b13d", "#3ddc84", "#8aa0b8", "#e6edf5"
SIGNAL_NAMES = {
    "vibration_mm_s": "Vibration",
    "temperature_c": "Temperature",
    "motor_current_a": "Motor current",
    "pressure_bar": "Pressure",
}
LOGO = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#ff4b4b" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l3-8 4 16 3-8h4"/></svg>'
)

esc = html.escape


def inject_css() -> None:
    css = (Path(__file__).parent / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def md(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def host(url: str) -> str:
    return urlparse(url).netloc or url or "not set"


def chip(text: str, kind: str = "") -> str:
    return f'<span class="mc-chip {kind}">{esc(text)}</span>'


def signals_text(drifted: dict) -> str:
    return ", ".join(SIGNAL_NAMES.get(m, m) for m in drifted) if drifted else "All signals within baseline"


def sparkline(values: pd.Series, color: str, width: int = 220, height: int = 38) -> str:
    v = pd.Series(values).dropna().reset_index(drop=True)
    if len(v) < 2:
        return f'<svg width="100%" height="{height}"></svg>'
    step = max(1, len(v) // 80)  # downsample to ~80 points
    v = v.iloc[::step].reset_index(drop=True)
    lo, hi = float(v.min()), float(v.max())
    span = (hi - lo) or 1.0
    pts = " ".join(
        f"{i / (len(v) - 1) * width:.1f},{height - 3 - (float(y) - lo) / span * (height - 6):.1f}" for i, y in enumerate(v)
    )
    return (
        f'<svg width="100%" height="{height}" viewBox="0 0 {width} {height}" preserveAspectRatio="none">'
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6" vector-effect="non-scaling-stroke" '
        'stroke-linejoin="round"/></svg>'
    )


def empty(title: str, body: str = "") -> None:
    md(f'<div class="mc-empty"><b>{esc(title)}</b>{esc(body)}</div>')


def action_steps(text: str) -> list[str]:
    """Split a recommended-action paragraph into short steps when it naturally has several."""
    t = text.strip()
    parts = [p.strip(" .;") for p in re.split(r"(?:^|\s)\d+[.)]\s+|\n+|;\s+|\.\s+(?=[A-Z])", t) if p and p.strip(" .;")]
    if len(parts) < 2:
        parts = [p.strip(" .") for p in re.split(r",\s+and\s+|\s+and then\s+|\s+then\s+", t) if p.strip(" .")]
    return [p[0].upper() + p[1:] for p in parts] if len(parts) >= 2 else []


# ---------------------------------------------------------------- trend chart
def trend_chart(frame: pd.DataFrame, cmp_: dict | None, height: int = 290) -> alt.LayerChart:
    g = frame[["timestamp", "vibration_mm_s"]].copy()
    t0 = g["timestamp"].iloc[0]
    base_end = g["timestamp"].iloc[min(detect.BASELINE_HOURS, len(g)) - 1]
    ymax = float(max(g["vibration_mm_s"].max() * 1.12, detect.FIXED_VIB_ALARM * 1.35))
    x = alt.X(
        "timestamp:T",
        title=None,
        axis=alt.Axis(format="%b %d", labelColor=MUTED, grid=False, tickCount={"interval": "day", "step": 1}, domainColor="#1f2937", tickColor="#1f2937"),
    )
    y_scale = alt.Scale(domain=[0, ymax])

    band = (
        alt.Chart(pd.DataFrame({"timestamp": [t0], "e": [base_end]}))
        .mark_rect(color=TEAL, opacity=0.06)
        .encode(x=x, x2="e:T")
    )
    band_lbl = (
        alt.Chart(pd.DataFrame({"timestamp": [t0], "y": [ymax * 0.96], "txt": ["7-day baseline"]}))
        .mark_text(align="left", dx=8, color=TEAL, fontSize=11, opacity=0.8)
        .encode(x=x, y=alt.Y("y:Q", scale=y_scale), text="txt:N")
    )
    line = (
        alt.Chart(g)
        .mark_line(color="#c9d6e3", strokeWidth=1.5)
        .encode(
            x=x,
            y=alt.Y(
                "vibration_mm_s:Q",
                title="Vibration (mm/s)",
                scale=y_scale,
                axis=alt.Axis(labelColor=MUTED, titleColor=MUTED, grid=True, gridColor="#151c2a", domain=False, tickCount=5),
            ),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Time", format="%b %d %H:%M"),
                alt.Tooltip("vibration_mm_s:Q", title="Vibration (mm/s)", format=".2f"),
            ],
        )
    )
    alarm = (
        alt.Chart(pd.DataFrame({"y": [detect.FIXED_VIB_ALARM]}))
        .mark_rule(color=RED, strokeDash=[6, 4], opacity=0.7)
        .encode(y=alt.Y("y:Q", scale=y_scale))
    )
    alarm_lbl = (
        alt.Chart(pd.DataFrame({"timestamp": [g["timestamp"].iloc[-1]], "y": [detect.FIXED_VIB_ALARM], "txt": [f"Fixed alarm {detect.FIXED_VIB_ALARM} mm/s"]}))
        .mark_text(align="right", dx=-4, dy=10, color=RED, fontSize=11)
        .encode(x=x, y=alt.Y("y:Q", scale=y_scale), text="txt:N")
    )
    layers = [line, band, band_lbl, alarm, alarm_lbl]  # all layers share one x field/axis

    if cmp_ and cmp_.get("ai_flag_time") is not None:
        ai_t = cmp_["ai_flag_time"]
        fx_t = cmp_.get("fixed_alarm_time")
        marks = [{"timestamp": ai_t, "c": TEAL, "txt": "AI flags drift", "align": "right", "dx": -6}]
        if fx_t is not None:
            marks.append({"timestamp": fx_t, "c": RED, "txt": "Fixed alarm fires", "align": "left", "dx": 6})
        m = pd.DataFrame(marks)
        m["y"] = ymax * 0.80
        layers.append(alt.Chart(m).mark_rule(strokeWidth=2).encode(x=x, color=alt.Color("c:N", scale=None)))
        for row in marks:
            layers.append(
                alt.Chart(pd.DataFrame([{**row, "y": ymax * 0.80}]))
                .mark_text(align=row["align"], dx=row["dx"], fontSize=12, fontWeight="bold", color=row["c"])
                .encode(x=x, y=alt.Y("y:Q", scale=y_scale), text="txt:N")
            )
        end_t = fx_t if fx_t is not None else g["timestamp"].iloc[-1]
        gap_txt = f"+{cmp_['hours_earlier']:g} h earlier" + ("" if fx_t is not None else " (fixed alarm never fired)")
        mid = ai_t + (end_t - ai_t) / 2
        layers.append(
            alt.Chart(pd.DataFrame({"timestamp": [ai_t], "e": [end_t], "y": [ymax * 0.90]}))
            .mark_rule(color=TEAL, strokeWidth=2, opacity=0.8)
            .encode(x=x, x2="e:T", y=alt.Y("y:Q", scale=y_scale))
        )
        layers.append(
            alt.Chart(pd.DataFrame({"timestamp": [mid], "y": [ymax * 0.90], "txt": [gap_txt]}))
            .mark_text(dy=-10, fontSize=13, fontWeight="bold", color=TEAL)
            .encode(x=x, y=alt.Y("y:Q", scale=y_scale), text="txt:N")
        )

    return (
        alt.layer(*layers)
        .properties(height=height, background="transparent", padding={"left": 4, "right": 12, "top": 8, "bottom": 4})
        .configure_view(strokeWidth=0)
        .configure_axis(labelFont="Inter", titleFont="Inter", labelFontSize=11, titleFontSize=11, titleFontWeight=500)
        .configure_text(font="Inter")
    )


def signals_chart(frame: pd.DataFrame, height: int = 200) -> alt.Chart:
    cols = [c for c in ("temperature_c", "motor_current_a") if c in frame]
    d = frame.melt(id_vars="timestamp", value_vars=cols, var_name="signal", value_name="value")
    d["signal"] = d["signal"].map(SIGNAL_NAMES)
    return (
        alt.Chart(d)
        .mark_line(strokeWidth=1.4)
        .encode(
            x=alt.X("timestamp:T", title=None, axis=alt.Axis(format="%b %d", labelColor=MUTED, grid=False, tickCount={"interval": "day", "step": 1})),
            y=alt.Y("value:Q", title=None, axis=alt.Axis(labelColor=MUTED, gridColor="#151c2a", domain=False)),
            color=alt.Color(
                "signal:N",
                scale=alt.Scale(range=[AMBER, TEAL]),
                legend=alt.Legend(orient="top", labelColor=MUTED, title=None),
            ),
            tooltip=[alt.Tooltip("timestamp:T", format="%b %d %H:%M"), "signal:N", alt.Tooltip("value:Q", format=".2f")],
        )
        .properties(height=height, background="transparent")
        .configure_view(strokeWidth=0)
    )


# ---------------------------------------------------------------- demo fallback
# A real Gemma-4-31B-IT response captured during testing on the bundled M3-Pump data.
# Only ever shown in Demo mode, after a live call fails, behind an explicit click, and labelled "not live".
SAMPLE_DIAGNOSIS = {
    "cause": "Bearing wear or lubrication breakdown",
    "confidence": 0.9,
    "urgency": "high",
    "action": "Inspect bearing grease condition and quantity, and perform a full bearing inspection within 72 hours.",
    "safety_note": "Never inspect or touch rotating equipment while running. Wear cut-resistant gloves and eye protection. "
    "Escalate to shift supervisor and stop the pump if vibration exceeds 7.1 mm/s.",
    "sources": ["Manual: 2. Bearing fault diagnosis", "Manual: 1. Normal operating ranges", "Manual: 5. Safety"],
    "uncertainty_note": "The signature strongly matches bearing wear, but lubrication breakdown vs. physical wear cannot be "
    "distinguished without inspection.",
    "human_review": True,
    "parse_ok": True,
    "raw": "",
    "evidence": "(cached sample: no sensor evidence was sent)",
    "call": {"model": "Gemma-4-31B-IT", "endpoint": "", "latency_s": 8.4, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 1147},
    "cached": True,
}
