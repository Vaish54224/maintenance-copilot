"""Maintenance Copilot: early-warning + grounded diagnosis for factory machines, powered by AMD."""
import pandas as pd
import streamlit as st

import amd_client
import config
import copilot
import detect
import retrieval
import ui
from ui import esc, md

st.set_page_config(page_title="Maintenance Copilot", page_icon="🛠️", layout="wide", initial_sidebar_state="expanded")
ui.inject_css()

REQUIRED_COLS = {"timestamp", "machine", *detect.METRICS}
SUGGESTIONS = [
    "How do I replace the drive-end bearing safely?",
    "What vibration level is dangerous?",
    "When was M3-Pump last serviced?",
]
DEMO_MACHINE = "M3-Pump"
AVATARS = {"user": ":material/engineering:", "assistant": ":material/smart_toy:"}
ss = st.session_state


def money(x: float) -> str:
    return f"&#36;{x:,.0f}"  # HTML entity: a literal "$" pair would be rendered as LaTeX by st.markdown


@st.cache_data
def load_default() -> pd.DataFrame:
    return detect.load_sensors()


@st.cache_resource
def get_index() -> retrieval.Index:
    return retrieval.Index(retrieval.load_chunks())


def record_call(call: dict) -> None:
    ss.setdefault("calls", []).append(call)
    ss["last_error"] = None


def key_status(key: str) -> str:
    """Describe the configured key without revealing it (length + shape only)."""
    if not key:
        return "missing"
    if any(t in key.lower() for t in ("<", ">", "your", "paste", "key-here")):
        return "placeholder, not a real key"
    return f"set · {len(key)} chars"


def show_model_error(e: Exception) -> None:
    ss["last_error"] = type(e).__name__
    if isinstance(e, amd_client.AMDRateLimited):
        st.warning(f"⏳ {e}")
    elif isinstance(e, amd_client.AMDError) and "HTTP 401" in str(e):
        st.error(
            f"AMD endpoint rejected the API key (HTTP 401). Key in use: {key_status(config.amd_settings()['api_key'])}. "
            "On Streamlit Cloud open **Manage app → Settings → Secrets**, set `AMD_API_KEY = \"...\"` to your real "
            "Token Factory key (in quotes, no spaces), save, then **Reboot app**."
        )
    elif isinstance(e, amd_client.AMDError):
        st.error(f"AMD endpoint error: {e}")
    else:
        st.error(f"Unexpected error while calling the model ({type(e).__name__}). Please try again.")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    md('<div class="mc-side-h">Business</div>')
    cost_per_hour = st.number_input("Downtime cost (USD per hour)", 100, 1_000_000, 5000, step=500)
    demo = st.toggle("Demo mode", value=False, help="Pre-selects M3-Pump and offers a clearly labelled cached sample if a live AMD call fails.")
    st.divider()
    md('<div class="mc-side-h">Data</div>')
    up = st.file_uploader("Upload sensor CSV (optional)", type="csv")
    md(
        '<div class="mc-hint">Hourly rows with columns '
        + " ".join(f"<code>{c}</code>" for c in ["timestamp", "machine", *detect.METRICS])
        + "</div>"
    )
    st.divider()
    md('<div class="mc-side-h">AI engine</div>')
    amd_panel = st.container()  # filled at the end so it reflects calls made this run


# ---------------------------------------------------------------- data
def read_upload(f) -> pd.DataFrame | None:
    """Parse and validate an uploaded sensor CSV; returns None (with a message) if unusable."""
    try:
        df = pd.read_csv(f)
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(f"Could not read CSV ({type(e).__name__}). Showing demo data.")
        return None
    df.columns = [c.strip() for c in df.columns]
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        st.sidebar.error(f"CSV is missing columns: {', '.join(sorted(missing))}. Showing demo data.")
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for m in detect.METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    df["machine"] = df["machine"].astype(str)
    df = df.dropna(subset=["timestamp", *detect.METRICS]).sort_values(["machine", "timestamp"])
    if df.empty:
        st.sidebar.error("CSV has no valid rows. Showing demo data.")
        return None
    short = df.groupby("machine").size()
    short = sorted(short[short <= detect.BASELINE_HOURS].index)
    if short:
        st.sidebar.warning(
            f"{', '.join(short)}: fewer than {detect.BASELINE_HOURS} hourly rows, so drift cannot be flagged yet."
        )
    st.sidebar.success(f"Using your CSV: {len(df):,} rows, {df.machine.nunique()} machines")
    return df


raw = read_upload(up) if up is not None else None
if raw is None:
    raw = load_default()
try:
    dfz = detect.add_zscores(raw)
    machines = sorted(dfz.machine.unique())
    statuses = {m: detect.machine_status(dfz, m) for m in machines}
except Exception as e:  # noqa: BLE001
    st.sidebar.error(f"Could not analyse the uploaded data ({type(e).__name__}). Showing demo data.")
    dfz = detect.add_zscores(load_default())
    machines = sorted(dfz.machine.unique())
    statuses = {m: detect.machine_status(dfz, m) for m in machines}
flagged = [m for m in machines if statuses[m]["drifted"]]
target = flagged[0] if flagged else machines[0]
cmp_ = detect.baseline_vs_ai(dfz, target) if flagged else None
has_win = bool(cmp_ and cmp_["ai_flag_time"] is not None)

# active machine shared by all sections
if demo and DEMO_MACHINE in machines and not ss.get("demo_applied"):
    ss["active"] = DEMO_MACHINE
    ss["demo_applied"] = True
if not demo:
    ss["demo_applied"] = False
if ss.get("active") not in machines:
    ss["active"] = target
active = ss["active"]


def set_active(m: str) -> None:
    ss["active"] = m


# ---------------------------------------------------------------- hero + KPIs
def amd_state() -> tuple[str, str]:
    if not config.is_configured():
        return "bad", "Not configured"
    if ss.get("last_error"):
        return "warn", "Degraded"
    return "ok", "Connected"


def render_hero() -> None:
    s = config.amd_settings()
    dot, label = amd_state()
    model = esc(s["model"] or "no model set")
    md(
        f"""<div class="mc-hero">
          <div class="mc-brand"><div class="mc-logo">{ui.LOGO}</div>
            <div><div class="mc-title">Maintenance Copilot</div>
            <div class="mc-sub">Catch machine failures hours earlier, with cited fixes</div></div></div>
          <div class="mc-amd" title="{label}"><span class="mc-dot {dot}"></span>
            Powered by <span class="amd">AMD</span> <b>{model}</b></div>
        </div>"""
    )


def render_kpis() -> None:
    calls = ss.get("calls", [])
    if calls:
        lat_val, lat_hint = f"{calls[-1]['latency_s']:.1f} s", f"last AMD call · {calls[-1]['total_tokens']:,} tokens"
    else:
        lat_val, lat_hint = "—", "no model calls yet"
    if has_win:
        saved = cmp_["hours_earlier"] * cost_per_hour
        k1 = (money(saved), f"at {money(cost_per_hour)}/h · {esc(target)}")
        k2 = (f"+{cmp_['hours_earlier']:g} h", f"vs fixed {detect.FIXED_VIB_ALARM} mm/s vibration alarm")
        fixed = "has not fired yet" if cmp_["fixed_never_fired"] else f"only fired {cmp_['fixed_alarm_time']:%b %d %H:%M}"
        explain = (
            f"Drift detection flagged <b>{esc(target)}</b> on <b>{cmp_['ai_flag_time']:%b %d %H:%M}</b>; the fixed alarm {fixed}. "
            f"<b>{cmp_['hours_earlier']:g} hours</b> of extra planning time × {money(cost_per_hour)}/h = <b>{money(saved)}</b> "
            "of unplanned downtime avoidable. <i>Estimate: assumes the early warning turns an unplanned stop into planned maintenance.</i>"
        )
    else:
        k1, k2 = (money(0), "no developing faults"), ("—", "nothing to warn about")
        explain = "All machines are within their normal baseline. Upload your own CSV in the sidebar to analyse another line."
    md(
        f"""<div class="mc-kpis">
          <div class="mc-kpi hero"><div class="lbl">Downtime cost avoided (est.)</div><div class="val">{k1[0]}</div><div class="hint">{k1[1]}</div></div>
          <div class="mc-kpi"><div class="lbl">Earlier than fixed alarm</div><div class="val teal">{k2[0]}</div><div class="hint">{k2[1]}</div></div>
          <div class="mc-kpi"><div class="lbl">Machines flagged</div><div class="val {'red' if flagged else ''}">{len(flagged)} <span style="color:#8aa0b8;font-size:1.1rem;font-weight:600">/ {len(machines)}</span></div>
            <div class="hint">{esc(", ".join(flagged)) if flagged else "fleet healthy"}</div></div>
          <div class="mc-kpi"><div class="lbl">AI latency</div><div class="val">{lat_val}</div><div class="hint">{lat_hint}</div></div>
        </div><div class="mc-explain">{explain}</div>"""
    )


# ---------------------------------------------------------------- fleet cards
def render_fleet() -> None:
    md(f'<div class="mc-sec"><h4>Fleet overview</h4><span>Select a machine · active: <b style="color:#e6edf5">{esc(active)}</b></span></div>')
    per_row = 5
    for r in range(0, len(machines), per_row):
        row = machines[r : r + per_row]
        cols = st.columns(per_row, gap="small")
        for i, (col, m) in enumerate(zip(cols, row)):
            s = statuses[m]
            drift = bool(s["drifted"])
            cls = "mc-card" + (" drift" if drift else "") + (" active" if m == active else "")
            status_chip = ui.chip("Drift detected", "bad") if drift else ui.chip("Normal", "ok")
            spark = ui.sparkline(s["frame"]["vibration_mm_s"], ui.RED if drift else ui.TEAL)
            with col:
                with st.container(key=f"fc_{r + i}"):
                    md(
                        f'<div class="{cls}"><div class="top"><span class="name">{esc(m)}</span>{status_chip}</div>'
                        f'<div class="sig">{esc(ui.signals_text(s["drifted"]))}</div>{spark}</div>'
                    )
                    st.button(f"Select {m}", key=f"fcb_{r + i}", on_click=set_active, args=(m,))


# ---------------------------------------------------------------- trends
def render_trends() -> None:
    s = statuses[active]
    c = detect.baseline_vs_ai(dfz, active) if s["drifted"] else None
    chips = ui.chip("Drift detected", "bad") if s["drifted"] else ui.chip("Within baseline", "ok")
    md(f'<div class="mc-sec"><h4>Vibration · {esc(active)}</h4><span>{chips}</span></div>')
    if s["frame"].empty or len(s["frame"]) < 2:
        ui.empty("Not enough data to plot", "This machine needs at least two readings.")
        return
    st.altair_chart(ui.trend_chart(s["frame"], c), width="stretch", theme=None)
    a, b = st.columns(2)
    with a, st.expander("Temperature and motor current"):
        st.altair_chart(ui.signals_chart(s["frame"]), width="stretch", theme=None)
    with b, st.expander("Fleet detail table"):
        rows = []
        for m in machines:
            d = statuses[m]["drifted"]
            rows.append(
                {
                    "Machine": m,
                    "Status": "Drift" if d else "Normal",
                    "Drifting signals": ui.signals_text(d) if d else "-",
                    "First flagged": min((v["first_flag"] for v in d.values()), default=None),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


# ---------------------------------------------------------------- diagnosis
def sources_html(hits: list[dict], used: list[str]) -> str:
    out = []
    for i, h in enumerate(hits, 1):
        u = h["source"] in used
        out.append(
            f'<details class="{"used" if u else ""}"><summary>{"✓ " if u else ""}[{i}] {esc(h["source"])}</summary>'
            f'<div class="ex">{esc(h["text"])}</div></details>'
        )
    return f'<div class="mc-sources">{"".join(out)}</div>'


def run_diagnosis(pick: str) -> None:
    model = config.amd_settings()["model"]
    with st.status("Retrieving manual and maintenance-log context…", expanded=True) as box:
        try:
            idx = get_index()
            st.write("✓ Context index ready (manual sections + work orders)")
            box.update(label=f"Calling AMD model {model}…")
            res = copilot.analyze(statuses[pick], idx)
            st.write(f"✓ AMD model answered in {res['call']['latency_s']} s")
            box.update(label="Validating answer…")
            st.write("✓ JSON validated" if res["parse_ok"] else "⚠ Answer could not be parsed: flagged for human review")
            record_call(res["call"])
            ss[f"res_{pick}"] = res
            ss.pop("diag_failed", None)
            box.update(label=f"Diagnosis ready · {res['call']['latency_s']} s", state="complete", expanded=False)
            st.rerun()  # refresh KPI strip, hero badge and AMD panel with this call
        except Exception as e:  # noqa: BLE001
            box.update(label="AMD call did not complete", state="error", expanded=False)
            ss["diag_failed"] = pick
            show_model_error(e)


def render_result(res: dict) -> None:
    if res.get("cached"):
        md(
            '<div class="mc-cached">CACHED SAMPLE RESPONSE · NOT LIVE — a Gemma-4-31B-IT answer recorded earlier on the '
            "demo data, shown because the live AMD call was unavailable.</div>"
        )
    if res["human_review"]:
        why = (
            "the model output could not be validated"
            if not res["parse_ok"]
            else ("model confidence is below 60%" if res["confidence"] < copilot.CONF_REVIEW_THRESHOLD else "urgency is high")
        )
        md(
            f'<div class="mc-review"><span>👷</span><div><b>Human review recommended</b> because {why}. '
            "Have a qualified technician confirm before acting.</div></div>"
        )
    urg = res["urgency"]
    conf = res["confidence"]
    call = res["call"]
    md(
        f"""<div class="mc-diag-head">
          <div class="cell"><div class="lbl">Urgency</div><span class="mc-urg {urg}">{esc(urg)}</span></div>
          <div class="cell"><div class="lbl">Model confidence</div><div class="mc-big">{conf:.0%}</div>
            <div class="mc-meter {'low' if conf < copilot.CONF_REVIEW_THRESHOLD else ''}"><i style="width:{conf * 100:.0f}%"></i></div></div>
          <div class="cell"><div class="lbl">Sources cited</div><div class="mc-big">{len(res['sources'])}</div></div>
        </div>"""
    )
    steps = ui.action_steps(res["action"])
    action_html = (
        "<ol>" + "".join(f"<li>{esc(s)}</li>" for s in steps) + "</ol>" if steps else f"<p>{esc(res['action'])}</p>"
    )
    left, right = st.columns([1.15, 1], gap="medium")
    with left:
        unc = (
            f'<h5 style="margin-top:16px">Uncertainty</h5><p style="color:#8aa0b8">{esc(res["uncertainty_note"])}</p>'
            if res["uncertainty_note"]
            else ""
        )
        md(
            f'<div class="mc-panel"><h5>Likely cause</h5><p style="font-size:1.15rem;font-weight:600">{esc(res["cause"])}</p>'
            f'<h5 style="margin-top:16px">Recommended action</h5>{action_html}{unc}</div>'
        )
    with right:
        md(
            f'<div class="mc-safety mc-panel"><h5>⚠ Safety</h5><p>{esc(res["safety_note"] or "Follow the lock-out/tag-out procedure before any work.")}</p></div>'
        )
        md('<div class="mc-panel" style="margin-top:12px"><h5>Sources</h5>' + sources_html(res["hits"], res["sources"]) + "</div>")
    origin = "cached sample · no AMD call made" if res.get("cached") else "live AMD call"
    md(
        f'<div class="mc-foot"><span>Model <b>{esc(call.get("model", ""))}</b></span><span>Latency <b>{call["latency_s"]} s</b></span>'
        f'<span>Tokens <b>{call["total_tokens"]:,}</b></span><span>{origin}</span></div>'
    )
    with st.expander("Sensor evidence sent to the model"):
        st.code(res["evidence"])
    if not res["parse_ok"] and res.get("raw"):
        with st.expander("Raw model output"):
            st.text(res["raw"])


def cached_sample() -> dict:
    by_src = {c["source"]: c for c in get_index().chunks}
    hits = [{**by_src[s], "score": 0} for s in ui.SAMPLE_DIAGNOSIS["sources"] if s in by_src]
    return {**ui.SAMPLE_DIAGNOSIS, "hits": hits}


def render_diagnosis() -> None:
    pick = active
    s = statuses[pick]
    configured = config.is_configured()
    a, b = st.columns([3, 1], vertical_alignment="center")
    with a:
        sub = ui.signals_text(s["drifted"]) if s["drifted"] else "no drift detected: a diagnosis is optional"
        md(
            f'<div class="mc-sec" style="margin:0"><h4>AI diagnosis · {esc(pick)}</h4></div>'
            f'<div style="color:#8aa0b8;font-size:.9rem">Sends the sensor evidence plus the most relevant manual sections and '
            f"past work orders to the AMD-hosted model. Signals: {esc(sub)}.</div>"
        )
    with b:
        clicked = st.button("Diagnose with AMD", type="primary", disabled=not configured, width="stretch")
    if not configured:
        st.info("The AMD endpoint is not configured, so live diagnosis is off. Set AMD_API_KEY, AMD_BASE_URL and AMD_MODEL in .env or Streamlit secrets.")
    if clicked:
        run_diagnosis(pick)
    res = ss.get(f"res_{pick}")
    can_cache = demo and pick == DEMO_MACHINE and (ss.get("diag_failed") == pick or not configured)
    if can_cache and not res:
        if st.button("Show cached sample response (not live)", key="cached_btn"):
            ss[f"res_{pick}"] = cached_sample()
            st.rerun()
    if res:
        render_result(res)
    elif not clicked:
        ui.empty("No diagnosis yet", " Click Diagnose with AMD to get a cited cause, action and safety note in seconds.")


# ---------------------------------------------------------------- chat
def render_chat() -> None:
    cm = active
    configured = config.is_configured()
    md(f'<div class="mc-sec"><h4>Technician chat · {esc(cm)}</h4><span>Answers come only from the manual and maintenance log</span></div>')
    history = ss.setdefault("chats", {}).setdefault(cm, [])
    with st.container(key="suggest"):
        cols = st.columns(len(SUGGESTIONS))
        for col, q in zip(cols, SUGGESTIONS):
            if col.button(q, disabled=not configured, key=f"sg_{q}"):
                ss["pending_q"] = q
    if not configured:
        st.info("Chat needs the AMD endpoint. Set AMD_API_KEY, AMD_BASE_URL and AMD_MODEL to enable it.")
    if not history and not ss.get("pending_q"):
        ui.empty("Ask anything about this machine", " Try one of the suggested questions above.")
    for m in history:
        with st.chat_message(m["role"], avatar=AVATARS[m["role"]]):
            st.markdown(m["content"])
            if m.get("hits"):
                md(sources_html(m["hits"], m["sources"]))
    typed = st.chat_input("Ask about procedures, limits or past work orders…", disabled=not configured)
    q = typed or ss.pop("pending_q", None)
    if q:
        history.append({"role": "user", "content": q})
        with st.chat_message("user", avatar=AVATARS["user"]):
            st.markdown(q)
        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            try:
                with st.spinner("Asking the AMD model…"):
                    out = copilot.ask(statuses[cm], q, get_index())
                record_call(out["call"])
                st.markdown(out["answer"])
                md(sources_html(out["hits"], out["sources"]))
                history.append({"role": "assistant", "content": out["answer"], "sources": out["sources"], "hits": out["hits"]})
                st.rerun()  # refresh KPI strip and AMD panel
            except Exception as e:  # noqa: BLE001
                history.pop()  # keep history clean so the question can simply be asked again
                show_model_error(e)


# ---------------------------------------------------------------- AMD panel
def render_amd_panel() -> None:
    s = config.amd_settings()
    dot, label = amd_state()
    calls = ss.get("calls", [])
    avg = f"{sum(c['latency_s'] for c in calls) / len(calls):.2f} s" if calls else "—"
    total = f"{sum(c['total_tokens'] for c in calls):,}" if calls else "0"
    last = f"{calls[-1]['latency_s']} s · {calls[-1]['total_tokens']:,} tok" if calls else "—"
    rows = [
        ("Endpoint", esc(ui.host(s["base_url"]))),
        ("Model", esc(s["model"] or "not set")),
        ("API key", key_status(s["api_key"])),
        ("Calls", str(len(calls))),
        ("Last call", last),
        ("Avg latency", avg),
        ("Total tokens", total),
    ]
    with amd_panel:
        md(
            f'<div class="mc-amdcard"><div class="head"><span class="mc-dot {dot}"></span>AMD Token Factory · {label}</div>'
            + "".join(f"<div class='row'><span>{k}</span><span>{v}</span></div>" for k, v in rows)
            + "</div>"
        )


# ---------------------------------------------------------------- page
def safe(fn, name: str) -> None:
    """Render a section; a failure shows a friendly message instead of a stack trace and never takes down the page."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001  (st.rerun/st.stop are BaseException and pass through)
        ui.empty(f"The {name} section could not be displayed", f" ({type(e).__name__}). Reload the page to try again.")


safe(render_hero, "header")
safe(render_kpis, "headline")
safe(render_fleet, "fleet")
t1, t2, t3 = st.tabs(["Trends", "AI diagnosis", "Technician chat"])
with t1:
    safe(render_trends, "trend")
with t2:
    safe(render_diagnosis, "diagnosis")
with t3:
    safe(render_chat, "chat")
safe(render_amd_panel, "AMD status")
