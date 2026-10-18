"""Maintenance Copilot: early-warning + grounded diagnosis for factory machines, powered by AMD."""
import pandas as pd
import streamlit as st

import amd_client
import config
import copilot
import detect
import retrieval

st.set_page_config(page_title="Maintenance Copilot", page_icon="🛠️", layout="wide")

REQUIRED_COLS = {"timestamp", "machine", *detect.METRICS}
URG_ICON = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}


@st.cache_data
def load_default() -> pd.DataFrame:
    return detect.load_sensors()


@st.cache_resource
def get_index() -> retrieval.Index:
    return retrieval.Index(retrieval.load_chunks())


def record_call(call: dict) -> None:
    st.session_state.setdefault("calls", []).append(call)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    cost_per_hour = st.number_input("Cost of unplanned downtime ($/hour)", 100, 1_000_000, 5000, step=500)
    up = st.file_uploader("Upload your own sensor CSV (optional)", type="csv")
    st.caption("Columns: " + ", ".join(sorted(REQUIRED_COLS)))

    st.divider()
    amd_panel = st.container()  # filled at the end of the script so it reflects calls made this run


def render_amd_panel() -> None:
    with amd_panel:
        st.subheader("⚡ Powered by AMD")
        s = config.amd_settings()
        if config.is_configured():
            st.success("AMD Token Factory connected")
            st.write(f"**Endpoint:** `{s['base_url']}`")
            st.write(f"**Model:** `{s['model']}`")
        else:
            st.error("AMD endpoint not configured. Set AMD_API_KEY, AMD_BASE_URL, AMD_MODEL.")
        calls = st.session_state.get("calls", [])
        if calls:
            last = calls[-1]
            c1, c2 = st.columns(2)
            c1.metric("Last latency", f"{last['latency_s']} s")
            c2.metric("Last tokens", last["total_tokens"])
            st.caption(
                f"{len(calls)} calls · avg latency {sum(c['latency_s'] for c in calls) / len(calls):.2f} s · "
                f"{sum(c['total_tokens'] for c in calls)} tokens total"
            )
        else:
            st.caption("No model calls yet.")


# ---------------------------------------------------------------- data
if up is not None:
    try:
        raw = pd.read_csv(up, parse_dates=["timestamp"])
        missing = REQUIRED_COLS - set(raw.columns)
        if missing:
            st.error(f"Uploaded CSV is missing columns: {sorted(missing)}. Using demo data instead.")
            raw = load_default()
        else:
            st.sidebar.success(f"Using your CSV ({len(raw)} rows)")
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read CSV ({type(e).__name__}). Using demo data instead.")
        raw = load_default()
else:
    raw = load_default()

dfz = detect.add_zscores(raw)
machines = sorted(dfz.machine.unique())
statuses = {m: detect.machine_status(dfz, m) for m in machines}
flagged = [m for m in machines if statuses[m]["drifted"]]

# ---------------------------------------------------------------- header + headline
st.title("🛠️ Maintenance Copilot")
st.caption("For maintenance engineers and shift supervisors: spot machine drift early, get a grounded cause and action in seconds.")

target = flagged[0] if flagged else machines[0]
cmp_ = detect.baseline_vs_ai(dfz, target) if flagged else None
if cmp_ and cmp_["ai_flag_time"] is not None:
    saved = cmp_["hours_earlier"] * cost_per_hour
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("💰 Downtime cost avoided (est.)", f"${saved:,.0f}")
    k2.metric("⏱️ Earlier warning vs fixed alarm", f"{cmp_['hours_earlier']} h")
    k3.metric("Machines flagged", f"{len(flagged)} of {len(machines)}")
    k4.metric("Fixed-threshold alarm", "never fired" if cmp_["fixed_never_fired"] else "fired late")
    fixed_txt = "has not fired yet" if cmp_["fixed_never_fired"] else f"fired {cmp_['fixed_alarm_time']:%b %d %H:%M}"
    st.info(
        f"**Measurable improvement:** drift detection flagged **{target}** on {cmp_['ai_flag_time']:%b %d %H:%M}; "
        f"a fixed {detect.FIXED_VIB_ALARM} mm/s vibration alarm {fixed_txt}. "
        f"That is **{cmp_['hours_earlier']} hours** of extra planning time × ${cost_per_hour:,}/h = **${saved:,.0f}** "
        "of unplanned downtime avoidable (estimate: assumes the early warning converts an unplanned stop into planned maintenance)."
    )
else:
    st.success("All machines are within their normal baseline.")

tab1, tab2, tab3 = st.tabs(["📈 Fleet & alerts", "🧠 AI diagnosis", "💬 Ask about this machine"])

# ---------------------------------------------------------------- tab 1
with tab1:
    rows = []
    for m in machines:
        d = statuses[m]["drifted"]
        rows.append(
            {
                "Machine": m,
                "Status": "⚠️ Drift" if d else "✅ Normal",
                "Drifting signals": ", ".join(d.keys()) or "-",
                "First flagged": min((v["first_flag"] for v in d.values()), default=None),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    sel = st.selectbox("Machine trend", machines, index=machines.index(target), key="trend_machine")
    g = statuses[sel]["frame"].set_index("timestamp")
    chart = g[["vibration_mm_s"]].copy()
    chart[f"fixed alarm ({detect.FIXED_VIB_ALARM})"] = detect.FIXED_VIB_ALARM
    st.line_chart(chart)
    st.line_chart(g[["temperature_c", "motor_current_a"]])

# ---------------------------------------------------------------- tab 2
with tab2:
    pick = st.selectbox("Machine to diagnose", machines, index=machines.index(target), key="diag_machine")
    if not statuses[pick]["drifted"]:
        st.success(f"{pick} shows no drift. No diagnosis needed.")
    if st.button("Diagnose with AMD", type="primary", disabled=not config.is_configured()):
        with st.spinner("Retrieving manual + log context and calling the AMD model…"):
            try:
                res = copilot.analyze(statuses[pick], get_index())
                record_call(res["call"])
                st.session_state[f"res_{pick}"] = res
            except amd_client.AMDError as e:
                st.error(f"AMD endpoint error: {e}")
    res = st.session_state.get(f"res_{pick}")
    if res:
        icon = URG_ICON[res["urgency"]]
        c1, c2, c3 = st.columns(3)
        c1.metric("Urgency", f"{icon} {res['urgency'].title()}")
        c2.metric("Model confidence", f"{res['confidence']:.0%}")
        c3.metric("Latency / tokens", f"{res['call']['latency_s']} s / {res['call']['total_tokens']}")
        if res["human_review"]:
            why = "model output was malformed" if not res["parse_ok"] else (
                "confidence is low" if res["confidence"] < copilot.CONF_REVIEW_THRESHOLD else "urgency is high"
            )
            st.warning(f"👷 **Human review recommended**: {why}. Have a qualified technician confirm before acting.")
        a, b = st.columns(2)
        with a:
            st.subheader("Likely cause")
            st.write(res["cause"])
            st.subheader("Recommended action")
            st.write(res["action"])
        with b:
            st.subheader("⛑️ Safety note")
            st.error(res["safety_note"] or "Follow LOTO procedure.")
            if res["uncertainty_note"]:
                st.subheader("Uncertainty")
                st.write(res["uncertainty_note"])
        st.subheader("Sources used")
        for h in res["hits"]:
            used = h["source"] in res["sources"]
            with st.expander(("✅ " if used else "") + h["source"] + f"  (relevance {h['score']})"):
                st.write(h["text"])
        with st.expander("Sensor evidence sent to the model"):
            st.code(res["evidence"])
        if not res["parse_ok"]:
            with st.expander("Raw model output"):
                st.text(res["raw"])

# ---------------------------------------------------------------- tab 3
with tab3:
    cm = st.selectbox("Machine", machines, index=machines.index(target), key="chat_machine")
    st.caption("Answers are grounded in the manual and maintenance log; sources are listed under each answer.")
    st.session_state.setdefault("chat", [])
    for m in st.session_state["chat"]:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m.get("sources"):
                st.caption("Sources: " + " · ".join(m["sources"]))
    q = st.chat_input("e.g. How do I replace the drive-end bearing safely?", disabled=not config.is_configured())
    if q:
        st.session_state["chat"].append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Asking the AMD model…"):
                    out = copilot.ask(statuses[cm], q, get_index())
                record_call({"latency_s": out["call"]["latency_s"], "total_tokens": out["call"]["total_tokens"]})
                srcs = [h["source"] for h in out["hits"]]
                st.write(out["answer"])
                st.caption("Sources: " + " · ".join(srcs))
                st.session_state["chat"].append({"role": "assistant", "content": out["answer"], "sources": srcs})
            except amd_client.AMDError as e:
                st.error(f"AMD endpoint error: {e}")


render_amd_panel()
