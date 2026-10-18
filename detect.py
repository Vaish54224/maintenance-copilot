"""Drift detection (z-score vs baseline) and baseline-vs-AI comparison."""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data"
METRICS = ["vibration_mm_s", "temperature_c", "motor_current_a", "pressure_bar"]
BASELINE_HOURS = 7 * 24
Z_LIMIT = 3.0
SUSTAIN = 6  # consecutive hours above limit before flagging (avoids noise)

# Typical fixed-threshold alarm = manual "warning" level for vibration (mm/s)
FIXED_VIB_ALARM = 4.5


def load_sensors() -> pd.DataFrame:
    return pd.read_csv(DATA / "sensors.csv", parse_dates=["timestamp"])


def add_zscores(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, g in df.groupby("machine"):
        g = g.sort_values("timestamp").copy()
        base = g.iloc[:BASELINE_HOURS]
        for m in METRICS:
            sd = base[m].std() or 1e-9
            g[f"z_{m}"] = (g[m] - base[m].mean()) / sd
        out.append(g)
    return pd.concat(out, ignore_index=True)


def _first_sustained(mask: pd.Series, times: pd.Series, sustain: int):
    run = mask.astype(int).rolling(sustain).sum()
    hit = run[run >= sustain]
    if hit.empty:
        return None
    idx = hit.index[0] - (sustain - 1)
    return times.loc[idx]


def machine_status(dfz: pd.DataFrame, machine: str) -> dict:
    g = dfz[dfz.machine == machine].reset_index(drop=True)
    post = g.iloc[BASELINE_HOURS:].reset_index(drop=True)
    drifted = {}
    for m in METRICS:
        zc = f"z_{m}"
        t = _first_sustained(post[zc].abs() > Z_LIMIT, post["timestamp"], SUSTAIN)
        if t is not None:
            drifted[m] = {"first_flag": t, "latest_z": round(float(g[zc].iloc[-1]), 2)}
    return {"machine": machine, "drifted": drifted, "latest": g.iloc[-1], "frame": g}


def baseline_vs_ai(dfz: pd.DataFrame, machine: str = "M3-Pump") -> dict:
    """How much earlier does drift detection flag a machine than a fixed vibration alarm?"""
    st = machine_status(dfz, machine)
    g = st["frame"]
    ai_time = min((d["first_flag"] for d in st["drifted"].values()), default=None)
    fixed = g[g["vibration_mm_s"] > FIXED_VIB_ALARM]
    fixed_time = fixed["timestamp"].iloc[0] if not fixed.empty else None
    end = g["timestamp"].iloc[-1]
    # If the fixed alarm never fires in the window, the fault is still undetected: measure to end of data.
    ref = fixed_time if fixed_time is not None else end
    hours_earlier = round((ref - ai_time).total_seconds() / 3600, 1) if ai_time is not None else 0.0
    return {
        "ai_flag_time": ai_time,
        "fixed_alarm_time": fixed_time,
        "fixed_never_fired": fixed_time is None,
        "hours_earlier": hours_earlier,
        "window_end": end,
    }
