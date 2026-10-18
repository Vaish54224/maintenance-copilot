"""Generate realistic fake industrial data for the Maintenance Copilot demo.

Creates in ./data:
  sensors.csv          hourly readings, 5 machines, 14 days
  maintenance_log.csv  past work orders
  pump_manual.md       excerpt-style pump maintenance manual

A bearing fault is injected into M3-Pump starting day 9 (gradual drift).
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
DAYS = 14
FAULT_START_HOUR = 9 * 24 + 6  # fault begins on day 9, 06:00
MACHINES = {
    "M1-Press": dict(vib=2.0, temp=62, current=41, pressure=6.2),
    "M2-Conveyor": dict(vib=1.2, temp=48, current=18, pressure=0.0),
    "M3-Pump": dict(vib=2.5, temp=58, current=22, pressure=4.8),
    "M4-Compressor": dict(vib=3.0, temp=71, current=55, pressure=8.1),
    "M5-Mixer": dict(vib=1.8, temp=52, current=30, pressure=0.0),
}

OUT = Path(__file__).parent / "data"


def build_sensors() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    start = pd.Timestamp("2026-10-04 00:00")
    n = DAYS * 24
    ts = pd.date_range(start, periods=n, freq="h")
    frames = []
    for name, base in MACHINES.items():
        daily = np.sin(np.arange(n) * 2 * np.pi / 24)
        vib = base["vib"] + 0.08 * daily + rng.normal(0, 0.10, n)
        temp = base["temp"] + 1.5 * daily + rng.normal(0, 0.6, n)
        cur = base["current"] + 0.6 * daily + rng.normal(0, 0.4, n)
        pres = base["pressure"] + rng.normal(0, 0.05, n) if base["pressure"] else np.zeros(n)
        if name == "M3-Pump":
            k = np.clip(np.arange(n) - FAULT_START_HOUR, 0, None)
            ramp = (k / 24.0) ** 1.4
            vib = vib + 0.55 * ramp  # bearing wear -> vibration rises
            temp = temp + 1.8 * ramp
            cur = cur + 0.5 * ramp
            pres = pres - 0.04 * ramp
        frames.append(
            pd.DataFrame(
                dict(
                    timestamp=ts,
                    machine=name,
                    vibration_mm_s=vib.round(3),
                    temperature_c=temp.round(2),
                    motor_current_a=cur.round(2),
                    pressure_bar=np.round(pres, 3),
                )
            )
        )
    return pd.concat(frames, ignore_index=True)


LOG = [
    ("2026-03-12", "M3-Pump", "Replaced drive-end bearing 6308-2RS after high vibration alarm", "Bearing wear", 6.5),
    ("2026-05-02", "M3-Pump", "Re-greased bearings (NLGI 2 lithium complex). Vibration back to 2.4 mm/s", "Lubrication", 1.0),
    ("2026-06-19", "M3-Pump", "Mechanical seal leak, replaced seal kit", "Seal failure", 4.0),
    ("2026-07-08", "M1-Press", "Hydraulic oil filter replaced, pressure stable", "Routine", 0.5),
    ("2026-08-21", "M4-Compressor", "Replaced intake filter, temp dropped 4 C", "Filter clogging", 1.5),
    ("2026-08-30", "M3-Pump", "Coupling misalignment found at 0.15 mm, realigned with laser tool", "Misalignment", 3.0),
    ("2026-09-10", "M5-Mixer", "Gearbox oil change", "Routine", 1.0),
    ("2026-09-22", "M2-Conveyor", "Belt tension adjusted", "Routine", 0.5),
    ("2026-09-28", "M3-Pump", "Noted slight bearing noise during round, no action. Monitor.", "Observation", 0.0),
]

MANUAL = """# Centrifugal Pump P-300 (M3-Pump) Maintenance Manual

## 1. Normal operating ranges
Vibration (velocity RMS) normal: 1.5 to 3.5 mm/s. Warning above 4.5 mm/s. Danger above 7.1 mm/s.
Bearing housing temperature normal: 50 to 70 C. Alarm above 85 C.
Motor current normal: 20 to 25 A at rated load.
Discharge pressure normal: 4.5 to 5.0 bar.

## 2. Bearing fault diagnosis
A slow, steady rise in vibration velocity together with a rise in bearing temperature and slightly higher motor current is the typical signature of bearing wear or lubrication breakdown.
Sudden vibration spikes without temperature change usually indicate cavitation or loose mounting, not bearing wear.
Check bearing grease condition and quantity first. Dry or contaminated grease accelerates wear.
If vibration has risen more than 40 percent over baseline, schedule bearing inspection within 72 hours.

## 3. Bearing replacement procedure
Drive-end bearing 6308-2RS, non-drive-end bearing 6307-2RS. Estimated time 6 hours with two technicians.
Isolate and lock out the pump (LOTO) before any work. Verify zero energy state.
Allow the pump casing to cool to below 40 C before disassembly. Hot surfaces cause burns.
Always check shaft alignment after replacement; tolerance 0.05 mm.

## 4. Lubrication
Use NLGI 2 lithium complex grease. Re-grease every 2000 operating hours, about 30 g per bearing.
Over-greasing raises bearing temperature and can cause failure, do not exceed 40 g.

## 5. Safety
Never inspect or touch rotating equipment while running. Wear cut-resistant gloves and eye protection.
Failure of a bearing at full speed can cause shaft seizure, seal rupture and release of process fluid. Escalate to the shift supervisor if vibration exceeds 7.1 mm/s and stop the pump.

## 6. Cavitation and seal issues
Cavitation sounds like gravel in the pump, with unstable discharge pressure. Check suction valve and NPSH.
Seal leaks show as fluid at the gland; replace seal kit, check shaft runout below 0.05 mm.
"""


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = build_sensors()
    # fixed "\n" line endings so the files are byte-identical on Windows and Linux
    df.to_csv(OUT / "sensors.csv", index=False, lineterminator="\n")
    pd.DataFrame(LOG, columns=["date", "machine", "note", "category", "downtime_hours"]).to_csv(
        OUT / "maintenance_log.csv", index=False, lineterminator="\n"
    )
    (OUT / "pump_manual.md").write_text(MANUAL, encoding="utf-8", newline="\n")
    print(f"Wrote {len(df)} sensor rows, {len(LOG)} log entries, manual to {OUT}")


if __name__ == "__main__":
    main()
