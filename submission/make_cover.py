"""Generate a simple cover image (1600x900) with matplotlib. Output: cover.png"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 14, 400)
vib = 2.5 + 0.08 * np.sin(x * 6) + np.where(x > 9, ((x - 9) ** 1.4) * 0.55 * 0.5, 0)
fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
fig.patch.set_facecolor("#0b0f17")
ax.set_facecolor("#0b0f17")
ax.plot(x, vib, color="#ff4b4b", lw=4)
ax.axhline(4.5, color="#8aa0b8", ls="--", lw=2)
ax.axvline(10.3, color="#35e0c2", lw=2.5)
ax.text(0.3, 4.65, "fixed alarm", color="#8aa0b8", fontsize=18)
ax.text(10.4, 2.1, "drift flagged\n39 h earlier", color="#35e0c2", fontsize=20, va="bottom")
ax.set_ylim(1.5, 7.5)
ax.axis("off")
fig.text(0.06, 0.88, "Maintenance Copilot", color="white", fontsize=58, fontweight="bold")
fig.text(0.06, 0.80, "Catch failures early · powered by AMD", color="#ff4b4b", fontsize=28)
plt.subplots_adjust(left=0.06, right=0.97, top=0.75, bottom=0.08)
out = Path(__file__).parent / "cover.png"
fig.savefig(out, facecolor=fig.get_facecolor())
print("wrote", out)
