#!/usr/bin/env python3
"""
replot_accuracy_L64_reviewer6.py

Reviewer 6 revision for L = 64:
- Y-axis fixed to 0.60--1.00
- No average-value text box covering low-temperature data
- Legend placed in lower-left corner
- Reads metrics_per_temp_bin.csv from the correct Generated_L64 folder
- Saves revised PDF and PNG in the same folder
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Larger bold publication-style fonts
plt.rcParams.update({
    "font.size": 15,
    "font.weight": "bold",
    "axes.labelsize": 17,
    "axes.labelweight": "bold",
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
})

# ============================================================
# USER SETTINGS
# ============================================================

L = 128

BASE_DIR = Path("/home/partha02965/JOB_SCRIPTA/Generated_L128")
CSV_FILE = BASE_DIR / "metrics_per_temp_bin.csv"

TC = 2.269

OUTPUT_PDF = BASE_DIR / f"accuracy_vs_temperature_L{L}_revised.pdf"
OUTPUT_PNG = BASE_DIR / f"accuracy_vs_temperature_L{L}_revised.png"

# ============================================================
# LOAD DATA
# ============================================================

if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find input CSV:\\n{CSV_FILE}\\n"
        f"Check that metrics_per_temp_bin.csv exists inside {BASE_DIR}."
    )

df = pd.read_csv(CSV_FILE)

required_columns = [
    "T_center",
    "Accuracy_missing",
    "Accuracy_observed",
    "Accuracy_overall",
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "The CSV file is missing required columns: "
        + ", ".join(missing_columns)
        + "\\nAvailable columns are:\\n"
        + ", ".join(df.columns.astype(str))
    )

# ============================================================
# EXTRACT VALUES
# ============================================================

T = df["T_center"].to_numpy()
accuracy_missing = df["Accuracy_missing"].to_numpy()
accuracy_observed = df["Accuracy_observed"].to_numpy()
accuracy_overall = df["Accuracy_overall"].to_numpy()

# ============================================================
# CREATE REVISED FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(8.5, 5.8))

ax.plot(
    T,
    accuracy_missing,
    marker="^",
    linewidth=2.0,
    markersize=5,
    label="Missing spins",
)

ax.plot(
    T,
    accuracy_observed,
    marker="v",
    linewidth=2.0,
    markersize=5,
    label="Observed spins",
)

ax.plot(
    T,
    accuracy_overall,
    marker="o",
    linewidth=2.0,
    markersize=5,
    label="Overall",
)

ax.axvline(
    TC,
    linestyle="--",
    linewidth=1.8,
    label=r"$T_c$",
)

# Reviewer 6 requested this range
ax.set_ylim(0.60, 1.00)

ax.set_xlim(float(T.min()), float(T.max()))

ax.set_xlabel("Temperature $T$", fontsize=17, fontweight="bold")
ax.set_ylabel("Reconstruction Accuracy", fontsize=17, fontweight="bold")

ax.set_title(
    rf"Reconstruction Accuracy vs Temperature ($L={L}$)",
    fontsize=18,
    fontweight="bold"
)

# Keep legend away from low-temperature high-accuracy curves
ax.legend(
    loc="lower left",
    fontsize=13,
    frameon=True
)

ax.grid(alpha=0.25)

ax.tick_params(
    axis="both",
    labelsize=14,
    width=1.4,
    length=5
)

for tick in ax.get_xticklabels():
    tick.set_fontweight("bold")

for tick in ax.get_yticklabels():
    tick.set_fontweight("bold")

plt.tight_layout()

plt.savefig(
    OUTPUT_PDF,
    bbox_inches="tight"
)

plt.savefig(
    OUTPUT_PNG,
    dpi=300,
    bbox_inches="tight"
)

print("Reviewer 6 revised figure created successfully.")
print(f"Input CSV : {CSV_FILE}")
print(f"PDF saved : {OUTPUT_PDF}")
print(f"PNG saved : {OUTPUT_PNG}")

plt.show()
