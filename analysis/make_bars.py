"""
make_bars.py

Draws the results figure: the delta to the baseline per log, for the
configuration containing all six encodings and for the best of the 64
combinations, shown for both split strategies side by side.

All values are taken from the Random Forest results, matching the results table
in the thesis. The best combination is selected on the test data and is
therefore optimistically biased; it is shown in the lighter colour.

Input:  Ergebnisse_aller_Logs.xlsx (next to this script)
Output: results_per_log.png (written next to this script)

Run:  python analysis/make_bars.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Resolve input/output next to this script, independent of the working directory
HERE = os.path.dirname(os.path.abspath(__file__))
XL_PATH = os.path.join(HERE, "Ergebnisse_aller_Logs.xlsx")
OUT_PATH = os.path.join(HERE, "results_per_log.png")
METRIC = "Test MAE (h)"
MODEL = "Random Forest"
ALL_SIX = "+ E1 + E2 + E3 + E4 + E5 + E6"

# Sheet prefix and display name, in the order used in the results table.
# The two logs without an interpretable result come first, the truncated
# Helpdesk variant last; both groups are separated by a dotted line.
LOGS = [
    ("BPIC2011", "BPIC2011"),
    ("Production", "Production"),
    ("rtf", "Road Traffic Fines"),
    ("BPIC2012_w", "BPIC2012-W"),
    ("BPIC2012", "BPIC2012"),
    ("Sepsis", "Sepsis"),
    ("BPIC2017", "BPIC2017"),
    ("Helpdesk", "Helpdesk"),
    ("Domestic", "Domestic Decl."),
]
TRUNCATED = ("helpdesk_{split}_resolved", "Helpdesk (truncated)")

# TUM colours
TUM_BLUE = (0 / 255, 101 / 255, 189 / 255)
TUM_BLUE_LIGHT = (152 / 255, 198 / 255, 234 / 255)
TUM_GREY = (0.6, 0.6, 0.6)


def deltas(sheet, xl):
    """Returns the delta of the all-six configuration and of the best one."""
    df = xl.parse(sheet)
    df = df[df["Modell"] == MODEL]
    baseline = df[df["Konfiguration"] == "Baseline"][METRIC].values[0]

    all_six = 100 * (df[df["Konfiguration"] == ALL_SIX][METRIC].values[0]
                     - baseline) / baseline

    encodings_only = df[(~df["Konfiguration"].astype(str).str.contains("Uhr"))
                        & (df["Konfiguration"] != "Baseline")]
    best = 100 * (encodings_only[METRIC].min() - baseline) / baseline

    return all_six, best


def collect(xl):
    """Builds the value matrix and the row labels."""
    rows, labels = [], []

    for prefix, name in LOGS:
        random = deltas(f"{prefix}_random", xl)
        temporal = deltas(f"{prefix}_temporal", xl)
        rows.append([random[0], random[1], temporal[0], temporal[1]])
        labels.append(name)

    pattern, name = TRUNCATED
    random = deltas(pattern.format(split="random"), xl)
    temporal = deltas(pattern.format(split="temporal"), xl)
    rows.append([random[0], random[1], temporal[0], temporal[1]])
    labels.append(name)

    return np.array(rows), labels


def main():
    xl = pd.ExcelFile(XL_PATH)
    data, labels = collect(xl)

    y = np.arange(len(labels))
    height = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(9, 5.4), sharey=True)

    for ax, (i_all, i_best, title) in zip(
            axes, [(0, 1, "Random split"), (2, 3, "Temporal split")]):

        ax.barh(y - height / 2, data[:, i_all], height=height,
                color=TUM_BLUE, label="all six encodings")
        ax.barh(y + height / 2, data[:, i_best], height=height,
                color=TUM_BLUE_LIGHT, label="best combination")

        ax.axvline(0, color="black", linewidth=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_title(title, fontsize=13, pad=8)
        ax.set_xlabel("delta to baseline in percent (MAE)", fontsize=11)
        ax.grid(axis="x", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=10)

        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(TUM_GREY)
        ax.spines["bottom"].set_color(TUM_GREY)

        # separate the uninterpretable logs at the top and the truncated
        # variant at the bottom
        ax.axhline(1.5, color=TUM_GREY, linewidth=0.8, linestyle=":")
        ax.axhline(len(labels) - 1.5, color=TUM_GREY, linewidth=0.8,
                   linestyle=":")

    axes[0].invert_yaxis()

    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=11,
               loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=2,
               frameon=False)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()