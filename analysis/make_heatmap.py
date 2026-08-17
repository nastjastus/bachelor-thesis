"""
make_heatmap.py

Draws the split contrast figure: the mean marginal contribution of each
encoding, per log, for both split strategies side by side.

The marginal contribution is computed in the same way as in
marginal_contribution.py. For a given encoding it averages the change in error
over all 32 pairs of configurations that differ only in that encoding, and the
result is averaged over the three models. An asterisk marks encodings whose
sign is the same in all three.

Input:  Ergebnisse_aller_Logs.xlsx (next to this script)
Output: split_contrast.png (written next to this script)

Run:  python analysis/make_heatmap.py
"""

import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# Resolve input/output next to this script, independent of the working directory
HERE = os.path.dirname(os.path.abspath(__file__))
XL_PATH = os.path.join(HERE, "Ergebnisse_aller_Logs.xlsx")
OUT_PATH = os.path.join(HERE, "split_contrast.png")
METRIC = "Test MAE (h)"
MODELS = ["Random Forest", "CART", "XGBoost"]
ENCODINGS = ["E1", "E2", "E3", "E4", "E5", "E6"]

# Sheet pattern and display name, in the order used in the thesis.
# The truncated Helpdesk sheet carries the split in the middle.
LOGS = [
    ("BPIC2011_{split}", "BPIC2011"),
    ("Production_{split}", "Production"),
    ("rtf_{split}", "RTF"),
    ("BPIC2012_w_{split}", "BPIC2012-W"),
    ("BPIC2012_{split}", "BPIC2012"),
    ("Sepsis_{split}", "Sepsis"),
    ("BPIC2017_{split}", "BPIC2017"),
    ("Helpdesk_{split}", "Helpdesk"),
    ("Domestic_{split}", "BPIC2020"),
    ("helpdesk_{split}_resolved", "Helpdesk (T)"),
]

# TUM colours
TUM_BLUE = (0 / 255, 101 / 255, 189 / 255)
TUM_ORANGE = (227 / 255, 114 / 255, 34 / 255)
WHITE = (1, 1, 1)

# Values beyond this limit are shown at the colour limit, so that the smaller
# contributions remain visible. The numbers in the cells are unaffected.
COLOUR_LIMIT = 12.0


def parse_configuration(label):
    """Turns a configuration label into the set of encodings it contains."""
    label = str(label)
    if label == "Baseline":
        return frozenset()
    if "Uhr" in label:
        return None
    return frozenset(re.findall(r"E\d", label))


def marginal_contribution(sheet, model, encoding, xl):
    """Mean change in error when the encoding is added to any combination."""
    df = xl.parse(sheet)
    df = df[df["Modell"] == model]
    if df.empty:
        return None

    baseline = df[df["Konfiguration"] == "Baseline"][METRIC].values[0]

    errors = {}
    for _, row in df.iterrows():
        combination = parse_configuration(row["Konfiguration"])
        if combination is None:
            continue
        errors[combination] = row[METRIC]

    deltas = [100 * (error - errors[combination - {encoding}]) / baseline
              for combination, error in errors.items()
              if encoding in combination and (combination - {encoding}) in errors]

    return float(np.mean(deltas)) if deltas else None


def build_matrix(split, xl):
    """Returns the value matrix and a mask of sign-consistent entries."""
    values = np.full((len(LOGS), len(ENCODINGS)), np.nan)
    consistent = np.zeros_like(values, dtype=bool)

    for i, (pattern, _) in enumerate(LOGS):
        sheet = pattern.format(split=split)
        if sheet not in xl.sheet_names:
            print(f"  sheet not found, skipped: {sheet}")
            continue
        for j, encoding in enumerate(ENCODINGS):
            per_model = [marginal_contribution(sheet, m, encoding, xl)
                         for m in MODELS]
            per_model = [v for v in per_model if v is not None]
            if not per_model:
                continue
            values[i, j] = np.mean(per_model)
            consistent[i, j] = all(v < 0 for v in per_model)

    return values, consistent


def draw_panel(ax, values, consistent, title, norm, cmap):
    image = ax.imshow(np.clip(values, -COLOUR_LIMIT, COLOUR_LIMIT),
                      cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(ENCODINGS)))
    ax.set_xticklabels(ENCODINGS, fontsize=13)
    ax.set_yticks(range(len(LOGS)))
    ax.set_yticklabels([name for _, name in LOGS], fontsize=11)
    ax.set_title(title, fontsize=13, pad=8)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isnan(values[i, j]):
                continue
            value = values[i, j]
            text = f"{value:.1f}"
            if consistent[i, j] and abs(value) >= 0.5:
                text += "*"
            ax.text(j, i, text, ha="center", va="center", fontsize=10,
                    color="white" if abs(value) > COLOUR_LIMIT * 0.55 else "black")

    # thin white grid between the cells
    ax.set_xticks(np.arange(-0.5, len(ENCODINGS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(LOGS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)

    return image


def main():
    xl = pd.ExcelFile(XL_PATH)
    cmap = LinearSegmentedColormap.from_list(
        "tum_diverging", [TUM_BLUE, WHITE, TUM_ORANGE], N=256)
    norm = TwoSlopeNorm(vmin=-COLOUR_LIMIT, vcenter=0, vmax=COLOUR_LIMIT)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.2))

    for ax, (split, title) in zip(axes, [("random", "Random split"),
                                         ("temporal", "Temporal split")]):
        values, consistent = build_matrix(split, xl)
        image = draw_panel(ax, values, consistent, title, norm, cmap)

    axes[1].set_yticklabels([])

    cbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, extend="both")
    cbar.set_label("mean marginal contribution (percentage points, MAE)",
                   fontsize=10)
    cbar.ax.tick_params(labelsize=10)

    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()