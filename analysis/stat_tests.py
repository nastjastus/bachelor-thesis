"""
stat_tests.py

Runs the significance tests reported in the evaluation chapter, following the
procedure used in the cross-benchmark for remaining time prediction (Verenich
et al. 2019), which in turn follows Demsar (2006).

Wilcoxon signed-rank test:
    Tests whether the deltas of a predefined configuration are systematically
    negative across the logs. Only predefined configurations are tested, since
    the best combination of a log is selected on the test data and is therefore
    optimistically biased. The best combination is reported alongside for
    comparison, but marked as biased.

Friedman test with Nemenyi post hoc:
    Ranks the baseline and the individual encodings within each log and tests
    whether the mean ranks differ. The critical difference of the Nemenyi test
    is CD = q * sqrt(k * (k + 1) / (6 * N)) with k configurations and N logs.
    The value q = 2.949 is the studentised range statistic for k = 7 at a
    significance level of 0.05, divided by sqrt(2), as tabulated in Demsar
    (2006). For k = 7 and N = 7 this gives CD = 3.41.

BPIC2011 and Production are excluded from the tests: the former has no
well-defined case end, the latter retains only 33 test cases after filtering.

Input:  Ergebnisse_aller_Logs.xlsx (next to this script)
Output: printed to the console

Run:  python analysis/stat_tests.py
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

# Resolve the input next to this script, independent of the working directory
XL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "Ergebnisse_aller_Logs.xlsx")
METRIC = "Test MAE (h)"
MODELS = ["Random Forest", "CART", "XGBoost"]

# Logs used for the tests; BPIC2011 and Production are left out
INTERPRETABLE = ["rtf", "BPIC2012_w", "BPIC2012", "Sepsis",
                 "BPIC2017", "Helpdesk", "Domestic"]

# Configurations fixed in advance, not selected on the test data
PREDEFINED = {
    "all six": "+ E1 + E2 + E3 + E4 + E5 + E6",
    "E1 + E3": "+ E1 + E3",
    "E1":      "+ E1",
}

# Configurations compared in the Friedman test
RANKED = ["Baseline", "+ E1", "+ E2", "+ E3", "+ E4", "+ E5", "+ E6"]

# Studentised range statistic for k = 7 at alpha = 0.05, divided by sqrt(2)
NEMENYI_Q = 2.949


def delta(xl, sheet, model, configuration):
    """Delta of a configuration to the baseline of the same log, in percent."""
    df = xl.parse(sheet)
    df = df[df["Modell"] == model]
    if df.empty:
        return np.nan
    baseline = df[df["Konfiguration"] == "Baseline"][METRIC].values[0]
    row = df[df["Konfiguration"] == configuration]
    if row.empty:
        return np.nan
    return 100 * (row[METRIC].values[0] - baseline) / baseline


def best_delta(xl, sheet, model):
    """Delta of the best of the 64 combinations, which is biased."""
    df = xl.parse(sheet)
    df = df[df["Modell"] == model]
    if df.empty:
        return np.nan
    baseline = df[df["Konfiguration"] == "Baseline"][METRIC].values[0]
    encodings_only = df[(~df["Konfiguration"].astype(str).str.contains("Uhr"))
                        & (df["Konfiguration"] != "Baseline")]
    return 100 * (encodings_only[METRIC].min() - baseline) / baseline


def wilcoxon(values, label):
    """Tests whether the deltas are systematically below or above zero."""
    values = np.array([v for v in values if not np.isnan(v)])
    if len(values) < 5:
        print(f"    {label:18} too few logs (n={len(values)})")
        return

    p_improve = stats.wilcoxon(values, alternative="less")[1]
    p_worsen = stats.wilcoxon(values, alternative="greater")[1]
    median = np.median(values)
    negative = int((values < 0).sum())

    print(f"    {label:18} n={len(values)}  median {median:+6.2f}%  "
          f"negative {negative}/{len(values)}  "
          f"p(improvement)={p_improve:.4f}  p(deterioration)={p_worsen:.4f}")


def friedman(xl, split, model):
    """Friedman test over baseline and individual encodings, logs as blocks."""
    matrix = []
    for log in INTERPRETABLE:
        sheet = f"{log}_{split}"
        df = xl.parse(sheet)
        df = df[df["Modell"] == model]
        if df.empty:
            continue
        values = []
        for configuration in RANKED:
            row = df[df["Konfiguration"] == configuration]
            if row.empty:
                values = []
                break
            values.append(row[METRIC].values[0])
        if values:
            matrix.append(values)

    if len(matrix) < 3:
        print("    too few logs for the Friedman test")
        return

    matrix = np.array(matrix)
    n_logs, n_configs = matrix.shape

    chi2, p = stats.friedmanchisquare(*[matrix[:, j] for j in range(n_configs)])
    print(f"    Friedman over {n_logs} logs: chi2={chi2:.2f}, p={p:.4f}")

    ranks = np.apply_along_axis(stats.rankdata, 1, matrix)
    mean_ranks = ranks.mean(axis=0)

    cd = NEMENYI_Q * np.sqrt(n_configs * (n_configs + 1) / (6 * n_logs))
    print(f"    Nemenyi critical difference: {cd:.2f}")

    print("    mean ranks (1 = best):")
    for j in np.argsort(mean_ranks):
        difference = mean_ranks[0] - mean_ranks[j]
        marker = ""
        if j != 0:
            marker = "  differs from baseline" if difference > cd else ""
        print(f"       {RANKED[j]:10} {mean_ranks[j]:.2f}{marker}")


def main():
    xl = pd.ExcelFile(XL_PATH)

    for split in ["random", "temporal"]:
        print("\n" + "=" * 78)
        print(f"{split.upper()} SPLIT, {METRIC}, {len(INTERPRETABLE)} interpretable logs")
        print("=" * 78)

        for model in MODELS:
            print(f"\n  Model: {model}")
            print("   Wilcoxon signed-rank test:")
            for label, configuration in PREDEFINED.items():
                values = [delta(xl, f"{log}_{split}", model, configuration)
                          for log in INTERPRETABLE]
                wilcoxon(values, label)
            values = [best_delta(xl, f"{log}_{split}", model)
                      for log in INTERPRETABLE]
            wilcoxon(values, "best (biased)")

        print(f"\n  Friedman test ({MODELS[0]}):")
        friedman(xl, split, MODELS[0])


if __name__ == "__main__":
    main()