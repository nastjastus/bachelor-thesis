"""
marginal_contribution.py

Computes the mean marginal contribution of each inter-case encoding from the
result file produced by the pipeline.

The delta of a single encoding added on its own underestimates its contribution
if the encoding only becomes effective in combination with others. The marginal
contribution avoids this: for a given encoding, it averages the change in error
over all 32 pairs of configurations that differ only in that encoding, that is,
Baseline vs E1, E2 vs E1E2, E3E4 vs E1E3E4, and so on.

The result is averaged over the three models. An encoding is marked with an
asterisk if its sign is the same in all three, which indicates that the
contribution is not an artefact of a single model.

Input:  Ergebnisse_aller_Logs.xlsx (next to this script), one sheet per log
        and split, each holding the 64 configurations for all three models.
Output: marginal_contribution.csv (written next to this script)

Run:  python analysis/marginal_contribution.py
"""

import os
import re

import numpy as np
import pandas as pd

# Resolve input/output next to this script, independent of the working directory
HERE = os.path.dirname(os.path.abspath(__file__))
XL_PATH = os.path.join(HERE, "Ergebnisse_aller_Logs.xlsx")
METRIC = "Test MAE (h)"
MODELS = ["Random Forest", "CART", "XGBoost"]
ENCODINGS = ["E1", "E2", "E3", "E4", "E5", "E6"]

# Sheet name per log, in the order used in the thesis.
# Note: the helpdesk_resolved sheets carry the split in the middle
# (helpdesk_{split}_resolved), unlike the others which end in _{split}.
LOGS = [
    ("Sepsis_{split}", "Sepsis"),
    ("BPIC2011_{split}", "BPIC2011"),
    ("Production_{split}", "Production"),
    ("BPIC2017_{split}", "BPIC2017"),
    ("Helpdesk_{split}", "Helpdesk"),
    ("BPIC2012_w_{split}", "BPIC2012-W"),
    ("BPIC2012_{split}", "BPIC2012"),
    ("Domestic_{split}", "BPIC2020 Domestic Decl."),
    ("rtf_{split}", "Road Traffic Fines"),
    ("helpdesk_{split}_resolved", "Helpdesk (truncated)"),
]


def parse_configuration(label):
    """Turns a configuration label into the set of encodings it contains.

    Returns None for the clock reference configuration, which is not part of
    the 64 combinations.
    """
    label = str(label)
    if label == "Baseline":
        return frozenset()
    if "Uhr" in label:
        return None
    return frozenset(re.findall(r"E\d", label))


def marginal_contribution(sheet, model, encoding, xl):
    """Mean change in error when the encoding is added to any combination.

    Expressed in percentage points of the baseline error of the same log.
    """
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

    deltas = []
    for combination, error in errors.items():
        if encoding not in combination:
            continue
        without = combination - {encoding}
        if without in errors:
            deltas.append(100 * (error - errors[without]) / baseline)

    return float(np.mean(deltas)) if deltas else None


def main():
    xl = pd.ExcelFile(XL_PATH)
    rows = []

    for split in ["random", "temporal"]:
        for sheet_pattern, log_name in LOGS:
            sheet = sheet_pattern.format(split=split)
            if sheet not in xl.sheet_names:
                print(f"  sheet not found, skipped: {sheet}")
                continue

            row = {"log": log_name, "split": split}
            for encoding in ENCODINGS:
                values = [marginal_contribution(sheet, m, encoding, xl)
                          for m in MODELS]
                values = [v for v in values if v is not None]
                if not values:
                    continue
                row[encoding] = np.mean(values)
                row[f"{encoding}_consistent"] = all(v < 0 for v in values)
            rows.append(row)

    result = pd.DataFrame(rows)

    for split in ["random", "temporal"]:
        subset = result[result["split"] == split]
        print(f"\nMean marginal contribution in percentage points ({split} split)")
        header = f"{'Log':26}" + "".join(f"{e:>7}" for e in ENCODINGS)
        print(header)
        for _, r in subset.iterrows():
            line = f"{r['log']:26}"
            for e in ENCODINGS:
                mark = "*" if r.get(f"{e}_consistent") else " "
                line += f"{r[e]:6.2f}{mark}"
            print(line)

    print("\nAn asterisk marks encodings whose sign is the same in all three models.")

    result.to_csv(os.path.join(HERE, "marginal_contribution.csv"), index=False)
    print("\nSaved: marginal_contribution.csv")


if __name__ == "__main__":
    main()