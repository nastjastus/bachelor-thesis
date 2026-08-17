"""
check_drift.py

Measures the distribution shift of E1 (open_cases_at_time) between the
training and test set under the temporal split.

Background: the feature-range diagnostic of the pipeline only counts test
values outside the training interval. A shift WITHIN the interval stays
invisible. The Kolmogorov-Smirnov statistic captures it: it is the maximum
difference of the cumulative distribution functions, 0 = identical,
1 = fully separated.

The means of both sets are also printed so that the direction of the shift
becomes visible, together with the feature range for comparison.

Run:  python analysis/check_drift.py
"""

import os
import sys
import numpy as np
import pandas as pd
import pm4py
from scipy import stats

# Allow importing config.py from the project root when run from analysis/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

CASE_COL = "case:concept:name"
ACT_COL = "concept:name"
TS_COL = "time:timestamp"

REAL_LOGS = ["bpic2011", "production", "rtf", "bpic2012_w",
             "bpic2012_all", "sepsis", "bpic2017", "helpdesk", "domestic"]

TRAIN_RATIO = 0.8


def complete_cases(df, end_acts, end_mode):
    if not end_acts:
        return set(df[CASE_COL].unique())
    if end_mode == "contains":
        return set(df[df[ACT_COL].isin(end_acts)][CASE_COL].unique())
    last = df.sort_values(TS_COL).groupby(CASE_COL)[ACT_COL].last()
    return set(last[last.isin(end_acts)].index)


def analyse(name):
    """Reconstructs the temporal split and measures the shift of E1."""
    cfg = config.LOGS[name]
    df = pm4py.read_xes(cfg["path"])
    df[TS_COL] = pd.to_datetime(df[TS_COL], utc=True, format="mixed").dt.tz_localize(None)
    df = df.sort_values(TS_COL).reset_index(drop=True)

    # Completeness filter
    keep = complete_cases(df, cfg.get("end_acts"), cfg.get("end_mode", "last"))
    df = df[df[CASE_COL].isin(keep)].copy()

    # Reconstruct E1: timeline of case starts (+1) and case ends (-1)
    starts = df.groupby(CASE_COL)[TS_COL].min()
    ends = df.groupby(CASE_COL)[TS_COL].max()
    timeline = pd.concat([
        pd.DataFrame({"t": starts.values, "d": 1}),
        pd.DataFrame({"t": ends.values, "d": -1}),
    ]).sort_values("t")
    timeline["open"] = timeline["d"].cumsum()
    df["open_cases_at_time"] = pd.merge_asof(
        df[[TS_COL]].sort_values(TS_COL),
        timeline[["t", "open"]].rename(columns={"t": TS_COL}),
        on=TS_COL, direction="backward")["open"].values

    # Temporal split at case level
    order = starts.sort_values()
    n_train = int(len(order) * TRAIN_RATIO)
    split_ts = order.iloc[n_train]
    train_cases = set(order.index[:n_train])
    # Drop crossing cases
    train_cases = {c for c in train_cases if ends[c] <= split_ts}
    test_cases = set(order.index[n_train:])

    tr = df[df[CASE_COL].isin(train_cases)]
    te = df[df[CASE_COL].isin(test_cases)]

    a = tr["open_cases_at_time"].dropna()
    b = te["open_cases_at_time"].dropna()
    if len(a) < 10 or len(b) < 10:
        return None

    ks = stats.ks_2samp(a, b).statistic
    out = 100 * ((b < a.min()) | (b > a.max())).mean()
    return dict(log=name, ks=ks, out_of_range=out,
                mean_train=a.mean(), mean_test=b.mean())


if __name__ == "__main__":
    rows = []
    for name in REAL_LOGS:
        try:
            r = analyse(name)
            if r:
                rows.append(r)
            print(f"  {name} done")
        except Exception as e:
            print(f"  {name} skipped: {e}")

    res = pd.DataFrame(rows)
    print("\nDistribution shift of E1, train -> test (temporal split)")
    print(f"{'Log':16}{'KS':>8}{'out%':>12}{'O train':>13}{'O test':>13}")
    for _, r in res.iterrows():
        print(f"{r['log']:16}{r['ks']:8.3f}{r['out_of_range']:11.1f}%"
              f"{r['mean_train']:13.1f}{r['mean_test']:13.1f}")

    print("\nReading: KS close to 0 = no shift, KS close to 1 = fully separated.")
    print("The out% column corresponds to the feature-range diagnostic of the pipeline.")

    res.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "drift_check.csv"),
               index=False)
    print("\nSaved: drift_check.csv")