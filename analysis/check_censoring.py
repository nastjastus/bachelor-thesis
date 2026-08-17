"""
check_censoring.py

Measures right-censoring and the resulting selection bias directly in the
data, without a model and without a timestamp feature.

Idea: if a log is right-censored, cases that start late cannot be long,
since they would otherwise extend beyond the end of the log. Late cases
are therefore systematically shorter, and the correlation between case
start time and case duration is clearly negative.

The completeness filter does not fix this but turns it into a selection
bias: of the late-starting cases, only the short ones survive, because
only those could finish before the end of the log. The correlation is
therefore computed before AND after the filter.

Expectation:
  censored logs -> clearly negative correlation (even after the filter)
  clean logs    -> correlation close to zero

Run:  python analysis/check_censoring.py
"""

import os
import sys
import pandas as pd
import numpy as np
import pm4py

# Allow importing config.py from the project root when run from analysis/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

CASE_COL = "case:concept:name"
ACT_COL = "concept:name"
TS_COL = "time:timestamp"

# only the real logs; synthetic ones have no case end by construction
REAL_LOGS = ["bpic2011", "production", "rtf", "bpic2012_w",
             "bpic2012_all", "sepsis", "bpic2017", "helpdesk"]


def case_table(df):
    """Per case: start time, end time, duration in hours."""
    g = df.groupby(CASE_COL)[TS_COL]
    t = pd.DataFrame({"start": g.min(), "end": g.max()})
    t["duration_h"] = (t["end"] - t["start"]).dt.total_seconds() / 3600.0
    return t


def corr_start_duration(t):
    """Correlation between start time and case duration."""
    if len(t) < 3:
        return np.nan
    start_num = t["start"].astype("int64").to_numpy(dtype=float)
    return float(np.corrcoef(start_num, t["duration_h"].to_numpy())[0, 1])


def complete_cases(df, end_acts, end_mode):
    """Returns the set of completed case IDs."""
    if not end_acts:
        return set(df[CASE_COL].unique())
    if end_mode == "contains":
        return set(df[df[ACT_COL].isin(end_acts)][CASE_COL].unique())
    last = df.sort_values(TS_COL).groupby(CASE_COL)[ACT_COL].last()
    return set(last[last.isin(end_acts)].index)


def analyse(name):
    cfg = config.LOGS[name]
    df = pm4py.read_xes(cfg["path"])
    df[TS_COL] = pd.to_datetime(df[TS_COL], utc=True, format="mixed").dt.tz_localize(None)

    t_all = case_table(df)
    r_all = corr_start_duration(t_all)

    keep = complete_cases(df, cfg.get("end_acts"), cfg.get("end_mode", "last"))
    t_keep = t_all.loc[sorted(keep)]
    r_keep = corr_start_duration(t_keep)

    # Ratio of log length to mean case duration, for cross-comparison
    log_len_h = (df[TS_COL].max() - df[TS_COL].min()).total_seconds() / 3600.0
    ratio = log_len_h / t_all["duration_h"].mean() if t_all["duration_h"].mean() > 0 else np.nan

    return dict(log=name,
                cases=len(t_all),
                kept_pct=100 * len(t_keep) / len(t_all),
                corr_all=r_all,
                corr_filtered=r_keep,
                ratio=ratio)


if __name__ == "__main__":
    rows = []
    for name in REAL_LOGS:
        try:
            rows.append(analyse(name))
            print(f"  {name} done")
        except Exception as e:
            print(f"  {name} skipped: {e}")

    res = pd.DataFrame(rows).sort_values("corr_filtered")

    print("\nSelection bias: corr(case start time, case duration)")
    print(f"{'Log':14}{'Cases':>8}{'kept%':>11}{'corr all':>11}{'corr filt.':>13}{'ratio':>13}")
    for _, r in res.iterrows():
        print(f"{r['log']:14}{r['cases']:8.0f}{r['kept_pct']:11.1f}"
              f"{r['corr_all']:11.3f}{r['corr_filtered']:13.3f}{r['ratio']:13.1f}")

    print("\nReading: strongly negative = late cases are systematically shorter,")
    print("i.e. right-censored. Close to zero = clean log.")

    res.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "censoring_check.csv"),
               index=False)
    print("\nSaved: censoring_check.csv")