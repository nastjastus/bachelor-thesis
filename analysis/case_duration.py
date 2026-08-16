"""
Recomputes the mean case duration for BPIC2017 and Domestic.

Important: computed on the FULL log (before the completeness filter), so
that the values match the other seven logs and the ratio column. The
timestamps are cleaned exactly as in pipeline.py (utc -> tz removed).

Run:  python analysis/case_duration.py
"""

import pandas as pd
import pm4py

CASE_COL = "case:concept:name"
TS_COL   = "time:timestamp"

LOGS = {
    "bpic2017": "data/BPI Challenge 2017 - Offer log.xes",
    "domestic": "data/DomesticDeclarations.xes",
}


def analyse(name, path):
    df = pm4py.read_xes(path)
    df[TS_COL] = pd.to_datetime(df[TS_COL], utc=True).dt.tz_localize(None)

    # Duration per case = last minus first event
    duration = df.groupby(CASE_COL)[TS_COL].agg(lambda x: x.max() - x.min())
    duration_h = duration.dt.total_seconds() / 3600.0

    # Log length = full observation period
    log_len_h = (df[TS_COL].max() - df[TS_COL].min()).total_seconds() / 3600.0

    mean_h   = duration_h.mean()
    median_h = duration_h.median()
    ratio    = log_len_h / mean_h

    print(f"\n=== {name} ===")
    print(f"  Cases:                 {df[CASE_COL].nunique():,}")
    print(f"  Mean case duration:    {mean_h:,.1f} h")
    print(f"  Median case duration:  {median_h:,.1f} h")
    print(f"  Total log length:      {log_len_h:,.1f} h")
    print(f"  Ratio (log / mean):    {ratio:.1f}")


if __name__ == "__main__":
    for name, path in LOGS.items():
        analyse(name, path)
    print("\nDone.")