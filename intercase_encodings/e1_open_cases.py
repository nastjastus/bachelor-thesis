"""
e1_open_cases.py

Encoding 1: open_cases_at_time

Measures the global system load at the exact time of an event.
How many cases are running simultaneously? Snapshot of the total workload.
"""

import pandas as pd
import numpy as np


def compute(df, case_times, ts_col):
    # For each case: start = +1, end = -1
    starts = case_times[["start_time"]].rename(columns={"start_time": "time"})
    starts["delta"] = 1
    ends = case_times[["end_time"]].rename(columns={"end_time": "time"})
    ends["delta"] = -1

    # Build timeline and compute running sum
    timeline = (
        pd.concat([starts, ends], ignore_index=True)
          .sort_values("time")
          .reset_index(drop=True)
    )
    timeline["open_cases_after"] = timeline["delta"].cumsum()

    # Carry the value onto each event
    df = pd.merge_asof(
        df,
        timeline[["time", "open_cases_after"]],
        left_on=ts_col,
        right_on="time",
        direction="backward",
    )
    df["open_cases_at_time"] = df["open_cases_after"].fillna(0).astype(int)

    return df