"""
e3_peer_window.py

Encoding 3: peer_cases_in_window

Measures the arrival rate of new cases within the time window.
Not how many cases are currently open (E1), but how many
have newly started in the last X days.
Trend: is a surge of new submissions currently building up?
"""

import numpy as np


def compute(df, case_times, ts_col, window_days):
    # Sorted array of all case start times for fast lookup
    case_starts_sorted = (
        case_times["start_time"]
        .sort_values()
        .values
        .astype("datetime64[ns]")
    )

    window_ns   = np.timedelta64(window_days, "D")
    peer_counts = []

    for ts in df[ts_col]:
        ts_val       = np.datetime64(ts, "ns")
        window_start = ts_val - window_ns

        # Number of cases in the window via binary search
        count = (
            np.searchsorted(case_starts_sorted, ts_val, side="left") -
            np.searchsorted(case_starts_sorted, window_start, side="left")
        )
        peer_counts.append(count)

    df["peer_cases_in_window"] = peer_counts

    return df