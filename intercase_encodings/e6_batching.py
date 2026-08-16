"""
e6_batching.py

Encoding 6: batch_indicator + batch_size

Measures whether a case is currently in a batch waiting state.
Not the general workload, but whether the case is waiting for a
time-driven trigger that blocks it.
Structural delay instead of resource delay.

batch_size      - number of OTHER cases that passed through the same
                  activity within the time window before the current event
batch_indicator - 1 if batch_size > 0, else 0

The window is causal, i.e. [t - window, t], so it only uses information
that already exists at prediction time.
"""

import numpy as np
import pandas as pd


def compute(df, act_col, ts_col, case_col, batch_window_min, causal=True):
    window_ns = np.timedelta64(int(batch_window_min * 60), "s")

    ts_all = df[ts_col].values.astype("datetime64[ns]")
    act_all = df[act_col].values
    case_all = df[case_col].values

    # Per activity: time-sorted arrays of timestamp and case ID
    activity_data = {}
    for act in pd.unique(act_all):
        mask = act_all == act
        ts_act = ts_all[mask]
        case_act = case_all[mask]
        order = np.argsort(ts_act, kind="stable")
        activity_data[act] = (ts_act[order], case_act[order])

    batch_size_values = []
    batch_indicator_values = []

    for current_time, current_act, current_case in zip(ts_all, act_all, case_all):
        ts_arr, case_arr = activity_data[current_act]

        if causal:
            # [t - 2*window, t]: past only
            idx_start = np.searchsorted(ts_arr, current_time - 2 * window_ns, side="left")
            idx_end = np.searchsorted(ts_arr, current_time, side="right")
        else:
            idx_start = np.searchsorted(ts_arr, current_time - window_ns, side="left")
            idx_end = np.searchsorted(ts_arr, current_time + window_ns, side="right")

        peers = case_arr[idx_start:idx_end]
        # Distinct OTHER cases: own lifecycle events do not count
        batch_size = len(np.unique(peers[peers != current_case]))

        batch_size_values.append(batch_size)
        batch_indicator_values.append(1 if batch_size > 0 else 0)

    df["batch_size"] = batch_size_values
    df["batch_indicator"] = batch_indicator_values

    return df