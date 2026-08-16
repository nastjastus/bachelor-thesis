"""
e4_avg_delay.py

Encoding 4: avg_delay_in_window

Measures the process speed compared to the normal state.
Not how many cases are running, but whether the system is currently
slower or faster than historically usual.
Quality instead of quantity.

Important: must be computed after the train/test split,
because the historical average may only be computed from
training data.
"""

import numpy as np


def compute(df, df_train, ts_col, window_days):
    # Compute the historical average from training data only
    # prefix_len > 1 because the first event of a case is always 0
    train_delays     = df_train.loc[
        df_train["prefix_len"] > 1, "time_since_last_event_s"
    ].values
    global_avg_delay = np.mean(train_delays)

    # Fast computation via searchsorted
    ts_array     = df[ts_col].values.astype("datetime64[ns]")
    delay_array  = df["time_since_last_event_s"].values
    prefix_array = df["prefix_len"].values
    window_ns    = np.timedelta64(window_days, "D")

    avg_delay_values = []
    for i, ts in enumerate(ts_array):
        window_start = ts - window_ns

        idx_end   = np.searchsorted(ts_array, ts, side="left")
        idx_start = np.searchsorted(ts_array, window_start, side="left")

        window_delays = delay_array[idx_start:idx_end]
        window_prefix = prefix_array[idx_start:idx_end]
        valid         = window_delays[window_prefix > 1]

        if len(valid) == 0:
            avg_delay_values.append(0.0)
        else:
            avg_delay_values.append(np.mean(valid) - global_avg_delay)

    df["avg_delay_in_window"] = avg_delay_values

    return df, global_avg_delay