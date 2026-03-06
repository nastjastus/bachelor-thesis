"""
e6_batching.py

Encoding 6: batch_indicator + batch_size

Misst ob ein Case gerade in einem Batch-Wartezustand ist.
Nicht die allgemeine Auslastung, sondern ob der Case auf einen
zeitgesteuerten Trigger wartet der ihn blockiert.
Strukturelle Verzögerung statt Ressourcen-Verzögerung.

batch_size      - Anzahl anderer Events derselben Activity
                  im Zeitfenster um das aktuelle Event
batch_indicator - 1 wenn batch_size > 0, sonst 0
"""

import numpy as np


def compute(df, act_col, ts_col, batch_window_min):
    batch_window_ns = np.timedelta64(batch_window_min * 60, "s")

    # Pro Activity sortiertes Array aller Timestamps
    activity_ts = {}
    for act in df[act_col].unique():
        ts_sorted      = df[df[act_col] == act][ts_col].values.astype("datetime64[ns]")
        activity_ts[act] = np.sort(ts_sorted)

    batch_size_values      = []
    batch_indicator_values = []

    for _, row in df.iterrows():
        current_time = np.datetime64(row[ts_col], "ns")
        current_act  = row[act_col]

        ts_arr    = activity_ts[current_act]

        # Alle Events derselben Activity im Fenster [t-5min, t+5min]
        idx_start = np.searchsorted(ts_arr, current_time - batch_window_ns, side="left")
        idx_end   = np.searchsorted(ts_arr, current_time + batch_window_ns, side="right")

        # -1 weil das Event selbst nicht mitgezählt wird
        batch_size = (idx_end - idx_start) - 1

        batch_size_values.append(batch_size)
        batch_indicator_values.append(1 if batch_size > 0 else 0)

    df["batch_size"]      = batch_size_values
    df["batch_indicator"] = batch_indicator_values

    return df