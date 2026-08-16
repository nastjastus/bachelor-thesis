"""
e5_queue_length.py

Encoding 5: queue_length_at_activity

Measures the activity-specific backlog at the time of an event.
How many cases are currently active and have not yet reached this
activity? Bottleneck detection at the activity level instead of
globally.

Note: for the first activity of a process this encoding is always 0,
because no active case can have not yet reached it.
"""

import numpy as np
import pandas as pd


def compute(df, case_times, case_col, act_col, ts_col):
    # For each case: when did it first see which activity
    first_occurrence = (
        df.groupby([case_col, act_col])[ts_col]
          .min()
          .reset_index()
          .rename(columns={ts_col: "first_seen"})
    )

    case_ids = case_times[case_col].values
    starts = case_times["start_time"].values.astype("datetime64[ns]")
    ends = case_times["end_time"].values.astype("datetime64[ns]")

    ts_all = df[ts_col].values.astype("datetime64[ns]")
    act_all = df[act_col].values

    queue_values = np.zeros(len(df), dtype=np.int64)

    for act in pd.unique(act_all):
        # +1 when the case starts, -1 when it reaches the activity.
        # If it never reaches it, then -1 at the case end.
        fs_map = (
            first_occurrence.loc[first_occurrence[act_col] == act]
                            .set_index(case_col)["first_seen"]
        )
        minus = pd.to_datetime(pd.Series(case_ids).map(fs_map)).values
        minus = np.where(np.isnat(minus), ends, minus)

        times = np.concatenate([starts, minus])
        deltas = np.concatenate([
            np.ones(len(starts), dtype=np.int64),
            -np.ones(len(minus), dtype=np.int64),
        ])

        order = np.argsort(times, kind="stable")
        times = times[order]
        queue_after = np.cumsum(deltas[order])

        # Lookup: last timeline entry STRICTLY before the event.
        # side="left" minus 1 excludes all simultaneous entries, so the
        # result is independent of the ordering for identical timestamps.
        mask = act_all == act
        idx = np.searchsorted(times, ts_all[mask], side="left") - 1
        queue_values[mask] = np.where(idx < 0, 0, queue_after[np.maximum(idx, 0)])

    df["queue_length_at_activity"] = queue_values

    return df