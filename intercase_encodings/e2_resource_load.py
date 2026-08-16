"""
e2_resource_load.py

Encoding 2: resource_load_at_time

Measures the resource competition at the time of an event.
Not how many cases are running in total, but how many cases are
currently occupying the same resource. Direct competitive pressure
on a specific resource.

A case occupies at most one resource at any time. SCHEDULE events
(without a resource) and excluded resources count as occupying nothing.
"""

import heapq
import pandas as pd


def _is_missing(res):
    """SCHEDULE events have no resource. Depending on the export path,
    pm4py/XES writes NaN or the string 'None' or 'nan' here."""
    return pd.isna(res) or str(res).strip() in {"None", "nan", "", "NaN"}


def compute(df, case_times, case_col, ts_col, res_col, exclude_resources):
    case_end_dict = dict(zip(
        case_times[case_col].values,
        case_times["end_time"].values.astype("datetime64[ns]"),
    ))

    held = {}    # case_id -> currently occupied resource
    counts = {}  # resource -> number of cases currently occupying it
    pending = []  # min-heap (end_time, case_id) for release
    scheduled = set()
    resource_load_values = []

    def release(case_id):
        r = held.pop(case_id, None)
        if r is None:
            return
        counts[r] -= 1
        if counts[r] <= 0:
            counts.pop(r, None)

    def acquire(case_id, r):
        held[case_id] = r
        counts[r] = counts.get(r, 0) + 1
        if case_id not in scheduled:
            heapq.heappush(pending, (case_end_dict[case_id], case_id))
            scheduled.add(case_id)

    case_arr = df[case_col].values
    res_arr = df[res_col].values
    ts_arr = df[ts_col].values.astype("datetime64[ns]")

    for case_id, current_res, current_time in zip(case_arr, res_arr, ts_arr):
        # 1. Finished cases release their resource
        while pending and pending[0][0] < current_time:
            release(heapq.heappop(pending)[1])

        # 2. Events without a resource (SCHEDULE) or excluded resources:
        #    the case occupies nothing at this moment.
        if _is_missing(current_res) or current_res in exclude_resources:
            release(case_id)
            resource_load_values.append(0)
            continue

        # 3. Rebook only on a real change. If the case already holds the
        #    resource (e.g. START followed by COMPLETE), do NOT count again.
        if held.get(case_id) != current_res:
            release(case_id)
            acquire(case_id, current_res)

        resource_load_values.append(counts[current_res])

    # Check the invariant: each held case is counted exactly once
    assert sum(counts.values()) == len(held), (
        f"inconsistent bookkeeping: {sum(counts.values())} counts "
        f"on {len(held)} occupying cases"
    )

    df["resource_load_at_time"] = resource_load_values

    return df