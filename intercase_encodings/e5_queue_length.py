"""
e5_queue_length.py

Encoding 5: queue_length_at_activity

Misst den activity-spezifischen Stau zum Zeitpunkt eines Events.
Wie viele Cases sind gerade aktiv und haben diese Activity
noch nicht erreicht? Engpass-Erkennung auf Activity-Ebene
statt global.
"""

import pandas as pd
import numpy as np


def compute(df, case_times, case_col, act_col, ts_col):
    # Für jeden Case speichern wann er welche Activity
    # zum ersten Mal gesehen hat
    first_occurrence = (
        df.groupby([case_col, act_col])[ts_col]
          .min()
          .reset_index()
          .rename(columns={ts_col: "first_seen"})
    )

    # Pro Activity eine Timeline bauen
    # +1 wenn Case startet, -1 wenn Case die Activity erreicht oder endet
    activities         = df[act_col].unique()
    activity_timelines = {}

    for act in activities:
        act_cases       = first_occurrence[first_occurrence[act_col] == act]
        first_seen_map  = act_cases.set_index(case_col)["first_seen"].to_dict()

        events = []
        for _, ct_row in case_times.iterrows():
            c = ct_row[case_col]
            events.append((ct_row["start_time"], +1))
            if c in first_seen_map:
                events.append((first_seen_map[c], -1))
            else:
                events.append((ct_row["end_time"], -1))

        events_df = pd.DataFrame(events, columns=["time", "delta"])
        events_df = events_df.sort_values("time").reset_index(drop=True)
        events_df["queue_after"] = events_df["delta"].cumsum()
        activity_timelines[act]  = events_df

    # Für jeden Event den Wert aus der passenden Timeline ablesen
    queue_values = []
    for _, row in df.iterrows():
        current_time = np.datetime64(row[ts_col], "ns")
        current_act  = row[act_col]

        timeline = activity_timelines[current_act]
        ts_arr   = timeline["time"].values.astype("datetime64[ns]")

        idx = np.searchsorted(ts_arr, current_time, side="left") - 1
        if idx < 0:
            queue_values.append(0)
        else:
            queue_values.append(int(timeline["queue_after"].iloc[idx]))

    df["queue_length_at_activity"] = queue_values

    return df