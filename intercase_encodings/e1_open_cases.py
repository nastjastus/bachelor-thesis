"""
e1_open_cases.py

Encoding 1: open_cases_at_time

Misst die globale Systemlast zum exakten Zeitpunkt eines Events.
Wie viele Cases laufen gerade gleichzeitig? Momentaufnahme der Gesamtauslastung.
"""

import pandas as pd
import numpy as np


def compute(df, case_times, ts_col):
    # Für jeden Case: Start = +1, Ende = -1
    starts = case_times[["start_time"]].rename(columns={"start_time": "time"})
    starts["delta"] = 1
    ends = case_times[["end_time"]].rename(columns={"end_time": "time"})
    ends["delta"] = -1

    # Timeline bauen und laufende Summe berechnen
    timeline = (
        pd.concat([starts, ends], ignore_index=True)
          .sort_values("time")
          .reset_index(drop=True)
    )
    timeline["open_cases_after"] = timeline["delta"].cumsum()

    # Wert auf jeden Event übertragen
    df = pd.merge_asof(
        df,
        timeline[["time", "open_cases_after"]],
        left_on=ts_col,
        right_on="time",
        direction="backward",
    )
    df["open_cases_at_time"] = df["open_cases_after"].fillna(0).astype(int)

    return df