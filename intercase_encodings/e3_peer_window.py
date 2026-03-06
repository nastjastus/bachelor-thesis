"""
e3_peer_window.py

Encoding 3: peer_cases_in_window

Misst die Ankunftsrate neuer Cases im Zeitfenster.
Nicht wie viele Cases gerade offen sind (E1), sondern wie viele
in den letzten X Tagen neu gestartet haben.
Trend: Läuft gerade eine Hochphase an neuen Einreichungen?
"""

import numpy as np


def compute(df, case_times, ts_col, window_days):
    # Sortiertes Array aller Case-Startzeitpunkte für schnelle Suche
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

        # Anzahl Cases im Fenster mit binärer Suche
        count = (
            np.searchsorted(case_starts_sorted, ts_val, side="left") -
            np.searchsorted(case_starts_sorted, window_start, side="left")
        )
        peer_counts.append(count)

    df["peer_cases_in_window"] = peer_counts

    return df