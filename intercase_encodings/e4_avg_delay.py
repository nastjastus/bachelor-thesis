"""
e4_avg_delay.py

Encoding 4: avg_delay_in_window

Misst die Prozessgeschwindigkeit im Vergleich zum Normalzustand.
Nicht wie viele Cases laufen, sondern ob das System gerade
langsamer oder schneller ist als historisch üblich.
Qualität statt Quantität.

Wichtig: muss nach dem Train/Test Split berechnet werden,
weil der historische Durchschnitt nur aus Trainingsdaten
berechnet werden darf.
"""

import numpy as np


def compute(df, df_train, ts_col, window_days):
    # Historischen Durchschnitt nur aus Trainingsdaten berechnen
    # prefix_len > 1 weil erstes Event eines Cases immer 0 hat
    train_delays     = df_train.loc[
        df_train["prefix_len"] > 1, "time_since_last_event_s"
    ].values
    global_avg_delay = np.mean(train_delays)

    # Schnelle Berechnung mit searchsorted
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