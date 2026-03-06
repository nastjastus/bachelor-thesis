"""
e2_resource_load.py

Encoding 2: resource_load_at_time

Misst den Ressourcen-Wettbewerb zum Zeitpunkt eines Events.
Nicht wie viele Cases insgesamt laufen, sondern wie viele Cases
gerade denselben Bearbeiter blockieren. Direkter Konkurrenzdruck
auf eine spezifische Ressource.
"""

import pandas as pd


def compute(df, case_times, case_col, ts_col, res_col, exclude_resources):
    # Hilfsvariablen
    case_end_dict = case_times.set_index(case_col)["end_time"].to_dict()
    last_res      = {}
    res_counts    = {}
    resource_load_values = []

    for _, row in df.iterrows():
        case_id      = row[case_col]
        current_res  = row[res_col]
        current_time = row[ts_col]

        # Abgeschlossene Cases aufräumen
        finished = [c for c, r in list(last_res.items())
                    if case_end_dict[c] < current_time]
        for c in finished:
            r = last_res.pop(c)
            res_counts[r] = res_counts.get(r, 1) - 1
            if res_counts[r] <= 0:
                res_counts.pop(r, None)

        # Ressourcenwechsel behandeln
        prev_res = last_res.get(case_id)
        if prev_res is not None and prev_res != current_res:
            res_counts[prev_res] = res_counts.get(prev_res, 1) - 1
            if res_counts[prev_res] <= 0:
                res_counts.pop(prev_res, None)

        # Aktuelles Event zählen
        if current_res not in exclude_resources:
            last_res[case_id] = current_res
            res_counts[current_res] = res_counts.get(current_res, 0) + 1
            resource_load_values.append(res_counts.get(current_res, 0))
        else:
            resource_load_values.append(0)

    df["resource_load_at_time"] = resource_load_values

    return df