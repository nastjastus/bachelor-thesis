import pandas as pd
import pm4py

# Load raw log
df = pm4py.read_xes("data/financial_log.xes")
df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True).dt.tz_localize(None)

print("Before:", len(df), "events,", df["case:concept:name"].nunique(), "cases")

# Keep only W-activities
df = df[df["concept:name"].str.startswith("W_")]

# Keep only COMPLETE events
df = df[df["lifecycle:transition"].str.upper() == "COMPLETE"]

# Sort by case and time
df = df.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)

print("After:", len(df), "events,", df["case:concept:name"].nunique(), "cases")

# Save as XES
event_log = pm4py.convert_to_event_log(df)
pm4py.write_xes(event_log, "data/bpic2012_w.xes")
print("Saved as data/bpic2012_w.xes")