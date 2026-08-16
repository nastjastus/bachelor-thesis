import pandas as pd
import pm4py

# Load CSV
# df = pd.read_csv("data/finale.csv")  # uncomment for Helpdesk
df = pd.read_csv("data/Production_Data.csv")

# Rename the relevant columns to the XES standard
df = df.rename(columns={
    "Case ID": "case:concept:name",
    "Activity": "concept:name",
    "Resource": "org:resource",
    "Complete Timestamp": "time:timestamp",
})

# Convert timestamp to a real datetime
df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], format="%Y/%m/%d %H:%M:%S.%f")

# Sort by case and time
df = df.sort_values(["case:concept:name", "time:timestamp"])

# Convert to XES and save
event_log = pm4py.convert_to_event_log(df)
# pm4py.write_xes(event_log, "data/helpdesk.xes")  # uncomment for Helpdesk
pm4py.write_xes(event_log, "data/production.xes")

# print("Done, saved as data/helpdesk.xes")  # uncomment for Helpdesk
print("Done, saved as data/production.xes")
print(f"Cases: {df['case:concept:name'].nunique()}, Events: {len(df)}")