"""Distribution of the last activity per case. Usage: python analysis/inspect_ends.py sepsis"""
import os
import sys
import pandas as pd
import pm4py

# Allow importing config.py from the project root when run from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LOGS, CASE_COL, ACT_COL, TS_COL

name = sys.argv[1]
df = pm4py.read_xes(LOGS[name]["path"])
df[TS_COL] = pd.to_datetime(df[TS_COL], utc=True).dt.tz_localize(None)
df = df.sort_values([CASE_COL, TS_COL])

last = df.groupby(CASE_COL)[ACT_COL].last()
print(f"\n{name}: {len(last):,} cases\n")
for act, n in last.value_counts().items():
    print(f"{n:7,}  {100*n/len(last):5.1f}%   {act}")

if name == "sepsis":
    ends_return = last[last == "Return ER"].index
    has_release = (
        df[df[CASE_COL].isin(ends_return)]
          .groupby(CASE_COL)[ACT_COL]
          .apply(lambda s: s.str.startswith("Release").any())
    )
    print(f"\nReturn ER cases with a Release before: {has_release.sum()} / {len(has_release)}")

if name == "production":
    # Does Packing ever occur without being the last activity?
    has_packing = df.groupby(CASE_COL)[ACT_COL].apply(lambda s: (s == "Packing").any())
    ends_packing = last == "Packing"
    print(f"\nCases with Packing anywhere:    {has_packing.sum()}")
    print(f"Cases ending with Packing:      {ends_packing.sum()}")

    # Do Final Inspection endings have a Packing before them?
    fi = last[last == "Final Inspection Q.C."].index
    print(f"Final Inspection endings with Packing before: "
          f"{has_packing[fi].sum()} / {len(fi)}")

if name == "domestic":
    for act in ["Declaration REJECTED by EMPLOYEE",
                "Declaration REJECTED by ADMINISTRATION",
                "Declaration REJECTED by MISSING",
                "Declaration SAVED by EMPLOYEE"]:
        n_total = df.loc[df[ACT_COL] == act, CASE_COL].nunique()
        n_last = (last == act).sum()
        if n_total:
            print(f"{act:42s} in {n_total:5d} cases, of which {n_last:5d} as last event "
                  f"({100*n_last/n_total:4.0f}%)")

if name == "bpic2012_all":
    terminals = ["A_APPROVED", "A_DECLINED", "A_CANCELLED"]
    has_term = df.groupby(CASE_COL)[ACT_COL].apply(lambda s: s.isin(terminals).any())
    print(f"\nCases with an A-terminal state anywhere: {has_term.sum()} / {len(has_term)} "
          f"({100*has_term.mean():.0f}%)")