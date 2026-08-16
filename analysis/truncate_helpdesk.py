"""
Truncates each Helpdesk case at the LAST 'Resolve ticket' event and
saves the result as data/helpdesk_resolved.xes.

This removes the administrative 30-day auto-close timer from the target
variable, so the remaining time measures only the actual processing up
to resolution.

Rules:
  - Cases without a 'Resolve ticket' event are discarded.
  - With multiple 'Resolve ticket' events, the case is cut at the LAST one
    (inclusive); all events after it are dropped.

Run:  python analysis/truncate_helpdesk.py
"""

import pandas as pd
import pm4py

CASE_COL = "case:concept:name"
ACT_COL  = "concept:name"
TS_COL   = "time:timestamp"

IN_PATH  = "data/helpdesk.xes"
OUT_PATH = "data/helpdesk_resolved.xes"
RESOLVE  = "Resolve ticket"


def main():
    df = pm4py.read_xes(IN_PATH)
    df[TS_COL] = pd.to_datetime(df[TS_COL], utc=True)
    df = df.sort_values([CASE_COL, TS_COL]).reset_index(drop=True)

    n_cases_before = df[CASE_COL].nunique()
    n_events_before = len(df)

    kept_parts = []
    dropped_no_resolve = 0

    for case_id, g in df.groupby(CASE_COL, sort=False):
        g = g.reset_index(drop=True)
        resolve_pos = g.index[g[ACT_COL] == RESOLVE]
        if len(resolve_pos) == 0:
            dropped_no_resolve += 1
            continue
        # LAST resolve event; keep everything up to and including it
        cut = resolve_pos[-1]
        kept_parts.append(g.iloc[: cut + 1])

    out = pd.concat(kept_parts, ignore_index=True)

    n_cases_after = out[CASE_COL].nunique()
    n_events_after = len(out)

    # Sanity check: is resolve now really the last event everywhere?
    last_act = out.sort_values(TS_COL).groupby(CASE_COL)[ACT_COL].last()
    share_resolve_last = (last_act == RESOLVE).mean()

    print("=== Truncation Helpdesk -> Resolve ticket ===")
    print(f"  Cases  before / after: {n_cases_before:,} / {n_cases_after:,}")
    print(f"  Events before / after: {n_events_before:,} / {n_events_after:,}")
    print(f"  discarded (no resolve): {dropped_no_resolve:,}")
    print(f"  share of cases with resolve as last event: {100*share_resolve_last:.1f}%")

    # New case durations for verification (should be much shorter now)
    dur = out.groupby(CASE_COL)[TS_COL].agg(lambda x: (x.max()-x.min()).total_seconds()/86400.0)
    print(f"  case duration (days) min/median/mean/max: "
          f"{dur.min():.2f} / {dur.median():.2f} / {dur.mean():.2f} / {dur.max():.2f}")

    pm4py.write_xes(out, OUT_PATH)
    print(f"\n  Saved: {OUT_PATH}")
    print("  Now add it as its own log in config.py.")


if __name__ == "__main__":
    main()