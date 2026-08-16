"""
pipeline.py

Main script of the pipeline.
Loads the log, computes features, trains models and
saves the results. All parameters are read from config.py.

Run: python pipeline.py
"""
import numpy as np
import pandas as pd
import pm4py
from itertools import combinations
from pathlib import Path

from config import (
    LOG_PATH, CASE_COL, ACT_COL, TS_COL, RES_COL,
    TRAIN_RATIO, RANDOM_STATE, WINDOW_DAYS,
    BATCH_WINDOW_MIN, EXCLUDE_RESOURCES, RF_PARAMS, 
    RESULT_DIR, SPLIT_MODE, FILTER_COMPLETE, DROP_CROSSING_CASES, 
    END_ACTIVITIES, END_MODE
)

from intercase_encodings import (
    e1_open_cases,
    e2_resource_load,
    e3_peer_window,
    e4_avg_delay,
    e5_queue_length,
    e6_batching,
)

from models import random_forest, cart, xgboost_model

from evaluate import train_and_evaluate, print_results, save_results

# 1. Load log

print("=" * 60)
print("1. LOAD LOG")
print("=" * 60)

df = pm4py.read_xes(str(LOG_PATH))
df = df.sort_values([CASE_COL, TS_COL]).reset_index(drop=True)  # sort by case ID then timestamp
df[TS_COL] = pd.to_datetime(df[TS_COL], utc=True).dt.tz_localize(None)  # normalize timestamps

print(f"Events:     {len(df):,}")
print(f"Cases:      {df[CASE_COL].nunique():,}")
print(f"Activities: {df[ACT_COL].nunique()}")
print(f"Period:     {df[TS_COL].min().date()} → {df[TS_COL].max().date()}")

# 2. Compute remaining time

print("\n" + "=" * 60)
print("2. REMAINING TIME (TARGET VARIABLE)")
print("=" * 60)

# Join case_end_time onto df as a new column

case_end = df.groupby(CASE_COL)[TS_COL].max().rename("case_end_time")
df       = df.join(case_end, on=CASE_COL)
df["remaining_time_s"] = (df["case_end_time"] - df[TS_COL]).dt.total_seconds()
df["remaining_time_h"] = df["remaining_time_s"] / 3600

print(f"Remaining Time (h) – min:  {df['remaining_time_h'].min():.1f}")
print(f"Remaining Time (h) – mean: {df['remaining_time_h'].mean():.1f}")
print(f"Remaining Time (h) – max:  {df['remaining_time_h'].max():.1f}")

# --- just an informational statistic: average case duration ---
case_duration = (
    df.groupby(CASE_COL)[TS_COL]
      .agg(start="min", end="max")
)

duration_h = (
    (case_duration["end"] - case_duration["start"])
    .dt.total_seconds() / 3600
)

print("\n[INFO] Case duration statistics:")
print(f"min:  {duration_h.min():.1f} h")
print(f"mean: {duration_h.mean():.1f} h")
print(f"max:  {duration_h.max():.1f} h")

t_rel = (df[TS_COL] - df[TS_COL].min()).dt.total_seconds()
censoring = np.corrcoef(t_rel, df["remaining_time_s"])[0, 1]
print(f"\n[DIAGNOSTIC] corr(log position, remaining time): {censoring:+.3f}")

# Utilization: case duration divided by space until end of log
t_end = df[TS_COL].max()
ct = df.groupby(CASE_COL)[TS_COL].agg(start="min", end="max")
duration = (ct["end"] - ct["start"]).dt.total_seconds()
space = (t_end - ct["start"]).dt.total_seconds()
utilization = (duration / space.replace(0, np.nan)).dropna()
print(f"[DIAGNOSTIC] mean utilization (duration / space until end of log): {utilization.mean():.2f}")

log_days = (df[TS_COL].max() - df[TS_COL].min()).total_seconds() / 86400
mean_duration_days = duration_h.mean() / 24
print(f"[DIAGNOSTIC] log length / mean case duration: {log_days/mean_duration_days:.1f}")

# 3. Intra-case features

print("\n" + "=" * 60)
print("3. INTRA-CASE FEATURES")
print("=" * 60)

df["prefix_len"] = df.groupby(CASE_COL).cumcount() + 1

case_start = df.groupby(CASE_COL)[TS_COL].min().rename("case_start_time")
df         = df.join(case_start, on=CASE_COL)
df["elapsed_time_s"] = (df[TS_COL] - df["case_start_time"]).dt.total_seconds()

df["time_since_last_event_s"] = (
    df.groupby(CASE_COL)[TS_COL]
      .diff()
      .dt.total_seconds()
      .fillna(0)
)

act_dummies      = pd.get_dummies(df[ACT_COL], prefix="act").astype(int)
df               = pd.concat([df, act_dummies], axis=1)
act_feature_cols = list(act_dummies.columns)

df["hour_of_day"] = df[TS_COL].dt.hour
df["day_of_week"] = df[TS_COL].dt.dayofweek
df["abs_time_s"]  = (df[TS_COL] - df[TS_COL].min()).dt.total_seconds()

BASE_FEATURES = [
    "prefix_len",
    "elapsed_time_s",
    "time_since_last_event_s",
    "hour_of_day",
    "day_of_week",
] + act_feature_cols

print(f"Base features: {len(BASE_FEATURES)} (incl. {len(act_feature_cols)} act dummies)")

# 4. Inter-case encodings

print("\n" + "=" * 60)
print("4. COMPUTE INTER-CASE ENCODINGS")
print("=" * 60)

df = df.sort_values(TS_COL).reset_index(drop=True)

# Helper table with start and end time of each case
case_times = (
    df.groupby(CASE_COL)[TS_COL]
      .agg(start_time="min", end_time="max")
      .reset_index()
)

print("\nComputing Encoding 1: open_cases_at_time ...")
df = e1_open_cases.compute(df, case_times, TS_COL)
print(f"  min / mean / max: {df['open_cases_at_time'].min()} / "
      f"{round(df['open_cases_at_time'].mean(), 2)} / "
      f"{df['open_cases_at_time'].max()}")

print("\nComputing Encoding 2: resource_load_at_time ...")
df = e2_resource_load.compute(df, case_times, CASE_COL, TS_COL, RES_COL, EXCLUDE_RESOURCES)
print(f"  min / mean / max: {df['resource_load_at_time'].min()} / "
      f"{round(df['resource_load_at_time'].mean(), 2)} / "
      f"{df['resource_load_at_time'].max()}")

print("\nComputing Encoding 3: peer_cases_in_window ...")
df = e3_peer_window.compute(df, case_times, TS_COL, WINDOW_DAYS)
print(f"  min / mean / max: {df['peer_cases_in_window'].min()} / "
      f"{round(df['peer_cases_in_window'].mean(), 2)} / "
      f"{df['peer_cases_in_window'].max()}")

print("\nComputing Encoding 5: queue_length_at_activity ...")
df = e5_queue_length.compute(df, case_times, CASE_COL, ACT_COL, TS_COL)
print(f"  min / mean / max: {df['queue_length_at_activity'].min()} / "
      f"{round(df['queue_length_at_activity'].mean(), 2)} / "
      f"{df['queue_length_at_activity'].max()}")

print("\nComputing Encoding 6: batch_indicator + batch_size ...")
df = e6_batching.compute(df, ACT_COL, TS_COL, CASE_COL, BATCH_WINDOW_MIN)
print(f"  batch_size      – min / mean / max: {df['batch_size'].min()} / "
      f"{round(df['batch_size'].mean(), 2)} / {df['batch_size'].max()}")
print(f"  batch_indicator – share of batches: "
      f"{round(df['batch_indicator'].mean() * 100, 1)}% of events")

print("\n" + "=" * 60)
print(f"5. TRAIN / TEST SPLIT ({SPLIT_MODE})")
print("=" * 60)

case_info = pd.DataFrame({
    "case_start_time": df.groupby(CASE_COL)[TS_COL].min(),
    "case_end_time_":  df.groupby(CASE_COL)[TS_COL].max(),
    "last_act":        df.sort_values(TS_COL).groupby(CASE_COL)[ACT_COL].last(),
}).reset_index()

n_all = len(case_info)
eligible = set(case_info[CASE_COL])

if FILTER_COMPLETE:
    if END_ACTIVITIES is None:
        print("[SPLIT] end_acts=None → filter skipped")
    else:
        if END_MODE == "contains":
            with_term = df[df[ACT_COL].isin(END_ACTIVITIES)][CASE_COL].unique()
            complete = set(with_term)
            print(f"[SPLIT] end_mode=contains")
        else:
            complete = set(case_info.loc[case_info["last_act"].isin(END_ACTIVITIES), CASE_COL])
        print(f"[SPLIT] complete: {len(complete):,} / {n_all:,} "
              f"({100*len(complete)/n_all:.1f}%) — discarded: {n_all-len(complete):,}")
        eligible &= complete

order = case_info[case_info[CASE_COL].isin(eligible)].sort_values("case_start_time")
n_train = int(len(order) * TRAIN_RATIO)

if SPLIT_MODE == "temporal":
    train_cases = set(order.iloc[:n_train][CASE_COL])
    test_cases  = set(order.iloc[n_train:][CASE_COL])

    if DROP_CROSSING_CASES and n_train < len(order):
        split_time = order.iloc[n_train]["case_start_time"]
        crossing = set(order.loc[
            order[CASE_COL].isin(train_cases) & (order["case_end_time_"] >= split_time),
            CASE_COL])
        print(f"[SPLIT] split point: {split_time}")
        print(f"[SPLIT] train cases crossing the split point: {len(crossing):,} / "
              f"{len(train_cases):,} ({100*len(crossing)/len(train_cases):.1f}%) — discarded")
        train_cases -= crossing

elif SPLIT_MODE == "random_case":
    rng = np.random.default_rng(RANDOM_STATE)
    cases = np.array(sorted(eligible))
    rng.shuffle(cases)
    train_cases = set(cases[:n_train])
    test_cases  = set(cases[n_train:])

else:
    raise ValueError(f"unknown SPLIT_MODE: {SPLIT_MODE}")

df_train = df[df[CASE_COL].isin(train_cases)].copy()

print(f"Train Cases: {len(train_cases):,}  ({len(df_train):,} Events)")
print(f"Test Cases:  {len(test_cases):,}")

# E4 after the split, with df_train as history
print("\nComputing Encoding 4: avg_delay_in_window ...")
df, global_avg_delay = e4_avg_delay.compute(df, df_train, TS_COL, WINDOW_DAYS)

# Rebuild df_train/df_test AFTER E4, saves the .loc assignment
df_train = df[df[CASE_COL].isin(train_cases)].copy()
df_test  = df[df[CASE_COL].isin(test_cases)].copy()

print(f"  Historical average (train): {global_avg_delay/3600:.2f} h")
print(f"  min / mean / max: "
      f"{round(df['avg_delay_in_window'].min()/3600, 2)} / "
      f"{round(df['avg_delay_in_window'].mean()/3600, 2)} / "
      f"{round(df['avg_delay_in_window'].max()/3600, 2)} h")

print("\n[DIAGNOSTIC] Feature range train vs. test:")
for f in ["open_cases_at_time", "peer_cases_in_window", "avg_delay_in_window",
          "resource_load_at_time", "queue_length_at_activity", "abs_time_s"]:
    tr, te = df_train[f], df_test[f]
    outside = 100 * ((te < tr.min()) | (te > tr.max())).mean()
    print(f"  {f:26s} train [{tr.min():8.1f}, {tr.max():8.1f}]  "
          f"test [{te.min():8.1f}, {te.max():8.1f}]  outside: {outside:5.1f}%")

Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)
df.to_csv(f"{RESULT_DIR}/debug_df.csv", index=False)


# 6. Define models

print("\n" + "=" * 60)
print("6. MODELS")
print("=" * 60)

MODELS = {
    "Random Forest": random_forest.get_model(RF_PARAMS),
    "CART":          cart.get_model(RANDOM_STATE),
    "XGBoost":       xgboost_model.get_model(RANDOM_STATE),
}

print(f"Models: {list(MODELS.keys())}")

# 7. Define configurations

TARGET_COL = "remaining_time_s"

print("\n" + "=" * 60)
print("7. CONFIGURATIONS")
print("=" * 60)

ENCODINGS = [
    ("E1", ["open_cases_at_time"]),
    ("E2", ["resource_load_at_time"]),
    ("E3", ["peer_cases_in_window"]),
    ("E4", ["avg_delay_in_window"]),
    ("E5", ["queue_length_at_activity"]),
    ("E6", ["batch_indicator", "batch_size"]),
]

CONFIGS = [
    ("Baseline", []),
]

for r in range(1, len(ENCODINGS) + 1):
    for combo in combinations(ENCODINGS, r):
        label = "+ " + " + ".join(name for name, _ in combo)
        features = [f for _, fs in combo for f in fs]
        CONFIGS.append((label, features))

print(f"Configurations: {len(CONFIGS)} (baseline + all combinations of E1–E6)")

# 8. Evaluation

print("\n" + "=" * 60)
print("8. EVALUATION OF ALL CONFIGURATIONS")
print("=" * 60)

all_results = train_and_evaluate(
    df_train, df_test, BASE_FEATURES, TARGET_COL,
    CONFIGS, MODELS
)

# 9. Print and save results

print("\n" + "=" * 60)
print("9. RESULTS")
print("=" * 60)

print_results(all_results)
save_results(all_results, result_dir=RESULT_DIR)

print("\n✓ Pipeline done!")
print("  Test a new log: change LOG_PATH in config.py")
print("  Add a new model: models/ folder and MODELS dict in pipeline.py")