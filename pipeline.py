"""
pipeline.py

Hauptskript der Pipeline.
Lädt den Log, berechnet Features, trainiert Modelle und
speichert die Ergebnisse. Alle Parameter werden aus config.py gelesen.

Ausführen: python pipeline.py
"""

import pandas as pd
import pm4py
from itertools import combinations

from config import (
    LOG_PATH, CASE_COL, ACT_COL, TS_COL, RES_COL,
    TRAIN_RATIO, RANDOM_STATE, WINDOW_DAYS,
    BATCH_WINDOW_MIN, EXCLUDE_RESOURCES, RF_PARAMS
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

# 1. Log laden

print("=" * 60)
print("1. LOG LADEN")
print("=" * 60)

df = pm4py.read_xes(str(LOG_PATH))
df = df.sort_values([CASE_COL, TS_COL]).reset_index(drop=True) # sortieren nach Case-ID dann Zeitstempel
df[TS_COL] = pd.to_datetime(df[TS_COL], utc=True).dt.tz_localize(None) # Zeitstempel berenigen 

print(f"Events:     {len(df):,}") 
print(f"Cases:      {df[CASE_COL].nunique():,}")
print(f"Activities: {df[ACT_COL].nunique()}")
print(f"Zeitraum:   {df[TS_COL].min().date()} → {df[TS_COL].max().date()}")

# 2. Remaining Time berechnen

print("\n" + "=" * 60)
print("2. REMAINING TIME (TARGET VARIABLE)")
print("=" * 60)

# Hier wird case_end_time als neue Spalte an df gejoint

case_end = df.groupby(CASE_COL)[TS_COL].max().rename("case_end_time")
df       = df.join(case_end, on=CASE_COL)
df["remaining_time_s"] = (df["case_end_time"] - df[TS_COL]).dt.total_seconds()
df["remaining_time_h"] = df["remaining_time_s"] / 3600

print(f"Remaining Time (h) – min:  {df['remaining_time_h'].min():.1f}")
print(f"Remaining Time (h) – mean: {df['remaining_time_h'].mean():.1f}")
print(f"Remaining Time (h) – max:  {df['remaining_time_h'].max():.1f}")

# --- nur Statistik aus Interesse: durchschnittliche Case Duration ---
case_duration = (
    df.groupby(CASE_COL)[TS_COL]
      .agg(start="min", end="max")
)

duration_h = (
    (case_duration["end"] - case_duration["start"])
    .dt.total_seconds() / 3600
)

print("\n[INFO] Case Duration Statistik:")
print(f"min:  {duration_h.min():.1f} h")
print(f"mean: {duration_h.mean():.1f} h")
print(f"max:  {duration_h.max():.1f} h")

# 3. Intra-Case Features

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

BASE_FEATURES = [
    "prefix_len",
    "elapsed_time_s",
    "time_since_last_event_s",
    "hour_of_day",
    "day_of_week",
] + act_feature_cols

print(f"Basis Features: {len(BASE_FEATURES)} (inkl. {len(act_feature_cols)} act-dummies)")

# 4. Inter-Case Encodings

print("\n" + "=" * 60)
print("4. INTER-CASE ENCODINGS BERECHNEN")
print("=" * 60)

df = df.sort_values(TS_COL).reset_index(drop=True)

# Hilfstabelle mit Start- und Endzeit jedes Cases
case_times = (
    df.groupby(CASE_COL)[TS_COL]
      .agg(start_time="min", end_time="max")
      .reset_index()
)

print("\nBerechne Encoding 1: open_cases_at_time ...")
df = e1_open_cases.compute(df, case_times, TS_COL)
print(f"  min / mean / max: {df['open_cases_at_time'].min()} / "
      f"{round(df['open_cases_at_time'].mean(), 2)} / "
      f"{df['open_cases_at_time'].max()}")

print("\nBerechne Encoding 2: resource_load_at_time ...")
df = e2_resource_load.compute(df, case_times, CASE_COL, TS_COL, RES_COL, EXCLUDE_RESOURCES)
print(f"  min / mean / max: {df['resource_load_at_time'].min()} / "
      f"{round(df['resource_load_at_time'].mean(), 2)} / "
      f"{df['resource_load_at_time'].max()}")

print("\nBerechne Encoding 3: peer_cases_in_window ...")
df = e3_peer_window.compute(df, case_times, TS_COL, WINDOW_DAYS)
print(f"  min / mean / max: {df['peer_cases_in_window'].min()} / "
      f"{round(df['peer_cases_in_window'].mean(), 2)} / "
      f"{df['peer_cases_in_window'].max()}")

print("\nBerechne Encoding 5: queue_length_at_activity ...")
df = e5_queue_length.compute(df, case_times, CASE_COL, ACT_COL, TS_COL)
print(f"  min / mean / max: {df['queue_length_at_activity'].min()} / "
      f"{round(df['queue_length_at_activity'].mean(), 2)} / "
      f"{df['queue_length_at_activity'].max()}")

print("\nBerechne Encoding 6: batch_indicator + batch_size ...")
df = e6_batching.compute(df, ACT_COL, TS_COL, BATCH_WINDOW_MIN)
print(f"  batch_size      – min / mean / max: {df['batch_size'].min()} / "
      f"{round(df['batch_size'].mean(), 2)} / {df['batch_size'].max()}")
print(f"  batch_indicator – Anteil Batches: "
      f"{round(df['batch_indicator'].mean() * 100, 1)}% der Events")

# 5. Train/Test Split

print("\n" + "=" * 60)
print("5. TRAIN / TEST SPLIT (TEMPORAL)")
print("=" * 60)

case_order = (
    df.groupby(CASE_COL)["case_start_time"]
      .min()
      .sort_values()
      .reset_index()
)
case_order.columns = [CASE_COL, "case_start_time"] # Absicherung dass die erst Spalte CASE_COL bennant bleibt

n_cases     = len(case_order) # zählt gesamt Anzahl
n_train     = int(n_cases * TRAIN_RATIO) # berechnet wie vile Cases ins Trainingsset kommen 
train_cases = set(case_order.iloc[:n_train][CASE_COL]) # alle cases VOR n_train
test_cases  = set(case_order.iloc[n_train:][CASE_COL]) # alle danach

df_train = df[df[CASE_COL].isin(train_cases)].copy()
df_test  = df[df[CASE_COL].isin(test_cases)].copy()

print(f"Train Cases: {len(train_cases):,}  ({len(df_train):,} Events)")
print(f"Test Cases:  {len(test_cases):,}  ({len(df_test):,} Events)")
print(f"Train Zeitraum: {df_train[TS_COL].min().date()} → {df_train[TS_COL].max().date()}")
print(f"Test  Zeitraum: {df_test[TS_COL].min().date()}  → {df_test[TS_COL].max().date()}")

# E4 nach dem Split berechnen
print("\nBerechne Encoding 4: avg_delay_in_window ...")
df, global_avg_delay = e4_avg_delay.compute(df, df_train, TS_COL, WINDOW_DAYS)
df_train["avg_delay_in_window"] = df["avg_delay_in_window"].loc[df_train.index]
df_test["avg_delay_in_window"]  = df["avg_delay_in_window"].loc[df_test.index]
print(f"  Historischer Durchschnitt (Train): {global_avg_delay/3600:.2f} h")
print(f"  min / mean / max: "
      f"{round(df['avg_delay_in_window'].min()/3600, 2)} / "
      f"{round(df['avg_delay_in_window'].mean()/3600, 2)} / "
      f"{round(df['avg_delay_in_window'].max()/3600, 2)} h")

df.to_csv("results/debug_df.csv", index=False)

# 6. Modelle definieren

print("\n" + "=" * 60)
print("6. MODELLE")
print("=" * 60)

MODELS = {
    "Random Forest": random_forest.get_model(RF_PARAMS),
    "CART":          cart.get_model(RANDOM_STATE),
    "XGBoost":       xgboost_model.get_model(RANDOM_STATE),
}

print(f"Modelle: {list(MODELS.keys())}")

# 7. Konfigurationen definieren

TARGET_COL = "remaining_time_s"

print("\n" + "=" * 60)
print("7. KONFIGURATIONEN")
print("=" * 60)

ENCODINGS = [
    ("E1", ["open_cases_at_time"]),
    ("E2", ["resource_load_at_time"]),
    ("E3", ["peer_cases_in_window"]),
    ("E4", ["avg_delay_in_window"]),
    ("E5", ["queue_length_at_activity"]),
    ("E6", ["batch_indicator", "batch_size"]),
]

CONFIGS = [("Baseline", [])]

for r in range(1, len(ENCODINGS) + 1):
    for combo in combinations(ENCODINGS, r):
        label = "+ " + " + ".join(name for name, _ in combo)
        features = [f for _, fs in combo for f in fs]
        CONFIGS.append((label, features))

print(f"Konfigurationen: {len(CONFIGS)} (Baseline + alle Kombinationen aus E1–E6)")

# 8. Evaluation

print("\n" + "=" * 60)
print("8. EVALUATION ALLER KONFIGURATIONEN")
print("=" * 60)

all_results = train_and_evaluate(
    df_train, df_test, BASE_FEATURES, TARGET_COL,
    CONFIGS, MODELS
)

# 9. Ergebnisse ausgeben und speichern

print("\n" + "=" * 60)
print("9. ERGEBNISSE")
print("=" * 60)

print_results(all_results)
save_results(all_results)

print("\n✓ Pipeline fertig!")
print("  Neuen Log testen: LOG_PATH in config.py ändern")
print("  Neues Modell hinzufügen: models/ Ordner und MODELS dict in pipeline.py")