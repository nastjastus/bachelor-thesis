"""
evaluate.py

Trainiert alle Modelle mit allen Konfigurationen und
berechnet RMSE und MAE für Train- und Testdaten.
Ergebnisse werden als CSV in results/ gespeichert.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path


def train_and_evaluate(df_train, df_test, base_features, target_col, configs, models):
    # Ergebnisse für alle Modelle und Konfigurationen sammeln
    all_results = {}

    for model_name, model in models.items():
        print(f"\nModell: {model_name}")
        print("-" * 40)

        # für jedes Modell neu gestartet 
        results_summary    = [] 
        baseline_rmse_test = None
        baseline_mae_test = None

        for config_name, extra_features in configs:
            feature_cols = base_features + extra_features

            X_train = df_train[feature_cols].values
            y_train = df_train[target_col].values
            X_test  = df_test[feature_cols].values
            y_test  = df_test[target_col].values

            # Modell trainieren
            # clone damit jede Konfiguration ein frisches Modell bekommt
            from sklearn.base import clone
            m = clone(model)
            m.fit(X_train, y_train)

            # Fehler berechnen
            y_pred_train = m.predict(X_train)
            rmse_train   = np.sqrt(mean_squared_error(y_train, y_pred_train)) / 3600
            mae_train    = mean_absolute_error(y_train, y_pred_train) / 3600

            y_pred_test = m.predict(X_test)
            rmse_test   = np.sqrt(mean_squared_error(y_test, y_pred_test)) / 3600
            mae_test    = mean_absolute_error(y_test, y_pred_test) / 3600

            # RMSE Delta zur Baseline berechnen
            if baseline_rmse_test is None:
                baseline_rmse_test = rmse_test
                delta_rmse_str = "—"
            else:
                delta_rmse     = rmse_test - baseline_rmse_test
                delta_rmse_str = f"{delta_rmse:+.2f} h" # "+" damit vorzeichen angezeigt wird ".2f" = zwei Dezimalstellen

            # MAE Delta zur Baseline berechnen
            if baseline_mae_test is None:
                baseline_mae_test = mae_test
                delta_mae_str = "—"
            else:
                delta_mae     = mae_test - baseline_mae_test
                delta_mae_str = f"{delta_mae:+.2f} h" # "+" damit vorzeichen angezeigt wird ".2f" = zwei Dezimalstellen

    

            results_summary.append({
                "Modell":           model_name,
                "Konfiguration":    config_name,
                "Features (n)":     len(feature_cols),
                "Train RMSE (h)":   round(rmse_train, 2),
                "Train MAE (h)":    round(mae_train,  2),
                "Test RMSE (h)":    round(rmse_test,  2),
                "Test MAE (h)":     round(mae_test,   2),
                "Delta Test RMSE":  delta_rmse_str,
                "Delta Test MAE":   delta_mae_str,
            })

            #print(f"  {config_name:35s} Test RMSE: {rmse_test:.2f} h  |  Delta: {delta_str}")
            total = len(configs)
            done  = len(results_summary)
            print(f"\r  Fortschritt: {done}/{total}", end="", flush=True)

        print()
        all_results[model_name] = pd.DataFrame(results_summary)

    return all_results


def print_results(all_results):
    # Ergebnisse für jedes Modell ausgeben
    for model_name, results_df in all_results.items():
        print(f"\n{model_name}")
        print("=" * 60)
        print(results_df.to_string(index=False))


def save_results(all_results):
    # Einzelne CSVs pro Modell speichern
    Path("results").mkdir(exist_ok=True)

    for model_name, results_df in all_results.items():
        filename = f"results/{model_name.lower().replace(' ', '_')}_results.csv"
        results_df.to_csv(filename, index=False)
        print(f"  Gespeichert: {filename}")

    # Alle Ergebnisse zusammen in einer CSV
    combined = pd.concat(all_results.values(), ignore_index=True)
    combined.to_csv("results/all_results.csv", index=False)
    print("  Gespeichert: results/all_results.csv")