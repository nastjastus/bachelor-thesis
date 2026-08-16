"""
evaluate.py

Trains all models with all configurations and
computes RMSE and MAE for train and test data.
Results are saved as CSV in results/.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path


def train_and_evaluate(df_train, df_test, base_features, target_col, configs, models):
    # Collect results for all models and configurations
    all_results = {}

    for model_name, model in models.items():
        print(f"\nModel: {model_name}")
        print("-" * 40)

        # reset for each model
        results_summary    = []
        baseline_rmse_test = None
        baseline_mae_test = None

        for config_name, extra_features in configs:
            feature_cols = base_features + extra_features

            X_train = df_train[feature_cols].values
            y_train = df_train[target_col].values
            X_test  = df_test[feature_cols].values
            y_test  = df_test[target_col].values

            # Train model
            # clone so each configuration gets a fresh model
            from sklearn.base import clone
            m = clone(model)
            m.fit(X_train, y_train)

            # Compute errors
            y_pred_train = m.predict(X_train)
            rmse_train   = np.sqrt(mean_squared_error(y_train, y_pred_train)) / 3600
            mae_train    = mean_absolute_error(y_train, y_pred_train) / 3600

            y_pred_test = m.predict(X_test)
            rmse_test   = np.sqrt(mean_squared_error(y_test, y_pred_test)) / 3600
            mae_test    = mean_absolute_error(y_test, y_pred_test) / 3600

            # Compute RMSE delta vs. baseline
            if baseline_rmse_test is None:
                baseline_rmse_test = rmse_test
                delta_rmse_str = "—"
            else:
                delta_rmse     = rmse_test - baseline_rmse_test
                delta_rmse_str = f"{delta_rmse:+.2f} h"  # "+" forces the sign, ".2f" = two decimals

            # Compute MAE delta vs. baseline
            if baseline_mae_test is None:
                baseline_mae_test = mae_test
                delta_mae_str = "—"
            else:
                delta_mae     = mae_test - baseline_mae_test
                delta_mae_str = f"{delta_mae:+.2f} h"  # "+" forces the sign, ".2f" = two decimals

            results_summary.append({
                "Model":            model_name,
                "Configuration":    config_name,
                "Features (n)":     len(feature_cols),
                "Train RMSE (h)":   round(rmse_train, 2),
                "Train MAE (h)":    round(mae_train,  2),
                "Test RMSE (h)":    round(rmse_test,  2),
                "Test MAE (h)":     round(mae_test,   2),
                "Delta Test RMSE":  delta_rmse_str,
                "Delta Test MAE":   delta_mae_str,
            })

            total = len(configs)
            done  = len(results_summary)
            print(f"\r  Progress: {done}/{total}", end="", flush=True)

        print()
        all_results[model_name] = pd.DataFrame(results_summary)

    return all_results


def print_results(all_results):
    # Print results for each model
    for model_name, results_df in all_results.items():
        print(f"\n{model_name}")
        print("=" * 60)
        print(results_df.to_string(index=False))


def save_results(all_results, result_dir="results"):
    # Create folder, including nested paths
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    for model_name, results_df in all_results.items():
        filename = f"{result_dir}/{model_name.lower().replace(' ', '_')}_results.csv"
        results_df.to_csv(filename, index=False)
        print(f"  Saved: {filename}")

    # All results combined in one CSV
    combined = pd.concat(all_results.values(), ignore_index=True)
    combined.to_csv(f"{result_dir}/all_results.csv", index=False)
    print(f"  Saved: {result_dir}/all_results.csv")