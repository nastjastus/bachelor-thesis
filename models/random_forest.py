"""
random_forest.py

Random Forest model for remaining time prediction.
Ensemble of many decision trees. Each tree learns
slightly different patterns, the prediction is the average of all trees.
More robust than a single tree.
"""

from sklearn.ensemble import RandomForestRegressor


def get_model(rf_params):
    return RandomForestRegressor(**rf_params)