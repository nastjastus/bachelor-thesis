"""
xgboost_model.py

XGBoost model for remaining time prediction.
Gradient boosting; unlike Random Forest, the trees are built
sequentially, each tree correcting the errors of the previous one.
Often stronger than Random Forest but more compute-intensive.
"""

import xgboost as xgb


def get_model(random_state):
    return xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=random_state,
        verbosity=0,
    )