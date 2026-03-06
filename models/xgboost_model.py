"""
xgboost_model.py

XGBoost Modell für Remaining Time Prediction.
Gradient Boosting - im Gegensatz zu Random Forest werden die Bäume
sequenziell gebaut, jeder Baum korrigiert die Fehler des vorherigen.
Oft stärker als Random Forest aber rechenintensiver.
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