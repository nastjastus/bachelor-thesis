"""
random_forest.py

Random Forest Modell für Remaining Time Prediction.
Ensemble aus vielen Entscheidungsbäumen. Jeder Baum lernt
leicht andere Muster, die Vorhersage ist der Durchschnitt aller Bäume.
Robuster als ein einzelner Baum.
"""

from sklearn.ensemble import RandomForestRegressor


def get_model(rf_params):
    return RandomForestRegressor(**rf_params)