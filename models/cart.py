"""
cart.py

CART - Classification and Regression Tree.
Ein einzelner Entscheidungsbaum. Einfacher und interpretierbarer
als Random Forest, aber anfälliger für Overfitting.
Dient als einfache Baseline für den Modellvergleich.
"""

from sklearn.tree import DecisionTreeRegressor


def get_model(random_state):
    return DecisionTreeRegressor(
        max_depth=10,
        min_samples_leaf=20,
        random_state=random_state,
    )