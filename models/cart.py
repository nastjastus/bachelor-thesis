"""
cart.py

CART - Classification and Regression Tree.
A single decision tree. Simpler and more interpretable
than a Random Forest, but more prone to overfitting.
Serves as a simple baseline for the model comparison.
"""

from sklearn.tree import DecisionTreeRegressor


def get_model(random_state):
    return DecisionTreeRegressor(
        max_depth=10,
        min_samples_leaf=20,
        random_state=random_state,
    )