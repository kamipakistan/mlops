"""
models.py
---------
Contains registry of regression models and hyperparameter search grids
for automated model selection and tuning in MLOps pipeline.
"""

# --------------------------- Imports --------------------------- #
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, ExtraTreesRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

# Optional models – import only if installed
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None


# --------------------------- Model Registry --------------------------- #
def get_regression_models():
    """
    Return dictionary of base regression models for evaluation.
    Note: More models = more compute time.
    """

    models = {
        # Linear models
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "ElasticNet": ElasticNet(),

        # Tree-based models
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Extra Trees": ExtraTreesRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "AdaBoost": AdaBoostRegressor(),

        # Distance based
        "KNN Regressor": KNeighborsRegressor(),

        # Kernel models
        "SVR (RBF)": SVR(kernel="rbf"),
        "Kernel Ridge": KernelRidge()
    }

    # Add only if installed — helps avoid dependency errors
    if XGBRegressor:
        models["XGBoost"] = XGBRegressor(objective="reg:squarederror", verbosity=0)

    if CatBoostRegressor:
        models["CatBoost"] = CatBoostRegressor(verbose=False)

    return models


# ----------------------- Hyperparameter Grids ----------------------- #
regression_param_grids = {
    # Linear models
    "Linear Regression": {},  # baseline model
    "Ridge Regression": {"alpha": [0.01, 0.1, 1.0, 10, 100]},
    "Lasso Regression": {"alpha": [0.001, 0.01, 0.1, 1, 10]},
    "ElasticNet": {
        "alpha": [0.01, 0.1, 1, 5],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    },

    # Tree models
    "Decision Tree": {
        "max_depth": [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    },
    "Extra Trees": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 4],
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 1],
    },

    # Distance + Kernel models
    "KNN Regressor": {
        "n_neighbors": [3, 5, 7, 10],
        "weights": ["uniform", "distance"],
    },
    "SVR (RBF)": {
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto"],
    },
    "Kernel Ridge": {
        "alpha": [0.01, 0.1, 1],
        "kernel": ["rbf", "linear", "polynomial"],
    },

    # Optional models
    "XGBoost": {
        "n_estimators": [200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.9, 1.0],
    } if XGBRegressor else {},

    "CatBoost": {
        "iterations": [300, 500],
        "depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
    } if CatBoostRegressor else {},
}