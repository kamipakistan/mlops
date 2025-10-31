import os
import sys
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import r2_score
from src.models import get_regression_models, regression_param_grids
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from sklearn.model_selection import GridSearchCV
logger = logging.getLogger(__name__)


@dataclass
class ModelTrainerConfig:
    """Configuration for saving the trained model"""
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    """
    Trains multiple ML models, evaluates them, selects the best one,
    and saves the trained model.
    """

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

        # Baseline models
        self.models = get_regression_models()

        # Hyperparameters for tuning only best model later
        self.param_grids = regression_param_grids


    def train_baseline_and_get_best(self, train_arr, test_arr):
        try:
            logging.info("Training baseline models...")

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            best_model = None
            best_model_name = ""
            best_score = -np.inf

            for name, model in self.models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                score = r2_score(y_test, preds)

                logging.info(f"{name} R2 Score = {score}")

                if score > best_score:
                    best_score = score
                    best_model = model
                    best_model_name = name

            logging.info(f"Best Baseline Model: {best_model_name} with score {best_score}")

            # Save baseline best model temporarily
            save_object("artifacts/best_baseline_model.pkl", best_model)

            return best_score, best_model_name

        except Exception as e:
            raise CustomException(e, sys)


    def tune_and_train_best_model(self, train_arr, test_arr, best_model_name):
        try:
            logging.info(f"Hyperparameter tuning for {best_model_name}...")

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            model = self.models[best_model_name]
            param_grid = self.param_grids[best_model_name]

            # If no params, skip tuning
            if not param_grid:
                logging.info("No hyperparameters to tune, using baseline model.")
                best_params_model = model

            else:
                grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1)
                grid.fit(X_train, y_train)

                best_params_model = grid.best_estimator_
                logging.info(f"Best params for {best_model_name}: {grid.best_params_}")

            # Retrain on full dataset with best model
            best_params_model.fit(X_train, y_train)
            preds = best_params_model.predict(X_test)
            tuned_score = r2_score(y_test, preds)

            logging.info(f"Tuned Model Score: {tuned_score}")

            # Save final tuned model
            save_object(self.model_trainer_config.trained_model_file_path, best_params_model)

            return tuned_score, best_model_name

        except Exception as e:
            raise CustomException(e, sys)
