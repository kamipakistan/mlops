import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

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

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logger.info("Splitting train and test data")

            # Split into features and target variable
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Define models to evaluate
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor(),
                'KNeighborsRegressor': KNeighborsRegressor()
            }

            logger.info("Evaluating all models...")
            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)

            # Determine best model
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            logger.info(f"Best model identified: {best_model_name} with score {best_model_score}")

            if best_model_score < 0.70:
                logger.error("No suitable model found (R2 < 0.70)")
                raise CustomException("No good model found!")

            # Refit best model on full training data before saving
            logger.info("Training the best model on full training data...")
            best_model.fit(X_train, y_train)

            # Save model
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
            logger.info(f"Model saved successfully at {self.model_trainer_config.trained_model_file_path}")

            # Predict & report score
            predictions = best_model.predict(X_test)
            r2 = r2_score(y_test, predictions)
            logger.info(f"Final R2 Score: {r2}")

            return r2

        except Exception as e:
            raise CustomException(e, sys)
