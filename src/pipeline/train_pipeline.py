import os
import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

# Initialize logger for this specific module
logger = logging.getLogger(__name__)

def run_training_pipeline():
    try:
        logger.info("Starting ML Pipeline")

        # Data Ingestion
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        logger.info("Data Ingestion completed")

        # Data Transformation
        transformer = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(
            train_path, test_path
        )
        logger.info(f"Data Transformation completed. Preprocessor saved at {preprocessor_path}")

        # Model Training + Baseline Selection
        trainer = ModelTrainer()
        best_score, best_model_name = trainer.train_baseline_and_get_best(train_arr, test_arr)
        logger.info(f"Best Baseline Model: {best_model_name} | Score: {best_score}")

        # Hyperparameter Tuning for Best Model
        tuned_score, tuned_model_name = trainer.tune_and_train_best_model(
            train_arr, test_arr, best_model_name
        )
        logger.info(f"Best Tuned Model: {tuned_model_name} | Score: {tuned_score}")

        logger.info("ML Pipeline Finished Successfully")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_training_pipeline()
