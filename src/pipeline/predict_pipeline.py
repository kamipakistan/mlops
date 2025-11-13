import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

# Initialize module-level logger
logger = logging.getLogger(__name__)


class PredictPipeline:
    def __init__(self):
        logger.info("Initializing PredictPipeline instance.")

    def predict(self, features: pd.DataFrame):
        """
        Run prediction pipeline using saved model and preprocessor.
        Includes extensive quality and information logs.
        """
        try:
            logger.info("Starting prediction process.")

            # Verify input type
            if not isinstance(features, pd.DataFrame):
                logger.warning(f"Expected features as pandas DataFrame, got {type(features)} instead.")
                raise ValueError("Input features must be a pandas DataFrame.")

            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            logger.info(f"Model path: {model_path}")
            logger.info(f"Preprocessor path: {preprocessor_path}")

            # Check if files exist
            if not os.path.exists(model_path):
                logger.error(f"Model file not found at path: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not os.path.exists(preprocessor_path):
                logger.error(f"Preprocessor file not found at path: {preprocessor_path}")
                raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")

            logger.info("Loading model and preprocessor objects...")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            logger.info("Model and preprocessor loaded successfully.")

            # Log feature info
            logger.info(f"Input features shape: {features.shape}")
            logger.debug(f"Input feature columns: {features.columns.tolist()}")
            logger.debug(f"First few rows of input data:\n{features.head()}")

            # Data Transformation
            logger.info("Transforming input features using preprocessor...")
            data_scaled = preprocessor.transform(features)
            logger.info(f"Data transformed successfully. Shape: {data_scaled.shape}")

            # Prediction
            logger.info("Performing predictions...")
            predictions = model.predict(data_scaled)
            logger.info(f"Predictions completed. Number of samples predicted: {len(predictions)}")

            # Log summary stats for prediction
            logger.debug(f"Predictions preview: {predictions[:5]}")

            return predictions

        except Exception as e:
            logger.exception("Exception occurred during prediction pipeline execution.")
            raise CustomException(e, sys)


class CustomData:
    """
    Represents custom input data for prediction.
    Provides a method to convert raw input into DataFrame.
    """

    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int
    ):
        logger.info("Initializing CustomData instance.")
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self) -> pd.DataFrame:
        """
        Convert the stored attributes into a pandas DataFrame.
        Includes quality validation and logging.
        """
        try:
            logger.info("Converting CustomData to DataFrame.")

            # Data validation checks
            if not isinstance(self.reading_score, (int, float)):
                logger.warning("Reading score should be numeric.")
            if not isinstance(self.writing_score, (int, float)):
                logger.warning("Writing score should be numeric.")

            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            df = pd.DataFrame(custom_data_input_dict)
            logger.info(f"DataFrame created successfully with shape {df.shape}.")
            logger.debug(f"DataFrame preview:\n{df.head()}")

            return df

        except Exception as e:
            logger.exception("Exception occurred while creating DataFrame from CustomData.")
            raise CustomException(e, sys)
