import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

# Initialize logger for this specific module
logger = logging.getLogger(__name__)


@dataclass
class DataTransformationConfig:
    """
    Configuration for storing the preprocessor object path.
    The processed object will be saved to the artifacts directory.
    """
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    """
    Responsible for creating preprocessing pipelines,
    applying transformations to training & test datasets,
    and saving the preprocessing object for later use.
    """

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates and returns a column transformer object
        combining numeric & categorical preprocessing pipelines.

        Returns:
            ColumnTransformer: A preprocessing object that handles numeric and categorical features.
        """
        try:
            # Define numerical and categorical features
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Pipeline for numerical columns: handle missing values & scale features
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="median")),  # Fill missing numerical values
                    ('scaler', StandardScaler())  # Scale numerical values
                ]
            )

            # Pipeline for categorical columns: handle missing values, encode, and scale
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="most_frequent")), # Fill missing categorical values
                    ('one_hot_encoder', OneHotEncoder()),  # One-hot encode categorical variables
                    ('scaler', StandardScaler(with_mean=False))  # Scale encoded categorical values
                ]
            )

            # Combine both pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            logger.info("Preprocessing pipelines created successfully.")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Reads training and testing data, applies the preprocessing pipeline,
        and returns transformed arrays along with the preprocessor file path.

        Args:
            train_path (str): Path to the training dataset
            test_path (str): Path to the testing dataset

        Returns:
            tuple: (train_array, test_array, preprocessor_path)
        """
        try:
            logger.info("Reading training and testing datasets.")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info("Dataset loaded successfully.")
            logger.info("Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"

            # Separate input & target features for training set
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate input & target features for testing set
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logger.info("Applying preprocessing object on training and testing data.")

            # Fit on training data and transform both train & test
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine input features and target variable into final arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logger.info(f"Preprocessing completed and saved at {self.data_transformation_config.preprocessor_obj_file_path}")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
