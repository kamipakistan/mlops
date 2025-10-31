import os
import sys
import dill
import pickle
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    """
    Save any Python object to disk using dill/pickle.

    Args:
        file_path (str): Path to save object
        obj (object): Object to be saved
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info(f" Object saved successfully at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load a Python object from disk.

    Args:
        file_path (str): Path to pickle file

    Returns:
        Loaded object
    """
    try:
        with open(file_path, "rb") as file_obj:
            logging.info(f"Loading object from: {file_path}")
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)