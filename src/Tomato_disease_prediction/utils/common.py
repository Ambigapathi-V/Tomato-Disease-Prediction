import os
import sys
import yaml
from src.Tomato_disease_prediction.logger import logger
from src.Tomato_disease_prediction.exception import CustomException
import joblib
import pandas as pd
from ensure import ensure_annotations
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,r2_score
from sklearn.model_selection import GridSearchCV
from box import ConfigBox
from pathlib import Path
from typing import List, Dict
from box.exceptions import BoxValueError
import numpy as np
import tensorflow as tf


@ensure_annotations
def read_yaml(path_to_yaml : Path) -> ConfigBox:
    """
    Reads a yaml file and returns a ConfigBox object.

    Args:
        path_to_yaml (Path): The path to the yaml file.
    
    Raises:
        ValueError: if yaml file is empty or malformed
    
    Returns:
        ConfigBox: A ConfigBox object containing the parsed data.
    """
    try:
        with open(path_to_yaml) as yamlfile:
            content = yaml.safe_load(yamlfile)
            logger.info(f"Successfully loaded yaml file: {path_to_yaml}")
            return ConfigBox(content)
    except BoxValueError as e:
        logger.error(f"Error reading yaml file: {path_to_yaml} - {str(e)}")
        raise ValueError(f"Error reading yaml file: {path_to_yaml}")
    except Exception as e:
        logger.error(f"Error reading yaml file: {path_to_yaml} - {str(e)}")
        raise CustomException(e,sys)
        
        

@ensure_annotations
def create_directories (path_to_directories: list , verbose = True):
    """
    Creates directories in the given path if they don't exist.
    
    Args:
        path_to_directories (list): A list of directories to be created.
        verbose (bool): A flag to indicate whether to print info messages.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory: {path}")

@ensure_annotations
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            joblib.dump(obj, file_obj)
            
    except Exception as e:
        logger.error(f"Error saving object: {file_path} - {str(e)}")
        raise CustomException(e, sys)
    
def load_object(file_path:str) -> object:
    '''
    Load object from file
    
    file_path: Str location where the object needs to be loaded
    
    Returns
    Object loaded from file
    
    '''
    try:
        logger.info(f'Loading object from {file_path}')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File {file_path} does not exist')
        with open(file_path, 'rb') as file_obj:
            print(file_obj)
            return joblib.load(file_obj)
    
    except Exception as e:
        raise CustomException(e, sys) from e
    
def get_data(input_path, image_size, batch_size):
    try:
        # Load dataset
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            input_path,
            shuffle=True,
            image_size=(image_size, image_size),
            batch_size=batch_size,
        )
        logger.info("Data loading completed.")
        return dataset
    except Exception as e:
        logger.error(f"Error occurred while loading data: {str(e)}")
        raise

    