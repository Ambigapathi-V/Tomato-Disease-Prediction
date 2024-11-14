from dataclasses import dataclass
from src.Tomato_disease_prediction.logger import logger
from pathlib import Path
from src.Tomato_disease_prediction.exception import CustomException

from src.Tomato_disease_prediction.constants import *
from src.Tomato_disease_prediction.utils.common import read_yaml, create_directories
from src.Tomato_disease_prediction.entity.config_entity import DataIngestionConfig,ModelTrainerConfig

class ConfigurationManager:
    def __init__(self,
                 config_filepath=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH,
                 schema_filepath=SCHEMA_FILE_PATH):
        # Read the configuration, params, and schema files
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        
        # Create directories if they don't exist
        create_directories([self.config['artifacts_root']])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        # Load data ingestion configuration
        config = self.config['data_ingestion']
        create_directories([config['root_dir']])
        
        # Create and return DataIngestionConfig object
        data_ingestion_config = DataIngestionConfig(
            input_path=Path(config['input_path']),
            output_path=Path(config['output_path']),
            image_size=config['image_size'],
            batch_size=config['batch_size']
        )
        
        return data_ingestion_config
    
    def model_trainer_config(self)->ModelTrainerConfig:
        # Load model trainer configuration
        config = self.config['model_trainer']
        create_directories([config['root_dir']])

        
        # Create and return ModelTrainer object
        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config['root_dir']),
            model_name=config['model_name'],
            output_path=Path(config['output_path']),
            input_path=Path(config['input_path']),
            image_size=config['image_size'],
            batch_size=config['batch_size'],
            epochs=config['epochs'],
            channels=config['channels']
        )
        
        return model_trainer_config
    
