from src.Tomato_disease_prediction.config.configuration import ConfigurationManager
from src.Tomato_disease_prediction.components.model_trainer import ModelTrainer
from src.Tomato_disease_prediction.logger import logger
from src.Tomato_disease_prediction.exception import CustomException
import os
import sys

STAGE_NAME = 'Data Ingestion Stage'

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass
    
    def initiate_model_trainer(self):
        
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.model_trainer_config()
            data_ingestion = ModelTrainer(data_ingestion_config)
            data_ingestion.training_model()
            
        except CustomException as e:
            logger.error(f"Error occurred during data ingestion: {e}")
            return CustomException(e,sys)
        


if __name__ == "__main__":
    try: 
        pipeline = ModelTrainerTrainingPipeline()
        logger.info(f">>>>>>>>>>>Starting {STAGE_NAME}<<<<<<<<<<<<<")
        pipeline.initiate_model_trainer()
        logger.info(f">>>>>>>>>>>Completed {STAGE_NAME}<<<<<<<<<<<<<")
        
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise CustomException(e, sys)

    