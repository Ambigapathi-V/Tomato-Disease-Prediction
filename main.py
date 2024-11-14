from src.Tomato_disease_prediction.logger import logger
from src.Tomato_disease_prediction.exception import CustomException
from src.Tomato_disease_prediction.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.Tomato_disease_prediction.pipeline.model_trainer_pipeline import ModelTrainerTrainingPipeline
import sys

STAGE_NAME = "Model Trainer Stage"

try: 
        pipeline = ModelTrainerTrainingPipeline()
        logger.info(f">>>>>>>>>>>Starting {STAGE_NAME}<<<<<<<<<<<<<")
        pipeline.initiate_model_trainer()
        logger.info(f">>>>>>>>>>>Completed {STAGE_NAME}<<<<<<<<<<<<<")
        logger.info("-----------------------------------------")
        
except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise CustomException(e, sys)