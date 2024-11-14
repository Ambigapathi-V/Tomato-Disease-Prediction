from dataclasses import dataclass
from src.Tomato_disease_prediction.logger import logger
from pathlib import Path
from src.Tomato_disease_prediction.exception import CustomException

@dataclass
class DataIngestionConfig:
    input_path: Path
    output_path: Path
    image_size : int
    batch_size: int

@dataclass
class ModelTrainerConfig:
    root_dir : Path
    model_name : str
    input_path : Path
    output_path : Path
    image_size : int
    batch_size : int
    epochs : int
    channels: int
    

    