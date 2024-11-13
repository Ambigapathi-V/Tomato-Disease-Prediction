from dataclasses import dataclass
from src.Tomato_disease_prediction.logger import logger
from pathlib import Path
from src.Tomato_disease_prediction.exception import CustomException

@dataclass
class DataIngestionConfig:
    input_directory: Path
    output_directory: Path
    