import os
import sys
import tensorflow as tf
from src.Tomato_disease_prediction.logger import logger
from src.Tomato_disease_prediction.exception import CustomException
from src.Tomato_disease_prediction.utils.common import create_directories, get_data
from src.Tomato_disease_prediction.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.input_path = config.input_path
        self.output_path = config.output_path
        self.image_size = config.image_size
        self.batch_size = config.batch_size
        
    def load_data(self):
        try:
            # Load dataset using the get_data function
            dataset = get_data(image_size=self.image_size, batch_size=self.batch_size, input_path=self.input_path)
            logger.info("Data loading completed.")
            
            # Create output directory if it doesn't exist
            create_directories([self.output_path])
            
            # Document the input path in a text file
            self._document_input_path()

            # Create class folders in the output directory
            self._create_class_folders(dataset.class_names)

            # Save images in their respective class folders
            self._save_images(dataset)
            
            logger.info("All images have been saved in respective folders.")
            return dataset
        
        except Exception as e:
            logger.error(f"Error occurred during data ingestion: {e}")
            raise CustomException(e, sys)

    def _document_input_path(self):
        """Writes the input path to a text file."""
        try:
            with open(os.path.join(self.output_path, "input_path.txt"), 'w') as file:
                file.write(str(self.input_path))
            logger.info("Input path documented.")
        except Exception as e:
            logger.error(f"Error documenting input path: {e}")
            raise CustomException(e, sys)

    def _create_class_folders(self, class_names):
        """Creates class folders in the output directory."""
        try:
            for class_name in class_names:
                class_folder = os.path.join(self.output_path, class_name)
                os.makedirs(class_folder, exist_ok=True)
                logger.info(f"Created folder for class: {class_name}")
        except Exception as e:
            logger.error(f"Error creating class folders: {e}")
            raise CustomException(e, sys)

    def _save_images(self, dataset):
        """Saves images in their respective class folders, skipping if already exists."""
        try:
            for batch_index, (images, labels) in enumerate(dataset):
                for i in range(len(images)):
                    class_label = dataset.class_names[labels[i]]
                    class_folder = os.path.join(self.output_path, class_label)
                    file_path = os.path.join(class_folder, f"image_{batch_index}_{i}.jpg")
                    
                    # Check if the image already exists
                    if not os.path.exists(file_path):
                        tf.keras.preprocessing.image.save_img(file_path, images[i].numpy().astype("uint8"))
                        logger.info(f"Saved image to {file_path}")
                    else:
                        logger.info(f"Image already exists at {file_path}, skipping.")
        except Exception as e:
            logger.error(f"Error saving images: {e}")
            raise CustomException(e, sys)
