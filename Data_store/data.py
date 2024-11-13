import os
import logging
from dagshub import get_repo_bucket_client

# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG,  # Set log level to DEBUG for detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format with timestamp
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler("upload_log.log")  # Log to a file
    ]
)

# Create a logger
logger = logging.getLogger()

# Initialize the S3 client for the DagsHub repository
s3 = get_repo_bucket_client("Ambigapathi-V/Tomato-Disease-Prediction")

# Function to upload a folder and its contents to S3
def upload_folder_to_s3(local_folder, s3_bucket, s3_folder):
    logger.info(f"Starting upload from {local_folder} to {s3_folder}...")
    
    # Traverse the directory structure
    for root, dirs, files in os.walk(local_folder):
        logger.debug(f"Looking in directory: {root}")
        
        for file in files:
            local_file_path = os.path.join(root, file)  # Full local path of the file
            relative_path = os.path.relpath(local_file_path, local_folder)  # Relative path from local_folder
            s3_key = os.path.join(s3_folder, relative_path).replace("\\", "/")  # Ensure paths are in S3 format

            # Debugging: Print the file being uploaded
            logger.debug(f"Preparing to upload file: {local_file_path} -> {s3_key}")

            try:
                # Upload the file to S3
                s3.upload_file(Bucket=s3_bucket, Filename=local_file_path, Key=s3_key)
                logger.info(f"Successfully uploaded: {local_file_path} -> {s3_key}")
            except Exception as e:
                logger.error(f"Failed to upload {local_file_path} -> {s3_key}. Error: {e}")

# Define the local folder and S3 folder
local_folder = "data/tomato/"  # Local path where your folders and images are stored
s3_bucket = "Tomato-Disease-Prediction"  # S3 bucket name
s3_folder = "tomato_disease_images"  # Desired path in the S3 bucket

# Upload the folder structure to S3
upload_folder_to_s3(local_folder, s3_bucket, s3_folder)
