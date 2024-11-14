import os
import sys
import tensorflow as tf
import mlflow
import dagshub
import matplotlib.pyplot as plt
from src.Tomato_disease_prediction.logger import logger
from src.Tomato_disease_prediction.exception import CustomException
from src.Tomato_disease_prediction.utils.common import create_directories, get_data
from src.Tomato_disease_prediction.entity.config_entity import ModelTrainerConfig
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping

# Initialize DagsHub and MLflow
dagshub.init(repo_owner='Ambigapathi-V', repo_name='Tomato-Disease-Prediction', mlflow=True)

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.input_path = config.input_path
        self.output_path = config.output_path
        self.image_size = config.image_size
        self.batch_size = config.batch_size
        self.model_name = config.model_name
        self.channels = config.channels
        self.epochs = config.epochs
        
    def get_dataset_partitions_tf(self, ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
        assert (train_split + test_split + val_split) == 1
        
        ds_size = len(ds)  # Get the total size of the dataset
        
        if shuffle:
            ds = ds.shuffle(shuffle_size, seed=1)  # Shuffle the dataset if needed
        
        train_size = int(train_split * ds_size)  # Calculate the size of the training set
        val_size = int(val_split * ds_size)      # Calculate the size of the validation set
        
        train_ds = ds.take(train_size)  # Create the training dataset
        val_ds = ds.skip(train_size).take(val_size)  # Create the validation dataset
        test_ds = ds.skip(train_size + val_size)     # Create the test dataset
        
        return train_ds, val_ds, test_ds  # Return the datasets
    

    def log_graphs(self, history):
        try:
            # Create and log loss and accuracy graphs
            self.plot_loss(history)
            self.plot_accuracy(history)
            
            # Log them as artifacts in DagsHub
            mlflow.log_artifact('loss_plot.png')
            mlflow.log_artifact('accuracy_plot.png')
            
        except Exception as e:
            logger.error(f"Error occurred while logging graphs: {e}")
            raise CustomException(e, sys)

    def plot_loss(self, history):
        # Plot training and validation loss
        plt.figure(figsize=(8, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss_plot.png')
        plt.close()

    def plot_accuracy(self, history):
        # Plot training and validation accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('accuracy_plot.png')
        plt.close()

    def training_model(self):
        try:
            # Load dataset using the get_data function
            dataset = get_data(image_size=self.image_size, batch_size=self.batch_size, input_path=self.input_path)
            logger.info("Data loading completed.")
            
            # Split dataset into training, validation, and testing sets
            train_ds, val_ds, test_ds = self.get_dataset_partitions_tf(dataset)
            logger.info("Dataset partitioning completed.")
            logger.info(f"Training set size: {len(train_ds)}, Validation set size: {len(val_ds)}, Test set size: {len(test_ds)}")
            
            train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
            val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
            test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
            
            # Resize and rescale images
            resize_and_rescale = tf.keras.Sequential([
                layers.Resizing(self.image_size, self.image_size),
                layers.Rescaling(1./255),
            ])
            
            # Data Augmentation
            data_augmentation = tf.keras.Sequential([
                layers.RandomFlip("horizontal_and_vertical"),
                layers.RandomRotation(0.2),
            ])
            
            # Mapping the images to the model
            train_ds = train_ds.map(
                lambda x, y: (data_augmentation(x, training=True), y)
            ).prefetch(buffer_size=tf.data.AUTOTUNE)
            
            input_shape = (self.image_size, self.image_size, self.channels)
            n_classes = 10  # Adjust based on your classes
            
            # Build the model
            model = models.Sequential([
                layers.Input(shape=input_shape),
                resize_and_rescale,
                layers.Conv2D(32, kernel_size=(3,3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dense(n_classes, activation='softmax'),
            ])
            
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])   
            logger.info("Model compilation completed.")
            logger.info(f"Model summary: {model.summary()}")
            
            # EarlyStopping callback
            early_stopping = EarlyStopping(
                monitor='accuracy',  # Monitor model accuracy
                patience=5,          # Number of epochs to wait for improvement
                restore_best_weights=True  # Restores the weights of the best epoch
            )
            
            # Train the model
            history = model.fit(
                train_ds,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=1,
                validation_data=val_ds,
                callbacks=[early_stopping]
            )
            logger.info("Model training completed.")
            
            # Evaluate the model and log final test results
            scores = model.evaluate(test_ds)
            logger.info(f"Test loss: {scores[0]}, Test accuracy: {scores[1]}")

            # Log graphs (loss and accuracy)
            self.log_graphs(history)
            self.plot_loss(history)
            self.plot_accuracy(history)
            
            model_version = max([int(f.split('.')[0]) for f in os.listdir(self.output_path) if f.split('.')[0].isdigit()] + [0]) + 1
            keras_model_path=model.save(os.path.join(self.output_path, f"{model_version}.keras"))

            base_dir = "../saved_models"
            model_version = max([int(f.split('.')[0]) for f in os.listdir(base_dir) if f.split('.')[0].isdigit()] + [0]) + 1
            keras_model_path=model.save(os.path.join(base_dir, f"{model_version}.keras"))
        except Exception as e:
            logger.error(f"Error occurred while training the model: {e}")
            raise CustomException(e, sys)