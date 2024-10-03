# Tomato Disease Prediction Using Deep Learning 🍅

## Overview

This project aims to develop a deep learning model that predicts diseases affecting tomato crops. By leveraging image processing and classification algorithms, this model helps farmers identify and manage diseases effectively, ultimately reducing crop losses and improving food security.

## Project Highlights

- **Disease Categories**: The model predicts the following diseases:
  - Tomato Bacterial Spot
  - Tomato Early Blight
  - Tomato Late Blight
  - Tomato Leaf Mold
  - Tomato Septoria Leaf Spot
  - Tomato Spider Mites (Two-Spotted Spider Mites)
  - Tomato Target Spot
  - Tomato Yellow Leaf Curl Virus
  - Tomato Mosaic Virus
  - Tomato Healthy

## Key Steps

1. **Data Collection**: A comprehensive dataset of over 10,000 images of tomato plants was collected and labeled for various diseases.

2. **Data Preprocessing**: 
   - **Data Augmentation**: Techniques such as rotation, zooming, and flipping were applied to enhance the model's performance.
   - **Normalization**: Pixel values were normalized to a range of 0 to 1.
   - **Dataset Splitting**: The dataset was divided into training (70%), validation (15%), and test sets (15%).

3. **Model Building**:
   - A custom Convolutional Neural Network (CNN) was designed and implemented.
   - Achieved an accuracy of 95% on the validation set.

4. **Hyperparameter Tuning**: Techniques like grid search and random search were used to optimize model parameters.

5. **Model Evaluation**: Performance metrics such as confusion matrix, precision, recall, and ROC curve were employed to assess model effectiveness.

6. **Deployment**: The model was deployed as an interactive web application using Streamlit, allowing users to upload images and receive instant predictions.

## Installation

To run this project locally, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/Ambigapathi-V/Tomato-Disease-Prediction.git

2. Navigate to the project directory:
   ```bash
   cd Tomato-Disease-Prediction

3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   
4. Run the Streamlit app:
   ```bash
   streamlit run app.py

## Usage

1. Open the Streamlit application in your web browser.
2. Upload an image of a tomato plant.
3. Click on the "Predict" button to receive an instant prediction about the presence of diseases.

## Acknowledgments

Special thanks to Dhruve Patel and the Codebasics team for their invaluable support and guidance throughout this project.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Explore the Project

- **Website**: [Tomato Disease Prediction App](https://ambigapathi-v-tomato-disease-prediction-app-iteodd.streamlit.app/)
- **LinkedIn**: [Ambigapathi V](https://www.linkedin.com/in/ambigapathi-v)


