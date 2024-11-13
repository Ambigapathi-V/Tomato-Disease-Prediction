![](https://github.com/pragyy/datascience-readme-template/blob/main/Headerheader.jpg)

# Tomato Disease Prediction

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/pragyy/datascience-readme-template?include_prereleases)
![GitHub last commit](https://img.shields.io/github/last-commit/pragyy/datascience-readme-template)
![GitHub pull requests](https://img.shields.io/github/issues-pr/pragyy/datascience-readme-template)
![GitHub](https://img.shields.io/github/license/pragyy/datascience-readme-template)
![contributors](https://img.shields.io/github/contributors/pragyy/datascience-readme-template) 
![codesize](https://img.shields.io/github/languages/code-size/pragyy/datascience-readme-template) 



## Project Overview

The **Tomato Disease Prediction project** is a machine learning solution aimed at transforming the way tomato plant diseases are identified and managed. Built on a deep learning framework, this project leverages advanced image classification techniques to detect diseases in tomato plants, helping farmers, agronomists, and agricultural advisors to address crop health issues early on. Early detection and accurate diagnosis play a crucial role in preventing the spread of diseases, minimizing crop loss, and improving agricultural productivity.

## Objectives
- **Accurate and Reliable Disease Detection:** Employ machine learning to classify and diagnose various diseases affecting tomato plants from leaf images, ensuring high precision in real-world agricultural settings.
- **Accessible Solution for Farmers and Agricultural Professionals:** Create a simple yet powerful tool that can be used by non-technical users in rural and agricultural settings, enabling faster and more effective decision-making.
- **Scalability for Broader Applications:** Design the model to be adaptable, allowing it to be retrained and scaled to detect diseases in other crops, making it versatile across agricultural domains.
## Features

1. **Automated Disease Detection:** Uses a CNN model to classify tomato leaf images as healthy or diseased, enabling quick and accurate diagnosis.
2. **High Accuracy:** Achieves reliable results through model optimization and data augmentation, ensuring robust performance in varied conditions.
3. **Simple User Interface:** Provides an easy-to-use interface where users can upload images and get instant predictions with confidence scores.
4. **Agricultural Impact:** Supports farmers in managing crop health proactively, helping to minimize crop loss and maximize yield.


## Demo

[App link](https://github.com/Ambigapathi-V/Tomato-Disease-Prediction/tree/main/app)


## Screenshots

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)

# Installation and Setup

**1.Clone the Repository:**

```bash
  git clone https://github.com/Ambigapathi-V/Tomato-Disease-Prediction
  cd Credit-Risk-Model
```
**2.Set Up a Virtual Environment:**

   ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**4. Run the main.py:**
```bash
python main.py
```

**4. Run the Streamlit**
```
streamlit run app.py
```


I like to structure it as below - 
## Codes and Resources Used
- **Editor Used:**   Visual Studio Code
- **Python Version:** Python 3.10.0

## Python packages Used

This section provides a list of dependencies required to replicate the **Tomato Disease Prediction project.** Categorizing packages by their purpose helps users install what they need and understand each package's role in the project.

## 📦 Python Packages Used

This section provides a list of dependencies required to replicate the **Tomato Disease Prediction** project. Categorizing packages by their purpose helps users install what they need and understand each package's role in the project.

- **General Purpose**: Commonly used packages for handling basic operations and internet resources:
  - `urllib`, `os`, `requests`

- **Data Manipulation**: Essential libraries for data handling, transformation, and preparation:
  - `pandas` - For data wrangling and manipulation.
  - `numpy` - For numerical operations and array handling.

- **Data Visualization**: Used to generate plots and visualizations to analyze data trends and model performance:
  - `matplotlib` - For creating static, animated, and interactive plots.
  - `seaborn` - Built on top of `matplotlib`, used for advanced visualizations.

- **Machine Learning**: Packages required for model development, training, and evaluation:
  - `scikit-learn` - For machine learning algorithms, evaluation metrics, and preprocessing utilities.
  - `tensorflow` - For building and training deep learning models.
  
- **Others**: Additional packages as needed for specific functions like statistical analysis, image handling, or interactive dashboards:
  - `scipy` - For scientific computations.
  - `PIL` - For handling and processing images.
  - `streamlit` - For building an interactive web application to deploy the model.

Feel free to add more categories, such as **Data Preparation** or **Statistical Analysis**, depending on the requirements and complexity of the project.


## 📊 Data

Data is the foundation of any data science project. Below is an overview of the datasets used in the **Tomato Disease Prediction** project, including links to the original data sources, data descriptions, and details of preprocessing steps applied.

### Data Sources
List all data sources, each with a brief description and a link for reference:
- **Primary Dataset**: [Tomato Disease Image Dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)  
  This dataset contains labeled images of healthy and diseased tomato leaves across different disease types, which is used for model training and evaluation.


### Data Description
Provide a brief description of the dataset, including:
- **Features/Columns**: List the key attributes, such as image label, disease type, or metadata if available.
- **Target Variable**: Describe the target variable, such as the disease type or binary classification problem.



## Data Ingestion

Images are collected and stored in a structured format, typically organized by disease type. This structure facilitates easy access and processing.


## Data Preprocessing

- **Image Resizing:**
All images are resized to a fixed dimension (e.g., 128x128 pixels). This ensures consistency in input size across the dataset and speeds up model training.

- **Data Normalization:**
Image pixel values are normalized to a range suitable for neural network training. Standard normalization is performed by subtracting the mean and dividing by the standard deviation of the dataset. This helps the model converge faster during training.

- **Data Augmentation:**
To improve the generalization of the model and prevent overfitting, data augmentation techniques such as rotation, zoom, and flipping can be applied. This increases the diversity of the training data.


- **Splitting the Data:**
The processed data is split into training and testing datasets. Typically, 80% of the data is used for training, and 20% is used for testing. This ensures that the model is trained on a majority of the data but also evaluated on unseen data.
## Code Structure

Explain the code structure and how it is organized, including any significant files and their purposes. This will help others understand how to navigate your project and find specific components. 

Here is the basic suggested skeleton for your data science repo (you can structure your repository as needed ):

```bash
├── data
│   ├── data1.csv
│   ├── data2.csv
│   ├── cleanedData
│   │   ├── cleaneddata1.csv
|   |   └── cleaneddata2.csv
├── data_ingestion.py
├── data_preprocessing.py
├── model_training.py
├── app.py
├── requirements.txt
├── main.py
├── Dockerfile
├── LICENSE
├── README.md
└── .gitignore
```

## Result And Evaluation


1. **Accuracy**:  
   Accuracy is the percentage of correctly predicted instances among all predictions. It is the most straightforward evaluation metric.

2. **Precision**:  
   Precision indicates the percentage of true positive predictions out of all positive predictions made by the model. It helps to evaluate the model’s ability to avoid false positives.

3. **Recall**:  
   Recall represents the percentage of true positive predictions out of all actual positive instances. It helps to evaluate the model’s ability to correctly identify positive cases.

4. **F1-Score**:  
   F1-Score is the harmonic mean of precision and recall. It provides a balance between the two, especially when dealing with imbalanced classes.

5. **Confusion Matrix**:  
   The confusion matrix is used to visualize the performance of a classification model. It shows the true positives, false positives, true negatives, and false negatives, helping to understand where the model is making mistakes.

6. **Classification Report**:  
   The classification report provides a summary of the precision, recall, F1-score, and support for each class. It is a comprehensive evaluation tool for multi-class classification problems.

   


## Future Work

### 1. **Model Improvement**
- Explore advanced deep learning models like CNNs, ResNet, or EfficientNet for better accuracy.
- Implement ensemble learning (e.g., Random Forest, XGBoost) for improved performance.
- Leverage transfer learning to fine-tune pre-trained models on large datasets.

### 2. **Data Augmentation and Synthetic Data**
- Apply data augmentation techniques (e.g., rotation, flipping, zooming) to improve generalization.
- Generate synthetic images using GANs to enhance the dataset and model robustness.

### 3. **Integration with IoT Devices**
- Implement real-time disease detection using IoT devices (e.g., cameras) in agricultural fields.
- Develop a mobile app for farmers to get disease predictions from images taken with their phones.

### 4. **Model Explainability**
- Use explainable AI techniques (e.g., SHAP, LIME) to make the model's predictions more interpretable.
- Create heatmaps to show which parts of the plant image contributed to disease classification.

### 5. **Scalability and Deployment**
- Deploy the model to the cloud for better scalability and easier access.
- Implement edge deployment (e.g., Raspberry Pi) for on-site, real-time disease detection.

### 6. **Multi-Class Classification and Multi-Modal Data**
- Expand the model to classify a broader range of diseases and plant types.
- Integrate additional data sources (e.g., environmental factors, soil conditions) to improve accuracy.

### 7. **Model Monitoring and Continuous Improvement**
- Set up performance monitoring to track accuracy and other metrics over time.
- Periodically retrain the model with new data to maintain and improve its performance.

## Deployment

To deploy this project run

```bash
  npm run deploy
```



    

## Acknowledgments

- **Dataset**: Special thanks to [Source of the Dataset] for providing the tomato disease dataset, which served as the foundation for this project.
- **Libraries and Tools**: We acknowledge the use of open-source libraries such as [scikit-learn](https://scikit-learn.org/), [TensorFlow](https://www.tensorflow.org/), [Keras](https://keras.io/), and [OpenCV](https://opencv.org/) for model building and data processing.
- **Contributors**: A big thank you to everyone who contributed to the development of this project, including code reviewers and testers who helped refine the model and its functionalities.
- **Mentors**: Thanks to mentors and experts who provided valuable insights and guidance throughout the development of this project.

This project would not have been possible without the support and resources provided by the open-source community and various individuals. We are grateful for your contributions.

## License

Specify the license under which your code is released. Moreover, provide the licenses associated with the dataset you are using. This is important for others to know if they want to use or contribute to your project. 

For this github repository, the License used is [MIT License](https://opensource.org/license/mit/).