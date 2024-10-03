import streamlit as st
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

# Check if the model path exists
model_path = "./saved_models/1.keras"
if os.path.exists(model_path):
    try:
        MODEL = tf.keras.models.load_model(model_path)
        # Class names
        CLASS_NAMES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_Target_Spot",
    "Tomato_Tomato_YellowLeaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_healthy"
]
    except Exception as e:
        st.error(f"Error loading the model: {e}")
else:
    st.error("Model file not found. Please check the path.")

# Function to read image file
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

# Streamlit app layout
st.set_page_config(page_title="Tomato Disease Prediction", layout="wide")
st.title("🍃 Tomato Disease Prediction 🍃")
st.write("Upload one or more images of Tomato leaves to check for diseases.")

# File uploader widget for multiple files
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    # Loop through the uploaded files
    for uploaded_file in uploaded_files:
        # Read the uploaded file as bytes
        image_data = uploaded_file.read()
        
        # Convert the image data to a numpy array
        image = read_file_as_image(image_data)
        
        # Display a smaller version of the uploaded image with styling
        st.image(image, caption=f'Uploaded Image: {uploaded_file.name}', width=150, output_format='auto')

        # Prepare the image for prediction
        img_batch = np.expand_dims(image, 0)

        # Predict the class
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        # Display prediction results with enhanced styling
        st.markdown("### Prediction Results", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color: #2E8B57;'>Predicted Class: {predicted_class}</h4>", unsafe_allow_html=True)
        st.markdown(f"<h5 style='color: #FF4500;'>Confidence: {confidence:.2%}</h5>", unsafe_allow_html=True)

# Sidebar for tips
st.sidebar.header("🌱 Tips for Better Predictions")
st.sidebar.write(
    """
    - Ensure the leaf is clearly visible and centered in the image.
    - Use well-lit conditions to avoid shadows.
    - The model is trained on specific diseases;
    - ensure the leaf corresponds to these categories.
    """
)

# Footer
st.markdown("---")
st.markdown("<small style='text-align: center;'>Developed with ❤️ by Ambigapathi</small>", unsafe_allow_html=True)

