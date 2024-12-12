import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# File path for the model (Git LFS will manage the file storage)
trained_model_alpha_path = 'Trained_ModelAlpha.keras'

# Function to download the model if it's not already downloaded (but now it’s handled by Git LFS)
def download_model_if_needed(model_path):
    if not os.path.exists(model_path):
        st.write(f"Downloading model: {model_path}...")
        # Normally here we'd use gdown or another method, but since Git LFS handles it, this can be skipped
    else:
        st.write(f"Model already downloaded: {model_path}")

# Download the model if it's not already downloaded
download_model_if_needed(trained_model_alpha_path)

# Load the model
try:
    trained_model_alpha = tf.keras.models.load_model(trained_model_alpha_path)
except Exception as e:
    st.error(f"Error loading Trained ModelAlpha: {str(e)}")

# Define class labels
class_labels = ['Cloudy', 'Rain', 'Shine', 'Sunrise']

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((150, 150))  # Resize to match model input size
    img_array = np.array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Streamlit app
st.title('Weather Image Classification')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Allow the user to select the model
    model_option = st.selectbox("Choose model:", ["Trained ModelAlpha"])

    # Select the model (Trained ModelAlpha in this case)
    if model_option == "Trained ModelAlpha":
        model = trained_model_alpha

    # Preprocess the image
    img = Image.open(uploaded_file)
    img_array = preprocess_image(img)

    # Make the prediction
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    # Display the prediction result
    st.write(f"Predicted Class: {predicted_class} with probability: {np.max(prediction) * 100:.2f}%")
