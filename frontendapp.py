import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import pandas as pd
import os
from PIL import Image

def load_model():
    try:
        model = tf.keras.models.load_model("models/handwriting_model.h5")  # Load your trained model
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    try:
        image = image.convert("RGB")  # Ensure image is in RGB format
        image = image.resize((64, 64))  # Resize to match training dimensions
        image_array = img_to_array(image) / 255.0  # Normalize
        return np.expand_dims(image_array, axis=0)  # Add batch dimension
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

st.title("Handwriting Detection App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    model = load_model()
    if model:
        processed_image = preprocess_image(image)
        if processed_image is not None:
            try:
                prediction = model.predict(processed_image)
                trainLabels = pd.read_csv("dataset/Training/training_labels.csv")
                classes = trainLabels["MEDICINE_NAME"].unique()
                if len(prediction.shape) > 1 and prediction.shape[1] == len(classes):
                    predicted_class = classes[np.argmax(prediction)]
                    st.write(f"Predicted Medicine Name: **{predicted_class}**")
                else:
                    st.error("Prediction output shape does not match expected class count.")
            except Exception as e:
                st.error(f"Error making prediction: {e}")