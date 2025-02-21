import streamlit as st
import os
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Check current working directory
st.write(f"Current working directory: {os.getcwd()}")

# Load the model (ensure the correct path)
try:
    model = load_model('models/handwriting_model.h5')  # Update the path if necessary
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Function to process image for prediction
def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((128, 128))  # Resize to model input size (adjust if necessary)
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI for uploading image
st.title("Doctor's Handwriting Recognition")

uploaded_image = st.file_uploader("Upload an image of handwritten text", type="jpg")

if uploaded_image is not None:
    try:
        # Display uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Process the image
        processed_image = preprocess_image(image)
        st.write(f"Processed Image Shape: {processed_image.shape}")

        # Predict using the model
        prediction = model.predict(processed_image)
        st.write(f"Prediction: {prediction}")  # Debugging line
        
        # Assuming the model returns a classification result
        predicted_text = np.argmax(prediction, axis=1)  # Get the class with the highest probability
        st.write(f"Predicted Text: {predicted_text}")

    except Exception as e:
        st.error(f"Error processing image or making prediction: {e}")
else:
    st.write("Please upload an image to start recognition.")
