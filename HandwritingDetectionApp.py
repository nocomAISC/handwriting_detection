import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import pytesseract 
import os


MODEL_PATH = "prescription_classification_VGG16.keras"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
    return None

def preprocess_image(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (128, 32)) / 255.0
    return np.expand_dims(image, axis=[0, -1])

def decode_prediction(prediction, char_list):
    return ''.join([char_list[i] for i in np.argmax(prediction, axis=-1) if i < len(char_list)]).strip()

st.title("Handwriting Recognition")
st.write("Upload an image with handwritten text, and we'll extract it for you!")


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])


model = load_model()
CHAR_LIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if model:
            try:
                prediction = model.predict(preprocess_image(image))
                word = decode_prediction(np.squeeze(prediction), CHAR_LIST)
                st.subheader(f"Predicted Word: `{word}`")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            extracted_text = pytesseract.image_to_string(image).strip()
            if extracted_text:
                st.subheader(f"Prescription Result: `{extracted_text}`")
    except Exception as e:
        st.error(f"Error: Unable to open image. Please upload a valid image file. {e}")
