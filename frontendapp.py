import streamlit as st
import numpy as np
import os
import pickle
from PIL import Image
import pytesseract  # Tesseract OCR for text extraction

# Ensure "models" directory exists
os.makedirs("models", exist_ok=True)
model_path = "models/handwriting_model.pkl"

# Dummy Model (Fallback if no real model exists)
class DummyModel:
    def predict(self, X):
        return ["Handwriting Detected"] if isinstance(X, np.ndarray) else ["Invalid Input"]

# Load or create a dummy model
def load_handwriting_model():
    try:
        if not os.path.exists(model_path):
            with open(model_path, "wb") as f:
                pickle.dump(DummyModel(), f)

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model if hasattr(model, "predict") else None
    except Exception as e:
        st.error(f"⚠️ **Error loading model:** {e}")
        return None

# UI Design
st.markdown("""
    <h1 style="text-align:center; font-size: 50px; color: #4CAF50;">✍️ Handwriting Recognition</h1>
    <p style="text-align:center; font-size: 22px; color: #555;">Upload an image with handwritten text, and we'll extract it for you!</p>
    <hr>
""", unsafe_allow_html=True)

# Image Upload
uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    # Load the model
    model = load_handwriting_model()

    # Image Processing & Text Extraction
    if model:
        try:
            # Use Tesseract to extract text from the image
            extracted_text = pytesseract.image_to_string(image).strip()

            if extracted_text:
                st.markdown(f"""
                    <div style="text-align: center; font-size: 30px; font-weight: bold; 
                        color: #4CAF50; background-color: #E8F5E9; padding: 20px; 
                        border-radius: 15px; box-shadow: 2px 2px 10px #bbb;">
                        ✅ Predicted Prescription:
                        <p style="font-size: 24px; font-weight: normal; color: #333;">{extracted_text}</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style="text-align: center; font-size: 28px; font-weight: bold; 
                        color: red; background-color: #FFEBEE; padding: 20px; 
                        border-radius: 15px; box-shadow: 2px 2px 10px #bbb;">
                        ⚠️ No text detected in the image.
                    </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ Error during text extraction: {e}")
    else:
        st.markdown("""
            <div style="text-align: center; font-size: 28px; font-weight: bold; 
                color: red; background-color: #FFEBEE; padding: 20px; 
                border-radius: 15px; box-shadow: 2px 2px 10px #bbb;">
                ❌ Model loading failed! Please try again later.
            </div>
        """, unsafe_allow_html=True)