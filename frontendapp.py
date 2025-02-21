import streamlit as st
import numpy as np
import os
import pickle
from PIL import Image

# Load the model using Pickle
def load_handwriting_model():
    model_path = "models/handwriting_model.pkl"
    
    if not os.path.exists(model_path):
        st.error(f"❌ **Model file not found:** `{model_path}`")
        return None
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    return model

# Preprocess image before feeding it into the model
def preprocess_image(image):
    try:
        image = image.convert("L")  # Convert to grayscale
        image = image.resize((64, 64))  # Resize to match training dimensions
        image_array = np.array(image) / 255.0  # Normalize
        return image_array.reshape(1, 64, 64, 1)  # Add batch dimension
    except Exception as e:
        st.error(f"⚠️ **Error processing image:** {e}")
        return None

# Streamlit UI
st.title("✍️ Handwriting Detection App")

uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # Load Model
    model = load_handwriting_model()

    if model:
        processed_image = preprocess_image(image)
        
        if processed_image is not None:
            prediction = model.predict(processed_image)
            st.markdown(f"""
                <div style="text-align: center; font-size: 28px; font-weight: bold; color: #4CAF50;
                    background-color: #E8F5E9; padding: 15px; border-radius: 10px;">
                    ✅ Prediction: {prediction}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="text-align: center; font-size: 28px; font-weight: bold; color: red;
                    background-color: #FFEBEE; padding: 15px; border-radius: 10px;">
                    ❌ No image found! Please upload a valid image.
                </div>
            """, unsafe_allow_html=True)


