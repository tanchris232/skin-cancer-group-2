import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

# Load your pre-trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('/Users/raphael/Documents/GitHub/skin-cancer-group-2/skin_cancer_classifier.h5')
    return model

model = load_model()

# Function to preprocess the image and make predictions
def preprocess_image(image):
    size = (224, 224)  # The size your model expects
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 255.0)  # Ensure this matches training preprocessing
    data = np.expand_dims(normalized_image_array, axis=0)
    return data

def predict_image(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return "Malignant" if prediction[0][0] > 0.5 else "Benign"

# Streamlit app layout
st.title("Mole Type Classification")
st.write("Please upload an image to determine whether it is classified as benign or malignant.")

uploaded_file = st.file_uploader("Please choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    label = predict_image(image)
    st.write(f'Prediction: {label}')
