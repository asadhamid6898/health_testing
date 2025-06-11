import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Load model once using st.cache_resource
@st.cache_resource
def load_fundus_model():
    return load_model("fundus_classifier.keras")

model = load_fundus_model()

def is_fundus(img: Image.Image):
    img = img.resize((128, 128)).convert('RGB')
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)[0][0]
    return prediction >= 0.5  # assuming binary classification

st.title("Fundus Image Verification")

uploaded_file = st.file_uploader("Upload a fundus image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if is_fundus(img):
        st.success("✔️ Verified: This is a valid fundus image.")
    else:
        st.error("❌ This does not appear to be a valid fundus image.")
