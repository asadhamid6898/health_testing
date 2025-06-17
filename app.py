import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Load the model once
@st.cache_resource
def load_fundus_model():
    return load_model("fundus_classifier.keras")

model = load_fundus_model()

# Preprocessing function
def is_fundus(img: Image.Image):
    img = img.resize((128, 128)).convert('RGB')  # Ensure correct size and 3 channels
    img_array = keras_image.img_to_array(img)    # shape (128, 128, 3)
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, 128, 128, 3)
    img_array = img_array / 255.0                # normalize
    prediction = model.predict(img_array)[0][0]  # get scalar
    return prediction >= 0.5

# Streamlit app UI
st.title("Fundus Image Verification")

uploaded_file = st.file_uploader("Upload a fundus image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    try:
        if is_fundus(img):
            st.success("✔️ Verified: This is a valid fundus image.")
        else:
            st.error("❌ This does not appear to be a valid fundus image.")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
