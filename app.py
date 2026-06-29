import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# Load Model
model = tf.keras.models.load_model("brain_tumor_ann.keras")

# Class Labels
classes = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

# Title
st.title("Brain Tumor Classification using ANN")
st.subheader("Team: ML Mavericks")

# Upload Image
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "png", "jpeg"]
)

# Reset Button
if st.button("Reset / Clear"):
    st.rerun()

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, width=300)

    if st.button("Predict"):

        start = time.time()

        img = image.resize((64, 64))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)

        index = np.argmax(prediction)

        confidence = prediction[0][index] * 100

        end = time.time()

        st.success(f"Prediction : {classes[index]}")
        st.info(f"Confidence : {confidence:.2f}%")
        st.write(f"Processing Time : {end-start:.3f} seconds")

        # Clinical Information
        st.subheader("Clinical Information")

        if classes[index] == "Glioma":
            st.write("Glioma is a brain tumor that develops from glial cells. It may cause headaches, seizures, and vision problems.")

        elif classes[index] == "Meningioma":
            st.write("Meningioma is usually a slow-growing tumor that develops in the meninges surrounding the brain.")

        elif classes[index] == "Pituitary":
            st.write("Pituitary tumors develop in the pituitary gland and may affect hormone production.")

        else:
            st.write("No brain tumor detected. The MRI appears normal.")

        # Model Details
        st.subheader("Model Details")
        st.write("Input Image Size : 64 × 64")
        st.write("Hidden Layers : 128 → 64 → 32")
        st.write("Activation : ReLU")
        st.write("Output : Softmax")
        st.write("Optimizer : Adam")
        st.write("Loss : Categorical Crossentropy")

# Team Details
st.markdown("---")
st.header("Team Details")

st.write("**Team Name:** ML Mavericks")
st.write("**Member 1:** Peerti Katti")
st.write("**Member 2:** Shivani Hatti")
st.write("**Member 3:** Keertana Hiremath")
st.write("**Member 4:** Basamma Hosamani")
st.write("**Department:** Artificial Intelligence & Machine Learning")
st.write("**College:** BVVS Basaveshwar Engineering College, Bagalkote")