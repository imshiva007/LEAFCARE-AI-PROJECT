from pathlib import Path

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# 1. Project paths
# =========================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "leaf_disease_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.txt"

# =========================
# 2. Page configuration
# =========================

st.set_page_config(
    page_title="Leaf Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# =========================
# 3. Load trained model
# =========================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        return file.read().splitlines()

# =========================
# 4. App interface
# =========================

st.title("🌿 Leaf Disease Detection")
st.write(
    "Upload a clear leaf image to get an AI-generated "
    "prediction of its plant category."
)

st.info(
    "This is an educational ML project. Predictions may be "
    "incorrect and should not replace expert diagnosis."
)

if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
    st.error(
        "Model files nahi mile. Pehle python train.py run karo."
    )
    st.stop()

try:
    model = load_model()
    class_names = load_class_names()
except Exception as error:
    st.error(f"Model load nahi ho paya: {error}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# 5. Predict uploaded image
# =========================

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded Leaf Image")
        st.image(image, use_container_width=True)

        image_resized = image.resize((160, 160))

        image_array = np.array(
            image_resized,
            dtype=np.float32
        )

        image_batch = np.expand_dims(image_array, axis=0)

        if st.button("Detect Leaf Category", type="primary"):
            with st.spinner("Analyzing leaf image..."):
                predictions = model.predict(
                    image_batch,
                    verbose=0
                )[0]

                predicted_index = int(np.argmax(predictions))
                predicted_class = class_names[predicted_index]
                confidence = float(predictions[predicted_index]) * 100

            st.subheader("Prediction Result")

            st.write("**Predicted category:**")
            st.success(predicted_class.replace("_", " "))

            st.write("**Model confidence score:**")
            st.progress(min(confidence / 100, 1.0))
            st.write(f"{confidence:.2f}%")

            st.caption(
                "Confidence is the model's score, not a guarantee "
                "that the prediction is correct."
            )

            st.subheader("Top 3 Predictions")

            top_indices = np.argsort(predictions)[-3:][::-1]

            for index in top_indices:
                st.write(
                    f"- {class_names[index].replace('_', ' ')}: "
                    f"{predictions[index] * 100:.2f}%"
                )

    except Exception as error:
        st.error(f"Image process nahi ho payi: {error}")