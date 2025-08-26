import streamlit as st
import numpy as np
import tensorflow as tf
from utils import one_hot_encode

# Load the trained model
model = tf.keras.models.load_model("trained_model.h5")

# Define class labels (adjust if needed)
classes = ['Coding', 'Non-Coding', 'Promoter']

# Streamlit UI
st.set_page_config(page_title="DNA Classifier", layout="centered")
st.title("🧬 DNA Sequence Classifier")
st.markdown("Paste a DNA sequence below to classify it as **Coding**, **Non-Coding**, or **Promoter**.")

sequence_input = st.text_area("DNA Sequence", height=200)

if st.button("Classify"):
    if sequence_input:
        encoded = one_hot_encode(sequence_input)
        prediction = model.predict(np.expand_dims(encoded, axis=0))[0]

        if prediction.ndim == 0 or prediction.shape == ():  # scalar output
            st.success(f"Prediction Score: {prediction:.4f}")
        else:
            predicted_class = classes[np.argmax(prediction)]
            st.success(f"Prediction: **{predicted_class}**")

            st.markdown("### Confidence Scores")
            for i, score in enumerate(prediction):
                st.write(f"{classes[i]}: `{score:.4f}`")
    else:
        st.warning("Please enter a DNA sequence.")
