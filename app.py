import streamlit as st
import numpy as np
import tensorflow as tf
from utils import one_hot_encode

# Load the trained model
model = tf.keras.models.load_model("trained_model.h5")

# Define class labels — adjust if your model uses different ones
classes = ['Coding', 'Non-Coding', 'Promoter']

# Streamlit UI
st.set_page_config(page_title="DNA Classifier", layout="centered")
st.title("🧬 DNA Sequence Classifier")
st.markdown("Paste a DNA sequence below to classify it as **Coding**, **Non-Coding**, or **Promoter**.")

# Input box
sequence_input = st.text_area("DNA Sequence", height=200)

# Classify button
if st.button("Classify"):
    if sequence_input:
        # Encode the sequence
        encoded = one_hot_encode(sequence_input)
        encoded = np.expand_dims(encoded, axis=0)  # Shape: (1, 1000, 4)

        # Make prediction
        prediction = model.predict(encoded)

        # Handle prediction shape
        if prediction.ndim == 2 and prediction.shape[1] == len(classes):
            prediction = prediction[0]  # Shape: (num_classes,)
            predicted_class = classes[np.argmax(prediction)]
            st.success(f"Prediction: **{predicted_class}**")

            st.markdown("### Confidence Scores")
            for i, score in enumerate(prediction):
                st.write(f"{classes[i]}: `{score:.4f}`")
        else:
            st.error("Unexpected model output shape. Please check your model architecture.")
    else:
        st.warning("Please enter a DNA sequence.")
