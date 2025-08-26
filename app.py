import streamlit as st
import numpy as np
import tensorflow as tf
from utils import one_hot_encode

# Load model
model = tf.keras.models.load_model("trained_model.h5")

st.title("🧬 DNA Sequence Classifier")
st.write("Paste a DNA sequence below to classify it:")

sequence_input = st.text_area("DNA Sequence", height=200)

if st.button("Classify"):
    if sequence_input:
        encoded = one_hot_encode(sequence_input)
        prediction = model.predict(np.expand_dims(encoded, axis=0))[0]
        classes = ['Coding', 'Non-Coding', 'Promoter']
        st.success(f"Prediction: **{classes[np.argmax(prediction)]}**")
        st.write("Confidence scores:")
        for i, score in enumerate(prediction):
            st.write(f"{classes[i]}: {score:.4f}")
    else:
        st.warning("Please enter a DNA sequence.")
