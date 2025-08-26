import streamlit as st
import numpy as np
import tensorflow as tf

# --- DNA One-Hot Encoding ---
def one_hot_encode(seq, max_len=1000):
    base_map = {'A': [1, 0, 0, 0],
                'T': [0, 1, 0, 0],
                'C': [0, 0, 1, 0],
                'G': [0, 0, 0, 1]}
    seq = seq.upper().replace('\n', '').replace(' ', '')
    seq = seq[:max_len].ljust(max_len, 'A')  # Pad with 'A' if too short
    return np.array([base_map.get(base, [0, 0, 0, 0]) for base in seq])

# --- Rebuild Model Architecture ---
def build_model(num_classes=3):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1000, 4)),
        tf.keras.layers.Conv1D(64, 7, activation='relu'),
        tf.keras.layers.MaxPooling1D(3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# --- Load Model Weights ---
model = build_model()
model.load_weights("model_weights.h5")  # Ensure this file is in your repo root

# --- Class Labels (adjust if needed) ---
classes = ['Coding', 'Non-Coding', 'Promoter']

# --- Streamlit UI ---
st.set_page_config(page_title="DNA Classifier", layout="centered")
st.title("🧬 DNA Sequence Classifier")
st.markdown("Paste a DNA sequence below to classify it as **Coding**, **Non-Coding**, or **Promoter**.")

sequence_input = st.text_area("DNA Sequence", height=200)

if st.button("Classify"):
    if sequence_input:
        encoded = one_hot_encode(sequence_input)
        encoded = np.expand_dims(encoded, axis=0)  # Shape: (1, 1000, 4)
        prediction = model.predict(encoded)[0]     # Shape: (num_classes,)

        # Safely zip predictions with class labels
        if len(prediction) == len(classes):
            predicted_class = classes[np.argmax(prediction)]
            st.success(f"Prediction: **{predicted_class}**")

            st.markdown("### Confidence Scores")
            for label, score in zip(classes, prediction):
                st.write(f"{label}: `{score:.4f}`")
        else:
            st.error("Mismatch between model output and class labels. Please check model configuration.")
    else:
        st.warning("Please enter a DNA sequence.")
