import streamlit as st
import numpy as np
import tensorflow as tf

# --- DNA Encoding ---
def one_hot_encode(seq, max_len=1000):
    base_map = {'A': [1, 0, 0, 0],
                'T': [0, 1, 0, 0],
                'C': [0, 0, 1, 0],
                'G': [0, 0, 0, 1]}
    seq = seq.upper().replace('\n', '').replace(' ', '')
    seq = seq[:max_len].ljust(max_len, 'A')
    return np.array([base_map.get(base, [0, 0, 0, 0]) for base in seq])

# --- Rebuild Model Architecture ---
def build_model():
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
        tf.keras.layers.Dense(3, activation='softmax')  # Adjust if you have different number of classes
    ])
    return model

# --- Load Model Weights ---
model = build_model()
model.load_weights("model_weights.h5")  # Make sure this file is in your repo root

# --- Class Labels ---
classes = ['Coding', 'Non-Coding', 'Promoter']  # Adjust if needed

# --- Streamlit UI ---
st.set_page_config(page_title="DNA Classifier", layout="centered")
st.title("🧬 DNA Sequence Classifier")
st.markdown("Paste a DNA sequence below to classify it as **Coding**, **Non-Coding**, or **Promoter**.")

sequence_input = st.text_area("DNA Sequence", height=200)

if st.button("Classify"):
    if sequence_input:
        encoded = one_hot_encode(sequence_input)
        encoded = np.expand_dims(encoded, axis=0)  # Shape: (1, 1000, 4)
        prediction = model.predict(encoded)[0]     # Shape: (3,)
        predicted_class = classes[np.argmax(prediction)]

        st.success(f"Prediction: **{predicted_class}**")
        st.markdown("### Confidence Scores")
        for i, score in enumerate(prediction):
            st.write(f"{classes[i]}: `{score:.4f}`")
    else:
        st.warning("Please enter a DNA sequence.")
