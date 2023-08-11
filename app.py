# Import library
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

# Load tokenizer dan model
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = tf.keras.models.load_model("BiLSTM.h5")

# Layout website
st.title("Cyberbullying Detection")
tweet = st.text_input("Masukkan tweet")
tweets = [tweet]

# Fungsi pembantu
def preprocess(tweets):
    dummy = tokenizer.texts_to_sequences(tweets)
    dummy = tf.keras.preprocessing.sequence.pad_sequences(dummy, maxlen = 100, padding = "post")
    return dummy

def predicts():
    dummy = preprocess(tweets)
    prediction = np.argmax(model.predict(dummy)[0])
    if prediction == 0:
        st.success("Termasuk tweet non-bullying")
    elif prediction == 1:
        st.error("Termasuk tweet bullying melibatkan ras")
    elif prediction == 2:
        st.error("Termasuk tweet bullying melibatkan agama")
    else:
        st.error("Error")

st.button("Predict", on_click = predicts)