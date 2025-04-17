from tensorflow.keras.models import load_model
import streamlit as st

@st.cache_resource
def load_emotion_model(path='app/model/emotion_cnn.h5'):
    model = load_model(path)
    return model
