import os
import gdown
from tensorflow.keras.models import load_model

MODEL_PATH = "app/model/emotion_cnn.h5"
DRIVE_URL = "https://drive.google.com/uc?id=1rlJq1neQyPk_LkMy5AxKubIZDH-g4dCH

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Model not found locally. Downloading from Google Drive...")
        gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
    else:
        print("Model already exists. Skipping download.")

def get_emotion_model():
    download_model()
    model = load_model(MODEL_PATH)
    return model

