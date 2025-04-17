# app/emotion_model/train_model.py

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import gdown

# Download fer2013.csv if not already available
def download_fer2013():
    os.makedirs('data', exist_ok=True)
    file_path = 'data/fer2013.csv'

    if not os.path.exists(file_path):
        print("fer2013.csv not found. Downloading from Google Drive...")
        file_id = '1BJLbmeVE1IgSZMGYbcAZ0rOi1kdnyRyx'  
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, file_path, quiet=False)
        print("Download complete.")
    else:
        print("fer2013.csv already exists.")

# Load and preprocess data
def load_and_preprocess():
    df = pd.read_csv('data/fer2013.csv')

    emotion_map = {0: 'Angry', 1: 'Happy', 2: 'Neutral', 3: 'Sad'}
    df = df[df['emotion'].isin(emotion_map.keys())]
    df['emotion'] = df['emotion'].map(emotion_map)

    X = np.array([np.fromstring(pix, sep=' ').reshape(48, 48, 1) for pix in df['pixels']])
    X = X / 255.0
    y = pd.get_dummies(df['emotion']).values

    return train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN model
def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and save model
def train_and_save():
    download_fer2013()
    X_train, X_test, y_train, y_test = load_and_preprocess()

    model = build_model()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=64)

    os.makedirs("app/emotion_model", exist_ok=True)
    model.save("app/emotion_model/emotion_cnn.h5")
    print("Model saved to app/emotion_model/emotion_cnn.h5")

if __name__ == "__main__":
    train_and_save()

