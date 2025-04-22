import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import librosa
import os

def build_cnn_model(input_shape=(64, 64, 1)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5)
    ])
    return model

def build_rnn_model(input_shape=(128, 128)):
    model = models.Sequential([
        layers.Reshape((128, 128, 1), input_shape=input_shape + (1,)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Reshape((63, 63 * 32)),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5)
    ])
    return model

def build_multimodal_model():
    face_input = layers.Input(shape=(64, 64, 1), name='face_input')
    speech_input = layers.Input(shape=(128, 128, 1), name='speech_input')
    cnn_model = build_cnn_model()(face_input)
    rnn_model = build_rnn_model()(speech_input)
    fused = layers.Concatenate()([cnn_model, rnn_model])
    fused = layers.Dense(256, activation='relu')(fused)
    fused = layers.Dropout(0.5)(fused)
    output = layers.Dense(3, activation='softmax', name='output')(fused)
    model = models.Model(inputs=[face_input, speech_input], outputs=output)
    return model

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model.h5')
    try:
        if os.path.exists(model_path):
            return tf.keras.models.load_model(model_path)
        else:
            raise FileNotFoundError("Model file not found. Train and save the model first.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return build_multimodal_model()  # Fallback to untrained model

def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('L').resize((64, 64))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = img_array.reshape(1, 64, 64, 1)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def preprocess_audio(audio_path):
    try:
        y, sr = librosa.load(audio_path)
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        if spectrogram.shape[1] > 128:
            spectrogram = spectrogram[:, :128]
        else:
            spectrogram = np.pad(spectrogram, ((0, 0), (0, 128 - spectrogram.shape[1])), mode='constant')
        spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
        spectrogram = spectrogram.reshape(1, 128, 128, 1)
        return spectrogram
    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        return None
