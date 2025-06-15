
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Dataset
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import streamlit as st
# import os
# from PIL import Image
# import cv2
# import librosa
# import soundfile as sf
# import speech_recognition as sr
# import seaborn as sns

# st.set_page_config(page_title="Dyslexia Detector", layout="wide")

# st.markdown("""
#     <style>
#     .main {
#         background-color: #f0f2f6;
#     }
#     .stButton>button {
#         color: white;
#         background: #4CAF50;
#         padding: 0.5em 1.5em;
#         border-radius: 8px;
#     }
#     .stRadio > div {
#         background: #e0e0e0;
#         padding: 10px;
#         border-radius: 10px;
#     }
#     </style>
# """, unsafe_allow_html=True)

# st.sidebar.image("https://img.icons8.com/ios-filled/100/brain.png", width=100)
# st.sidebar.title("🧠 Dyslexia Detection")
# st.sidebar.info("ML-powered detection using handwriting and speech")
# st.sidebar.markdown("---")
# st.sidebar.markdown("📂 Upload sample to begin\n🔊 Supports image and audio")

# class DyslexiaDataset(Dataset):
#     def __init__(self, data, labels):
#         self.data = torch.tensor(data, dtype=torch.float32)
#         self.labels = torch.tensor(labels, dtype=torch.float32)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx], self.labels[idx]

# class LiquidNeuralNetwork(nn.Module):
#     def __init__(self, input_size):
#         super(LiquidNeuralNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_size, 64)
#         self.liquid_layer = nn.LSTM(64, 32, batch_first=True)
#         self.fc2 = nn.Linear(32, 1)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = x.unsqueeze(1)
#         x, _ = self.liquid_layer(x)
#         x = x[:, -1, :]
#         x = torch.sigmoid(self.fc2(x))
#         return x.squeeze()

# def extract_features_from_image(image_path):
#     img = cv2.imread(image_path, 0)
#     img = cv2.resize(img, (64, 64))
#     features = img.flatten() / 255.0
#     return features

# def load_real_data(folder="handwriting_samples"):
#     X, y = [], []
#     for fname in os.listdir(folder):
#         label = 1 if "dyslexic" in fname else 0
#         features = extract_features_from_image(os.path.join(folder, fname))
#         X.append(features)
#         y.append(label)
#     if not X:
#         return np.array([]), np.array([]), np.array([]), np.array([])
#     return train_test_split(np.array(X), np.array(y), test_size=0.2, random_state=42)

# def extract_features_from_audio(audio_path):
#     y, sr = librosa.load(audio_path, sr=None)
#     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#     return np.mean(mfccs.T, axis=0)

# def train_model(model, train_loader, criterion, optimizer, device):
#     model.train()
#     for epoch in range(10):
#         total_loss = 0
#         for data, labels in train_loader:
#             data, labels = data.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(data)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# def evaluate_model(model, test_loader, device):
#     model.eval()
#     all_preds = []
#     all_labels = []
#     with torch.no_grad():
#         for data, labels in test_loader:
#             data, labels = data.to(device), labels.to(device)
#             outputs = model(data)
#             preds = (outputs > 0.5).int()
#             all_preds.extend(preds.cpu().numpy().flatten().tolist())
#             all_labels.extend(labels.cpu().numpy().flatten().tolist())
#     report = classification_report(all_labels, all_preds, output_dict=True)
#     df = pd.DataFrame(report).transpose()
#     st.write("### Classification Report")
#     st.dataframe(df.style.background_gradient(cmap='Blues'))
#     st.write("### Prediction Distribution")
#     st.bar_chart(pd.Series(all_preds).value_counts())

# def get_prescription(label):
#     try:
#         df = pd.read_csv("prescriptions.csv")
#         suggestion = df[df['label'] == label]['suggestion'].values
#         return suggestion[0] if len(suggestion) > 0 else "No suggestion found."
#     except FileNotFoundError:
#         return "Prescription file not found. Please add prescriptions.csv."


# def streamlit_app(model):
#     st.title("🧠 Dyslexia Detection App")
#     st.subheader("Modern ML-powered Dyslexia Screening")

#     option = st.radio("Select Input Type:", ["Handwriting Sample", "Speech Sample"])

#     if option == "Handwriting Sample":
#         uploaded_file = st.file_uploader("Choose handwriting image", type=["jpg", "png", "jpeg"])
#         if uploaded_file is not None:
#             image = Image.open(uploaded_file)
#             st.image(image, caption="Uploaded Sample", use_column_width=True)
#             image_path = "temp_image.png"
#             image.save(image_path)

#             features = extract_features_from_image(image_path)
#             tensor_input = torch.tensor(features, dtype=torch.float32).to(next(model.parameters()).device)
#             model.eval()
#             with torch.no_grad():
#                 prediction = model(tensor_input.unsqueeze(0))
#             label = "Dyslexic" if prediction.item() > 0.5 else "Non-Dyslexic"
#             st.success(f"🧾 Prediction: **{label}** with confidence {prediction.item():.2f}")
#             st.info(f"📘 Prescription: {get_prescription(label)}")

#     elif option == "Speech Sample":
#         audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
#         if audio_file is not None:
#             audio_path = "temp_audio.wav"
#             with open(audio_path, "wb") as f:
#                 f.write(audio_file.read())
#             features = extract_features_from_audio(audio_path)
#             tensor_input = torch.tensor(features, dtype=torch.float32).to(next(model.parameters()).device)
#             model.eval()
#             with torch.no_grad():
#                 prediction = model(tensor_input.unsqueeze(0))
#             label = "Dyslexic" if prediction.item() > 0.5 else "Non-Dyslexic"
#             st.success(f"🧾 Prediction: **{label}** with confidence {prediction.item():.2f}")
#             st.info(f"📘 Prescription: {get_prescription(label)}")

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     X_train, X_test, y_train, y_test = load_real_data()

#     if len(X_train) == 0:
#         st.warning("No handwriting data found. Please add sample images to 'handwriting_samples' folder.")
#     else:
#         train_dataset = DyslexiaDataset(X_train, y_train)
#         test_dataset = DyslexiaDataset(X_test, y_test)
#         train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#         test_loader = DataLoader(test_dataset, batch_size=32)

#         model = LiquidNeuralNetwork(input_size=X_train.shape[1]).to(device)
#         criterion = nn.BCELoss()
#         optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#         train_model(model, train_loader, criterion, optimizer, device)
#         evaluate_model(model, test_loader, device)

#         streamlit_app(model)
# FULL INTEGRATED DYSLEXIA DETECTION APP
# FULL INTEGRATED DYSLEXIA DETECTION APP (UPDATED)
# FULL INTEGRATED DYSLEXIA DETECTION APP (UPDATED)
# FULL INTEGRATED DYSLEXIA DETECTION APP (UPDATED)
# FULL INTEGRATED DYSLEXIA DETECTION APP (UPDATED)
import os
import speech_recognition as sr
from difflib import SequenceMatcher
import pandas as pd
import streamlit as st
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
import librosa
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import classification_report
from pydub import AudioSegment
import io
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import tempfile

# ------------------- SETUP -------------------
st.set_page_config(page_title="Dyslexia Detection App", layout="wide")

st.sidebar.image("https://img.icons8.com/ios-filled/100/brain.png", width=100)
st.sidebar.title("🧠 Dyslexia Detection")
option = st.sidebar.radio("Select Input Type:", ["Handwriting Sample", "Speech Sample", "Read-Aloud Assessment"])

# ------------------- IMAGE MODEL -------------------
def extract_features_from_image(image_path):
    img = cv2.imread(image_path, 0)
    img = cv2.resize(img, (64, 64))
    features = img.flatten() / 255.0
    return features

# ------------------- AUDIO MODEL -------------------
def extract_features_from_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# ------------------- READ-ALOUD MODULE -------------------
sample_levels = {
    "Beginner": "The cat sat on the mat.",
    "Intermediate": "Reading is a wonderful way to learn new things.",
    "Advanced": "The efficiency of the neural network greatly depends on the dataset's variance."
}

def convert_audio_to_wav(input_bytes):
    audio = AudioSegment.from_file(io.BytesIO(input_bytes))
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    buffer.seek(0)
    return buffer

def speech_to_text(audio_bytes):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_bytes) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"API error: {e}"

def compare_sentences(expected, spoken):
    ratio = SequenceMatcher(None, expected.lower(), spoken.lower()).ratio()
    return ratio, 1 - ratio

def save_history(name, level, sentence, spoken, accuracy):
    data = {
        "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "User": [name],
        "Level": [level],
        "Target Sentence": [sentence],
        "Spoken Sentence": [spoken],
        "Accuracy": [f"{accuracy*100:.2f}%"]
    }
    df_new = pd.DataFrame(data)
    if os.path.exists("reading_history.csv"):
        df_existing = pd.read_csv("reading_history.csv")
        df_existing = pd.concat([df_existing, df_new], ignore_index=True)
        df_existing.to_csv("reading_history.csv", index=False)
    else:
        df_new.to_csv("reading_history.csv", index=False)

# ------------------- MODEL CLASS -------------------
class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(LiquidNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.liquid_layer = nn.LSTM(64, 32, batch_first=True)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(1)
        x, _ = self.liquid_layer(x)
        x = x[:, -1, :]
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze()

# ------------------- TRAINING SETUP -------------------
def load_real_data(folder="handwriting_samples"):
    X, y = [], []
    for fname in os.listdir(folder):
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            label = 1 if "dyslexic" in fname.lower() else 0
            features = extract_features_from_image(os.path.join(folder, fname))
            X.append(features)
            y.append(label)
    if not X:
        return np.array([]), np.array([]), np.array([]), np.array([])
    return train_test_split(np.array(X), np.array(y), test_size=0.2, random_state=42)

@st.cache_resource
def train_model_once():
    X_train, X_test, y_train, y_test = load_real_data("handwriting_samples")
    if len(X_train) == 0:
        return None
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                   torch.tensor(y_train, dtype=torch.float32))
    model = LiquidNeuralNetwork(input_size=X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for epoch in range(5):
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model

# ------------------- MAIN STREAMLIT APP -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = train_model_once()
if model is None:
    st.warning("No training data found in 'handwriting_samples'. Please add some sample images.")

if option == "Handwriting Sample" and model:
    st.subheader("Upload Handwriting Sample")
    uploaded_file = st.file_uploader("Choose handwriting image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Sample", use_column_width=True)
        image_path = "temp_img.png"
        image.save(image_path)

        features = extract_features_from_image(image_path)
        tensor_input = torch.tensor(features, dtype=torch.float32).to(device)

        model.eval()
        with torch.no_grad():
            prediction = model(tensor_input.unsqueeze(0))
        label = "Dyslexic" if prediction.item() > 0.5 else "Non-Dyslexic"
        st.success(f"🗞️ Prediction: **{label}** with confidence {prediction.item():.2f}")
        prediction = model(tensor_input.unsqueeze(0))
        label = "Dyslexic" if prediction.item() > 0.5 else "Non-Dyslexic"
        st.success(f"🗞️ Prediction: **{label}** with confidence {prediction.item():.2f}")

elif option == "Speech Sample":
    st.subheader("Upload Speech Sample")
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    if audio_file:
        try:
            wav_buffer = convert_audio_to_wav(audio_file.read())
            with open("temp_audio.wav", "wb") as out:
                out.write(wav_buffer.read())
                wav_buffer.seek(0)

            y, sr = librosa.load(wav_buffer, sr=None)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features = np.mean(mfccs.T, axis=0)
            tensor_input = torch.tensor(features, dtype=torch.float32).to(device)

            model = LiquidNeuralNetwork(input_size=len(features)).to(device)
            model.load_state_dict(torch.load("dyslexia_model.pth", map_location=device))
            model.eval()
            with torch.no_grad():
                prediction = model(tensor_input.unsqueeze(0))
            label = "Dyslexic" if prediction.item() > 0.5 else "Non-Dyslexic"
            st.success(f"🗞️ Prediction: **{label}** with confidence {prediction.item():.2f}")

        except Exception as e:
            st.error(f"❌ Error processing audio: {e}")

elif option == "Read-Aloud Assessment":
    level = st.radio("📚 Select Difficulty Level:", list(sample_levels.keys()))
    sample_sentence = sample_levels[level]
    st.subheader("📘 Please read this sentence aloud:")
    st.markdown(f"**{sample_sentence}**")

    st.markdown("Use a tool like Vocaroo or your phone's mic recorder, or record directly if supported.")
    st.markdown("You can also use your **microphone** to speak directly.")
    st.markdown("### 🎤 Record via Microphone")
    webrtc_ctx = webrtc_streamer(
        key="readaloud",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        async_processing=True,
    )

    audio_input = None
    if webrtc_ctx.audio_receiver:
        try:
            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=5)
            audio = b"".join([f.to_ndarray().tobytes() for f in audio_frames])
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio)
                audio_input = open(tmp.name, "rb")
        except Exception as e:
            st.warning(f"Microphone recording failed: {e}")

    audio_input = st.file_uploader("🎤 Or upload a recording (WAV/MP3)", type=["wav", "mp3"])
    username = st.text_input("Enter Your Name for Report:", "User")

    if audio_input and username:
        try:
            wav_buffer = convert_audio_to_wav(audio_input.read())
            spoken_text = speech_to_text(wav_buffer)
            st.write("🗣️ You said:", spoken_text)

            if spoken_text == "Could not understand audio" or spoken_text.startswith("API error"):
                st.error(spoken_text)
            else:
                accuracy, error_rate = compare_sentences(sample_sentence, spoken_text)
                st.progress(int(accuracy * 100))
                st.write(f"**Reading Accuracy:** {accuracy*100:.2f}%")

                if accuracy < 0.6:
                    st.error("⚠️ Significant reading difficulties detected.")
                    st.info("📘 Suggested: Phoneme training, sentence reconstruction games, speech therapy")
                elif accuracy < 0.85:
                    st.warning("⚠️ Minor signs of dyslexia. Monitor reading fluency regularly.")
                    st.info("📘 Suggested: Daily reading aloud practice, phonics exercises")
                else:
                    st.success("✅ Excellent reading accuracy. No signs of dyslexia detected.")

                save_history(username, level, sample_sentence, spoken_text, accuracy)
        except Exception as e:
            st.error(f"❌ Error processing audio: {e}")

    if os.path.exists("reading_history.csv"):
        st.markdown("---")
        st.write("### 📊 Past Reading History")
        df_hist = pd.read_csv("reading_history.csv")
        st.dataframe(df_hist.tail(10).style.highlight_max(axis=0))




