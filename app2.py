import streamlit as st
import cv2
import tempfile
import mediapipe as mp
import pandas as pd
import plotly.graph_objects as go
import io
import base64
import os
import numpy as np
from PIL import Image
from gtts import gTTS

# Initialisation MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# Fonctions utilitaires
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def estimate_height(landmarks, frame_height):
    nose_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y * frame_height
    ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * frame_height
    pixel_height = abs(ankle_y - nose_y)
    real_height_cm = (pixel_height / 500) * 170
    return real_height_cm

def estimate_weight(height_cm):
    return (height_cm / 100) ** 2 * 22

def classify_zone(angle):
    if 90 <= angle <= 130:
        return "✅ Safe"
    elif (60 <= angle < 90) or (130 < angle <= 150):
        return "⚠️ Caution"
    else:
        return "❌ Danger"

def calculate_torque(weight_kg, hip, knee, frame_width):
    weight_force = weight_kg * 0.6 * 9.81
    hip_x = hip[0] * frame_width
    knee_x = knee[0] * frame_width
    d = abs(knee_x - hip_x) / 100
    torque = weight_force * d
    return torque

# ====== DESIGN: BG IMAGE ======
def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """, unsafe_allow_html=True)

set_background("assets/background.jpg")


# ====== HEADER ======
st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B;'>🏋️ AI Squat Analyzer</h1>
    <h4 style='text-align: center;'>Analyse biomécanique en temps réel des squats</h4>
""", unsafe_allow_html=True)


# Charger le CSS du chatbot
with open("assets/chatbot.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Charger le JS du chatbot
with open("assets/chatbot.js") as f:
    js_code = f.read()

# === Charger logo en base64 ===
with open("assets/logo.png", "rb") as f:
    logo_base64 = base64.b64encode(f.read()).decode()

# Charger l'icône en base64
with open("assets/chatbot_icon.png", "rb") as f:
    icon_base64 = base64.b64encode(f.read()).decode()

# === Injecter le HTML du chatbot ===
chatbot_html = f"""
<button id="chatButton">
    <img src="data:image/png;base64,{icon_base64}">
</button>

<div id="chatWindow">
    <div id="chatContent"></div>
    <input id="chatInput" placeholder="Écrire..." onkeydown="if(event.key==='Enter') sendMessage()">
</div>
"""
st.markdown(chatbot_html, unsafe_allow_html=True)

# Injecter JS
st.components.v1.html(f"""
<script>{js_code}</script>
""", height=0, width=0)

# === Footer fixe ===
st.markdown("""
<footer>
    powered by Yassine Melouli - 2025
</footer>
""", unsafe_allow_html=True)

# ====== OPTION DE SOURCE ======
option = st.radio("🎥 Choose Input Source : ", ["📁 Upload video", "📷 Use webcam"])


# ====== FONCTION : feedback vocal via gTTS ======
def speak(text):
    tts = gTTS(text, lang='fr')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        tts.save(f.name)
        st.audio(f.read(), format="audio/mp3")

# ====== FONCTION DÉTECTION MOCK (à remplacer) ======
def analyse_squat_from_video(video_path):
    st.info("🔍 Analyse de la vidéo en cours...")

    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    frame_counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_counter > 200:
            break

        frame = cv2.resize(frame, (640, 480))
        stframe.image(frame, channels="BGR")
        frame_counter += 1

    cap.release()
    st.success("✅ Analyse terminée !")
    speak("Analyse terminée !")


# ====== UPLOAD VIDEO ======
if option == "📁 Télécharger une vidéo":
    uploaded_file = st.file_uploader("📤 Upload une vidéo MP4", type=["mp4"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        analyse_squat_from_video(tfile.name)

# ====== CAMERA ======
elif option == "📷 Utiliser la webcam":
    st.warning("📷 Mode webcam est en cours de développement.")
    st.info("Utilisez la version vidéo pour l’instant.")


