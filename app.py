import streamlit as st
import cv2
import tempfile
import numpy as np
import mediapipe as mp
import pandas as pd
import plotly.graph_objects as go
import io

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

# Streamlit interface
st.title("AI Squat Biomechanics Analyzer")

option = st.radio(
    "Choose Input Source:",
    ("Webcam", "Upload Video")
)

if option == "Upload Video":
    uploaded_file = st.file_uploader("Upload your squat video", type=['mp4', 'mov', 'avi'])
    if uploaded_file is not None:
        # Save temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)

elif option == "Webcam":
    cap = cv2.VideoCapture(0)

if option and (uploaded_file is not None or option == "Webcam"):
    frame_counter = 0
    squat_started = False
    squat_count = 0

    angles_left = []
    angles_right = []
    torques_left = []
    torques_right = []
    zones_left = []
    zones_right = []

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # LEFT
            hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            angle_left = calculate_angle(hip_left, knee_left, ankle_left)
            zone_left = classify_zone(angle_left)

            # RIGHT
            hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            angle_right = calculate_angle(hip_right, knee_right, ankle_right)
            zone_right = classify_zone(angle_right)

            height_cm = estimate_height(landmarks, frame.shape[0])
            weight_kg = estimate_weight(height_cm)

            torque_left = calculate_torque(weight_kg, hip_left, knee_left, frame.shape[1])
            torque_right = calculate_torque(weight_kg, hip_right, knee_right, frame.shape[1])

            if angle_left < 100 or angle_right < 100:
                if not squat_started:
                    squat_started = True
                    squat_count += 1
            else:
                if squat_started:
                    squat_started = False

            # Annoter frame
            cv2.putText(frame, f"Left Angle: {int(angle_left)} deg {zone_left}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f"Right Angle: {int(angle_right)} deg {zone_right}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            angles_left.append(angle_left)
            angles_right.append(angle_right)
            torques_left.append(torque_left)
            torques_right.append(torque_right)
            zones_left.append(zone_left)
            zones_right.append(zone_right)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB")

        frame_counter += 1

    cap.release()

    # Afficher stats finales
    st.write(f"Total Squats: {squat_count}")
    st.write(f"Mean Left Angle: {np.mean(angles_left):.2f}")
    st.write(f"Mean Right Angle: {np.mean(angles_right):.2f}")

    # Plotly graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=angles_left, name="Left Knee Angle"))
    fig.add_trace(go.Scatter(y=angles_right, name="Right Knee Angle"))
    fig.update_layout(title="Knee Angles Over Time",
                      yaxis_title="Angle (deg)")
    st.plotly_chart(fig)

    # CSV download
    df = pd.DataFrame({
        "Frame": list(range(frame_counter)),
        "Angle_Left": angles_left,
        "Angle_Right": angles_right,
        "Zone_Left": zones_left,
        "Zone_Right": zones_right,
        "Torque_Left": torques_left,
        "Torque_Right": torques_right
    })

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Metrics CSV", csv, "squat_metrics.csv", "text/csv")

