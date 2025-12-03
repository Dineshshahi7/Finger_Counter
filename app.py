import cv2
import mediapipe as mp
import streamlit as st
import numpy as np

st.title("🖐️ Finger Counter – Computer Vision")
st.write("Show your hand to the webcam. The model will detect how many fingers you raised.")

run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Hand landmark index for each fingertip
finger_tips = [4, 8, 12, 16, 20]  

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to access camera.")
        break

    frame = cv2.flip(frame, 1)  
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    finger_count = 0

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        lm_list = []
        h, w, c = frame.shape

        for id, lm in enumerate(hand_landmarks.landmark):
            lm_list.append([id, int(lm.x * w), int(lm.y * h)])

        # Thumb Logic
        # Compare x-coordinates for thumb detection (works for both hands)
        if lm_list[4][1] > lm_list[3][1]:  
            finger_count += 1

        # Other Fingers Logic (compare y-coordinates)
        for tip in [8, 12, 16, 20]:
            if lm_list[tip][2] < lm_list[tip - 2][2]:
                finger_count += 1

        # Display Result
        cv2.putText(frame, f"{finger_count}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    FRAME_WINDOW.image(frame, channels="BGR")

cap.release()
