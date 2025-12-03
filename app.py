import streamlit as st
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

st.title("🖐️ Finger Counter App (Streamlit Cloud Compatible)")
st.write("Show your hand to the camera to count your fingers.")

start = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

detector = HandDetector(maxHands=1, detectionCon=0.7)

cap = cv2.VideoCapture(0)

while start:
    ret, img = cap.read()
    img = cv2.flip(img, 1)

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        fingers = detector.fingersUp(hand)  # returns [0,1,1,0,0] etc.
        count = sum(fingers)

        cv2.putText(img, f'{count}', (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)

    FRAME_WINDOW.image(img, channels="BGR")

cap.release()
