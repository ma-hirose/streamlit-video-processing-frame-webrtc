import cv2
import streamlit as st
import numpy as np

camera_index = 0
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    st.error("Cannot open the camera. Check camera connections.")
else:
    stframe = st.empty()
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Could not retrieve frame.")
            break

        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        stframe.image(frame_rgb)

    cap.release()