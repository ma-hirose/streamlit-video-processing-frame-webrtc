import streamlit as st
import cv2
import numpy as np
import time
from collections import deque

def main():
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        st.error("Cannot open the camera. Check camera connections.")
        return

    # フレーム表示用プレースホルダ
    stframe = st.empty()

    # FPS 計算用
    n = 30  # 直近何フレームで平均を取るか
    time_buffer = deque(maxlen=n)  
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Could not retrieve frame.")
            break

        # 現在時間の取得
        current_time = time.time()
        
        # 1フレーム前との経過時間を追加
        time_buffer.append(current_time - prev_time)
        prev_time = current_time

        # deque に貯めた時間から平均FPSを計算
        if len(time_buffer) > 0:
            avg_time = sum(time_buffer) / len(time_buffer)
            fps = 1.0 / avg_time if avg_time != 0 else 0
        else:
            fps = 0

        # BGR から RGB へ変換
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 画面に画像とFPSを表示
        stframe.image(frame_rgb, caption=f"FPS: {fps:.2f}")

    cap.release()

if __name__ == "__main__":
    main()