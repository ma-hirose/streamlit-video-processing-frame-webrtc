import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import random

# フィルタ関数の定義

def random_circles(img: np.ndarray, n_points=2000) -> np.ndarray:
    out_img = img.copy()
    height, width, _ = out_img.shape
    for _ in range(n_points):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        b, g, r = out_img[y, x]
        radius = random.randint(5, 20)
        cv2.circle(out_img, center=(x, y), radius=radius, color=(int(b), int(g), int(r)), thickness=-1)
    return out_img

def some_filter(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edges_3ch

def mosaic_by_resize(img: np.ndarray, factor: int = 5) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // factor, h // factor), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return mosaic

# VideoProcessorクラス
class VideoProcessor:
    def __init__(self):
        self.filter_type = "None"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if self.filter_type == "Random Circles":
            img = random_circles(img)
        elif self.filter_type == "Mosaic":
            img = mosaic_by_resize(img)
        elif self.filter_type == "Edge Detection":
            img = some_filter(img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlitアプリのメイン部分
st.title("Real-time Filter Switcher")

# フィルタ選択のラジオボタン
filter_type = st.radio(
    "Choose a filter:",
    ("None", "Random Circles", "Mosaic", "Edge Detection")
)

# WebRTCストリーマーの設定
ctx = webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

# フィルタの選択をVideoProcessorに反映
if ctx.video_processor:
    ctx.video_processor.filter_type = filter_type