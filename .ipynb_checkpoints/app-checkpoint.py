import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import av #strealing video library

st.title('Streamlit App Test')
st.write('Gray Scale -> Color')

#Class
class VideoProcessor:

    def __init__(self) -> None:
        self.test_state = None

    def recv(self,frame):

        img = frame.to_ndarray(format = 'bgr24')
        if self.test_state == True:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = av.VideoFrame.from_ndarray(img, format='gray')
        else:
            img = av.VideoFrame.from_ndarray(img, format='bgr24')

        return img

ctx = webrtc_streamer(key='example', video_processor_factory=VideoProcessor)

if ctx.video_processor:
    ctx.video_processor.test_state = st.checkbox('Gray Scale -> Color ')