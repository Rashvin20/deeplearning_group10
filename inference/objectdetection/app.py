import streamlit as st
import torch
import av
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


WEIGHTS_PATH = "best.pt"
CONF_THRESH = 0.25

device = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(layout="centered")
st.title("🍎 Live Food Detection (YOLOv8)")
st.write("YOLOv8 • OpenImages-trained • Real-time")


@st.cache_resource
def load_model():
    model = YOLO(WEIGHTS_PATH)
    model.to(device)
    model.fuse()  # speed + stability
    return model

model = load_model()

st.success("Model loaded successfully")

class YOLOVideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        with torch.no_grad():
            results = model(
                img,
                conf=CONF_THRESH,
                device=device,
                verbose=False
            )

        annotated = results[0].plot()  # draws boxes + labels safely

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


webrtc_streamer(
    key="yolo-live",
    video_processor_factory=YOLOVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
