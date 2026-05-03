import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import mediapipe as mp
import cv2
import numpy as np

st.set_page_config(page_title="SPEED Shake", layout="centered")
st.title("🥤 SPEED Shake Campaign")
st.subheader("กำปั้นแล้วเขย่ารัวๆ เพื่อสะสมคะแนน!")

mp_hands = mp.solutions.hands

class ShakeProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.count = 0
        self.prev_y = 0
        self.direction = 0 # 1 คือขึ้น, -1 คือลง
        self.min_movement = 0.015 # ระยะขยับขั้นต่ำ (ปรับให้เซนซิทีฟขึ้นได้ที่นี่)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # ใช้จุดข้อมือ (0) เป็นหลักในการตรวจจับการสั่น
                current_y = hand_landmarks.landmark[0].y
                
                # คำนวณส่วนต่างการเคลื่อนที่
                movement = current_y - self.prev_y
                
                # Logic: ถ้าขยับเกินระยะที่กำหนด และเปลี่ยนทิศทาง = นับคะแนน
                if abs(movement) > self.min_movement:
                    if movement > 0 and self.direction != -1: # กำลังลง
                        self.direction = -1
                    elif movement < 0 and self.direction != 1: # กำลังขึ้น
                        self.direction = 1
                        self.count += 1 # นับ 1 เมื่อขยับขึ้นครบ 1 จังหวะ
                
                self.prev_y = current_y
                
                # วาดเส้นจุดเชื่อมต่อให้เห็นการกำมือ
                mp.solutions.drawing_utils.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

ctx = webrtc_streamer(
    key="speed-shake-micro",
    video_processor_factory=ShakeProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={
        "video": {"width": 480, "height": 640, "frameRate": 30}, # เพิ่ม FrameRate ให้ลื่นเพื่อจับการเขย่า
        "audio": False
    },
    async_processing=True,
)

if ctx.video_processor:
    st.markdown(f"<h1 style='text-align: center; color: #FF4B4B; font-size: 120px;'>{ctx.video_processor.count}</h1>", unsafe_allow_html=True)
    if st.button("Reset Score"):
        ctx.video_processor.count = 0
