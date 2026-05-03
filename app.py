import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import mediapipe as mp
import cv2

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="SPEED Shake", layout="centered")
st.title("🥤 SPEED Shake Campaign")
st.subheader("ขยับมือขึ้น-ลง เพื่อสะสมคะแนน!")

# ใช้ MediaPipe สำหรับหาจุดบนมือ
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class ShakeProcessor(VideoProcessorBase):
    def __init__(self):
        # สร้างตัวตรวจจับมือไว้ในนี้
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.counter = 0
        self.stage = "down"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # กลับข้างให้เหมือนกระจก
        
        # แปลงสีภาพให้ MediaPipe อ่านออก
        results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # พิกัด Y ของข้อมือ (จุดที่ 0)
                wrist_y = hand_landmarks.landmark[0].y
                
                # Logic การนับ
                if wrist_y < 0.4:
                    self.stage = "up"
                if wrist_y > 0.6 and self.stage == "up":
                    self.stage = "down"
                    self.counter += 1
                
                # วาดเส้นมือโชว์บนจอ
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ส่วนสำคัญ: การเรียกใช้ WebRTC ที่เสถียรที่สุดสำหรับมือถือ
ctx = webrtc_streamer(
    key="speed-shake-final",
    video_processor_factory=ShakeProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True, # เพิ่มความลื่นไหลบนมือถือ
)

if ctx.video_processor:
    st.markdown(f"<h1 style='text-align: center; color: #FF4B4B;'>Score: {ctx.video_processor.counter}</h1>", unsafe_allow_html=True)
    if st.button("เริ่มใหม่ (Reset Score)"):
        ctx.video_processor.counter = 0
