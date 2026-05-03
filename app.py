import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import mediapipe as mp
import cv2

# ตั้งค่าหน้าเว็บแบบ Clean & Minimal
st.set_page_config(page_title="SPEED Shake", layout="centered")
st.title("🥤 SPEED Shake Campaign")
st.subheader("ขยับมือขึ้น-ลง เพื่อสะสมคะแนน!")

# ระบบนับคะแนนในหน่วยความจำชั่วคราว
if 'count' not in st.session_state:
    st.session_state.count = 0

class ShakeProcessor(VideoProcessorBase):
    def __init__(self):
        self.hand_tracker = mp.solutions.hands.Hands(
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.7
        )
        self.counter = 0
        self.stage = "down"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirror mode
        
        results = self.hand_tracker.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # ใช้จุด Wrist (0) ในการตรวจจับพิกัด Y
                wrist_y = hand_landmarks.landmark[0].y
                
                # ปรับ Logic การนับให้สมบูรณ์ (Threshold 0.4 และ 0.6)
                if wrist_y < 0.4:
                    self.stage = "up"
                if wrist_y > 0.6 and self.stage == "up":
                    self.stage = "down"
                    self.counter += 1
                
                # วาดจุดเชื่อมต่อบนมือเพื่อให้ผู้เล่นรู้ว่าระบบตรวจจับเจอ
                mp.solutions.drawing_utils.draw_landmarks(
                    img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# เริ่มรันกล้อง
ctx = webrtc_streamer(
    key="speed-shake",
    video_processor_factory=ShakeProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}, # ช่วยให้กล้องติดง่ายขึ้นบนเน็ตมือถือ
    media_stream_constraints={"video": True, "audio": False},
)

# แสดงคะแนน
if ctx.video_processor:
    st.markdown(f"### 🚀 Score: {ctx.video_processor.counter}")
    if st.button("Reset Score"):
        ctx.video_processor.counter = 0
