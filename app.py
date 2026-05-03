import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import mediapipe as mp

# ตั้งค่าหน้าเว็บให้ดูสะอาดตาแบบ Minimal
st.set_page_config(page_title="SPEED Shake", layout="centered")
st.title("🥤 SPEED Shake Counter")
st.write("เขย่ามือถือหรือขยับมือขึ้น-ลงเพื่อสะสมคะแนน!")

# โหลดโมเดล MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

class ShakeProcessor(VideoProcessorBase):
    def __init__(self):
        self.count = 0
        self.stage = "down"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Flip ภาพเหมือนกระจก
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # ใช้จุดข้อมือ (Wrist) เป็นตัววัด (Index 0)
                wrist_y = hand_landmarks.landmark[0].y
                
                # Logic การนับ ขึ้น-ลง
                if wrist_y < 0.4: self.stage = "up"
                if wrist_y > 0.6 and self.stage == "up":
                    self.stage = "down"
                    self.count += 1
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ส่วนแสดงผลบนหน้าเว็บ
ctx = webrtc_streamer(key="shake", video_processor_factory=ShakeProcessor)

if ctx.video_processor:
    st.header(f"คะแนนปัจจุบัน: {ctx.video_processor.count} ครั้ง")
