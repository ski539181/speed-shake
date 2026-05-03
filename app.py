import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import mediapipe as mp
import cv2

# ส่วนหัว
st.title("🥤 SPEED Shake Campaign")
st.subheader("ขยับมือขึ้น-ลง เพื่อสะสมคะแนน!")

class ShakeProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.count = 0
        self.stage = "down"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # ใช้จุดข้อมือ (0) หรือกลางฝ่ามือ (9) ก็ได้ครับ
                wrist_y = hand_landmarks.landmark[0].y
                
                # --- ปรับค่าใหม่ตามรูปของคุณ ---
                # ในรูปมือคุณอยู่ประมาณกลางจอ (0.5) 
                # เราจะตั้งให้จุดสูงสุดที่ 0.45 และต่ำสุดที่ 0.55 เพื่อให้นับง่ายขึ้น
                if wrist_y < 0.45: # เมื่อมือยกขึ้นสูงกว่าระดับอก
                    self.stage = "up"
                
                if wrist_y > 0.55 and self.stage == "up": # เมื่อมือลดลงต่ำกว่าเดิม
                    self.stage = "down"
                    self.count += 1
                
                # วาดเส้นเพื่อเช็คว่า AI ยังจับมือเราอยู่ไหม
                mp.solutions.drawing_utils.draw_landmarks(
                    img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

ctx = webrtc_streamer(
    key="speed-shake-v4",
    video_processor_factory=ShakeProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

if ctx.video_processor:
    # แสดงคะแนนตัวใหญ่ๆ กลางจอ
    st.markdown(f"<h1 style='text-align: center; color: red; font-size: 100px;'>{ctx.video_processor.count}</h1>", unsafe_allow_html=True)
    if st.button("Reset Score"):
        ctx.video_processor.count = 0
