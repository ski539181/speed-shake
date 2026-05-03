import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import mediapipe as mp
import cv2

st.set_page_config(page_title="SPEED Shake", layout="centered")
st.title("🥤 SPEED Shake Campaign")

# ส่วนที่จองไว้สำหรับแสดงคะแนน (อยู่บนสุดเพื่อให้เห็นชัด)
score_placeholder = st.empty()

class ShakeProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.count = 0
        self.prev_y = 0
        self.direction = 0
        self.min_movement = 0.015

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                current_y = hand_landmarks.landmark[0].y
                movement = current_y - self.prev_y
                if abs(movement) > self.min_movement:
                    if movement > 0 and self.direction != -1: self.direction = -1
                    elif movement < 0 and self.direction != 1:
                        self.direction = 1
                        self.count += 1
                self.prev_y = current_y
                mp.solutions.drawing_utils.draw_landmarks(img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ส่วน WebRTC
ctx = webrtc_streamer(
    key="speed-shake-live",
    video_processor_factory=ShakeProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": {"width": 480, "height": 640}, "audio": False},
    async_processing=True,
)

# --- ส่วนสำคัญ: ดึงคะแนนออกมาโชว์โดยไม่ต้องกดปุ่ม ---
# เราจะใช้ Loop สั้นๆ เพื่อดึงค่าจาก ctx.video_processor มาโชว์ตลอดเวลา
if ctx.video_processor:
    while ctx.state.playing:
        score_placeholder.markdown(
            f"<h1 style='text-align: center; color: #FF4B4B; font-size: 150px; margin-top: -50px;'>{ctx.video_processor.count}</h1>", 
            unsafe_allow_html=True
        )
        import time
        time.sleep(0.1) # อัปเดตทุก 0.1 วินาทีเพื่อให้คะแนนลื่นไหล
else:
    score_placeholder.markdown("<h1 style='text-align: center; color: gray;'>0</h1>", unsafe_allow_html=True)

if st.button("Reset Score"):
    if ctx.video_processor:
        ctx.video_processor.count = 0
