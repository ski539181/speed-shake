import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import mediapipe as mp
import cv2

# ส่วนหัวของเว็บ
st.title("🥤 SPEED Shake Campaign")
st.subheader("ขยับมือขึ้น-ลง เพื่อสะสมคะแนน!")

# --- จุดสำคัญ: ใช้ session_state เก็บตัวแปรนับคะแนน ---
if 'total_count' not in st.session_state:
    st.session_state.total_count = 0

class ShakeProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.count = 0
        self.stage = "down"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirror
        
        # แปลงสีให้ AI อ่าน
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_img)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # ใช้พิกัดข้อมือ (Wrist - จุดที่ 0)
                wrist_y = hand_landmarks.landmark[0].y
                
                # ปรับระยะ Threshold ให้กว้างขึ้นเพื่อให้ตรวจจับง่าย (0.3 และ 0.7)
                if wrist_y < 0.3:
                    self.stage = "up"
                if wrist_y > 0.7 and self.stage == "up":
                    self.stage = "down"
                    self.count += 1
                
                # วาดเส้นมือเพื่อให้รู้ว่าระบบยังทำงานอยู่
                mp.solutions.drawing_utils.draw_landmarks(
                    img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# เริ่มต้นระบบกล้อง
ctx = webrtc_streamer(
    key="speed-shake-fix",
    video_processor_factory=ShakeProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# --- ส่วนแสดงคะแนนแบบ Real-time ---
if ctx.video_processor:
    # ดึงคะแนนจากตัวประมวลผลมาแสดงบนหน้าเว็บ
    score = ctx.video_processor.count
    st.markdown(f"<h1 style='text-align: center; color: red; font-size: 80px;'>{score}</h1>", unsafe_allow_html=True)
    
    if st.button("Reset Score"):
        ctx.video_processor.count = 0
else:
    st.info("กรุณากด 'Start' เพื่อเริ่มเปิดกล้องและนับคะแนนครับ")
