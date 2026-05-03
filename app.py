import streamlit as st

# ตั้งค่าหน้าเว็บไว้บนสุด
st.set_page_config(page_title="SPEED Shake", layout="centered")
st.title("🥤 SPEED Shake Campaign")
st.subheader("ขยับมือขึ้น-ลง เพื่อสะสมคะแนน!")

# ตรวจสอบการนำเข้า Library ทีละขั้นตอน
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
    import mediapipe as mp
    import av
    import cv2
except ImportError:
    st.error("กำลังติดตั้งระบบเพิ่มเติม... กรุณารอ 1-2 นาทีแล้ว Refresh อีกครั้ง")
    st.stop()

class ShakeProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.counter = 0
        self.stage = "down"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                wrist_y = hand_landmarks.landmark[0].y
                if wrist_y < 0.4:
                    self.stage = "up"
                if wrist_y > 0.6 and self.stage == "up":
                    self.stage = "down"
                    self.counter += 1
                mp.solutions.drawing_utils.draw_landmarks(
                    img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

ctx = webrtc_streamer(
    key="speed-shake-v3",
    video_processor_factory=ShakeProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

if ctx.video_processor:
    st.markdown(f"<h1 style='text-align: center;'>Score: {ctx.video_processor.counter}</h1>", unsafe_allow_html=True)
