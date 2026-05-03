import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import mediapipe as mp
import cv2

# ตั้งค่าหน้าเว็บให้เบาที่สุด
st.set_page_config(page_title="SPEED Shake", layout="centered")
st.title("🥤 SPEED Shake Campaign")

# โหลด MediaPipe ไว้ข้างนอกเพื่อประหยัด RAM
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class ShakeProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0, # 0 = เร็วที่สุด (เหมาะกับมือถือและแก้ดีเลย์)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.count = 0
        self.stage = "down"
        self.frame_count = 0 

    def recv(self, frame):
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        # --- แก้ดีเลย์: ประมวลผล AI 1 เฟรม เว้น 1 เฟรม ---
        if self.frame_count % 2 == 0:
            results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # ใช้จุดที่ 9 (กลางฝ่ามือ) จะนิ่งกว่าข้อมือในระยะใกล้
                    hand_center_y = hand_landmarks.landmark[9].y
                    
                    # ปรับจังหวะนับให้กว้างขึ้นตามระยะมือในรูปเดิมของคุณ
                    if hand_center_y < 0.48: 
                        self.stage = "up"
                    if hand_center_y > 0.52 and self.stage == "up":
                        self.stage = "down"
                        self.count += 1
                    
                    # วาดเส้นเฉพาะตอนประมวลผลเพื่อลดการใช้ CPU
                    mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ปรับปรุงการเชื่อมต่อ WebRTC ให้ลื่นขึ้น
ctx = webrtc_streamer(
    key="speed-shake-optimized",
    video_processor_factory=ShakeProcessor,
    # iceServers ช่วยให้การส่งข้อมูลข้ามเครือข่ายมือถือไม่ดีเลย์
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"], "urls": ["stun:stun1.l.google.com:19302"]}]
    },
    media_stream_constraints={
        "video": {
            "width": {"ideal": 480}, # ลดความละเอียดภาพลงเพื่อให้ประมวลผลไวขึ้น
            "height": {"ideal": 640},
            "frameRate": {"ideal": 20}
        },
        "audio": False
    },
    async_processing=True, # สำคัญ: แยกการวาดภาพกับการประมวลผลออกจากกัน
)

if ctx.video_processor:
    # ใช้พื้นที่ HTML เพื่อแสดงคะแนนให้ไวที่สุด
    score_place = st.empty()
    score_place.markdown(f"<h1 style='text-align: center; color: red; font-size: 100px;'>{ctx.video_processor.count}</h1>", unsafe_allow_html=True)
    
    if st.button("Reset Score"):
        ctx.video_processor.count = 0
