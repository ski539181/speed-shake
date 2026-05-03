import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import mediapipe as mp
import cv2
import time
import requests
import pandas as pd

st.set_page_config(page_title="SPEED Shake World Cup", layout="centered")

# --- ระบบเช็กประเทศจาก IP ---
def get_country():
    try:
        response = requests.get('https://ipapi.co/json/', timeout=5)
        data = response.json()
        return data.get('country_name', 'Unknown'), data.get('country_code', '🏳️')
    except:
        return "Unknown", "🏳️"

# --- ระบบจัดอันดับ (Mockup Database) ---
if 'leaderboard' not in st.session_state:
    st.session_state.leaderboard = []

# --- หน้าแรก: ใส่ชื่อ ---
if 'nickname' not in st.session_state:
    st.title("🥤 SPEED Shake Campaign")
    st.subheader("กรุณาใส่ชื่อเพื่อเริ่มแข่งขัน")
    nick = st.text_input("Your Nickname:", max_chars=15)
    if st.button("Start Game"):
        if nick:
            st.session_state.nickname = nick
            st.session_state.country, st.session_state.flag = get_country()
            st.rerun()
        else:
            st.warning("โปรดใส่ชื่อก่อนเล่นครับ")
    st.stop()

# --- โครงสร้างตัวนับคะแนน ---
class ShakeProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(max_num_hands=1)
        self.counter = 0
        self.stage = "down"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                wrist_y = hand_landmarks.landmark[0].y
                if wrist_y < 0.4: self.stage = "up"
                if wrist_y > 0.6 and self.stage == "up":
                    self.stage = "down"
                    self.counter += 1
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- หน้าเล่นเกม ---
st.title(f"🥤 Player: {st.session_state.nickname} ({st.session_state.flag})")

if 'start_time' not in st.session_state:
    if st.button("🔴 กดเพื่อเริ่มจับเวลา 20 วินาที"):
        st.session_state.start_time = time.time()
        st.rerun()
else:
    elapsed = time.time() - st.session_state.start_time
    time_left = max(0, 20 - int(elapsed))
    
    if time_left > 0:
        st.header(f"⏱️ เวลาที่เหลือ: {time_left} วินาที")
        ctx = webrtc_streamer(
            key="speed-shake-timer",
            video_processor_factory=ShakeProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
        )
        if ctx.video_processor:
            st.session_state.final_score = ctx.video_processor.counter
            st.markdown(f"<h1 style='text-align: center; color: red;'>Score: {st.session_state.final_score}</h1>", unsafe_allow_html=True)
        time.sleep(1)
        st.rerun()
    else:
        st.balloons()
        st.header("🏁 หมดเวลา!")
        final_s = st.session_state.get('final_score', 0)
        st.subheader(f"คะแนนสุดท้ายของคุณคือ: {final_s}")
        
        # บันทึกลงตาราง (ในเซสชั่นนี้)
        new_data = {"Name": st.session_state.nickname, "Country": st.session_state.country, "Score": final_s}
        st.session_state.leaderboard.append(new_data)
        
        if st.button("เล่นอีกครั้ง"):
            del st.session_state.start_time
            st.rerun()

# --- แสดงอันดับ ---
st.write("---")
st.subheader("🏆 Global Leaderboard")
if st.session_state.leaderboard:
    df = pd.DataFrame(st.session_state.leaderboard)
    st.table(df.sort_values(by="Score", ascending=False).head(10))
