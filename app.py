# app.py

import streamlit as st
import cv2
import tempfile
import time
from main_backend import process_frame

st.set_page_config(layout="wide")
st.title("Real-Time Human Fall and Activity Detection Using Computer Vision")

# -------- STATE --------
if "mode" not in st.session_state:
    st.session_state.mode = None

if "running" not in st.session_state:
    st.session_state.running = False

if "paused" not in st.session_state:
    st.session_state.paused = False

if "cap" not in st.session_state:
    st.session_state.cap = None

if "logs" not in st.session_state:
    st.session_state.logs = []

if "last_frame" not in st.session_state:
    st.session_state.last_frame = None

if "last_dashboard" not in st.session_state:
    st.session_state.last_dashboard = None
# ----------------------

# -------- INPUT UI --------
if st.session_state.mode is None:

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎥 Start Webcam"):
            st.session_state.mode = "webcam"
            st.session_state.running = True
            st.rerun()

    with col2:
        uploaded = st.file_uploader("📁 Upload Video", type=["mp4", "avi", "mov"])

        if uploaded is not None:
            temp = tempfile.NamedTemporaryFile(delete=False)
            temp.write(uploaded.read())

            st.session_state.mode = "video"
            st.session_state.video_path = temp.name
            st.session_state.running = True
            st.session_state.paused = False

            st.rerun()  # 🔥 IMPORTANT FIX

# -------- OUTPUT UI --------
else:

    col1, col2, col3 = st.columns(3)

    # STOP
    with col1:
        if st.button("⏹ Stop"):
            st.session_state.running = False
            st.session_state.mode = None
            st.session_state.paused = False

            if st.session_state.cap:
                st.session_state.cap.release()
                st.session_state.cap = None

            st.session_state.logs = []
            st.session_state.last_frame = None
            st.session_state.last_dashboard = None

            st.rerun()

    # VIDEO CONTROLS
    if st.session_state.mode == "video":

        with col2:
            if st.button("⏸ Pause"):
                st.session_state.paused = True

        with col3:
            if st.button("▶ Resume"):
                if st.session_state.paused:
                    st.session_state.paused = False

    # LAYOUT
    vcol, dcol, lcol = st.columns([2, 1, 1])

    video = vcol.empty()
    dash = dcol.empty()
    logs_ui = lcol.empty()

    # INIT CAPTURE
    if st.session_state.cap is None:
        if st.session_state.mode == "webcam":
            st.session_state.cap = cv2.VideoCapture(0)
        else:
            st.session_state.cap = cv2.VideoCapture(st.session_state.video_path)

    cap = st.session_state.cap

    while st.session_state.running:

        # PAUSE MODE
        if st.session_state.paused:

            if st.session_state.last_frame is not None:
                video.image(st.session_state.last_frame, channels="BGR")

            if st.session_state.last_dashboard is not None:
                dash.markdown(f"""
### 📊 Dashboard
👥 Total: {st.session_state.last_dashboard['total']}  
🟢 Standing: {st.session_state.last_dashboard['standing']}  
🔵 Walking: {st.session_state.last_dashboard['walking']}  
🟡 Sitting: {st.session_state.last_dashboard['sitting']}  
🔴 Fall: {st.session_state.last_dashboard['fall']}  
⚫ Unknown: {st.session_state.last_dashboard['unknown']}  
""")

            log_html = "<br>".join(st.session_state.logs[-30:])
            logs_ui.markdown(f"""
### 🚨 Activity Logs
<div style='height:300px; overflow-y:auto; border:1px solid gray; padding:10px'>
{log_html}
</div>
""", unsafe_allow_html=True)

            time.sleep(0.1)
            continue

        # NORMAL RUN
        ret, frame = cap.read()

        if not ret:
            break

        frame, dashboard, new_logs = process_frame(frame)

        st.session_state.last_frame = frame
        st.session_state.last_dashboard = dashboard

        video.image(frame, channels="BGR")

        dash.markdown(f"""
### 📊 Dashboard
👥 Total: {dashboard['total']}  
🟢 Standing: {dashboard['standing']}  
🔵 Walking: {dashboard['walking']}  
🟡 Sitting: {dashboard['sitting']}  
🔴 Fall: {dashboard['fall']}  
⚫ Unknown: {dashboard['unknown']}  
""")

        st.session_state.logs.extend(new_logs)

        log_html = "<br>".join(st.session_state.logs[-30:])
        logs_ui.markdown(f"""
### 🚨 Activity Logs
<div style='height:300px; overflow-y:auto; border:1px solid gray; padding:10px'>
{log_html}
</div>
""", unsafe_allow_html=True)

        time.sleep(0.03)

    if cap:
        cap.release()
        st.session_state.cap = None