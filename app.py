import tempfile
import time

import cv2
import streamlit as st

from main_backend import process_frame, reset_state


st.set_page_config(layout="wide")
st.title("Real-Time Human Activity and Fall Detection")
st.caption("Temporal activity analysis with stabilized states, motion features, and fall-risk monitoring.")


def init_state():
    defaults = {
        "mode": None,
        "running": False,
        "paused": False,
        "cap": None,
        "logs": [],
        "last_frame": None,
        "last_dashboard": None,
        "video_path": None,
        "frame_index": 0,
        "source_fps": 30.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def release_capture():
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None


def reset_app_state():
    release_capture()
    reset_state()
    st.session_state.running = False
    st.session_state.paused = False
    st.session_state.mode = None
    st.session_state.logs = []
    st.session_state.last_frame = None
    st.session_state.last_dashboard = None
    st.session_state.video_path = None
    st.session_state.frame_index = 0
    st.session_state.source_fps = 30.0


def fit_frame_for_display(frame, max_height=620, max_width=980):
    height, width = frame.shape[:2]
    scale = min(max_height / max(height, 1), max_width / max(width, 1), 1.0)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def render_dashboard(target, dashboard):
    status_line = ""
    if dashboard.get("status_message"):
        status_line = f"Status: `{dashboard['status_message']}`  \n"
    target.markdown(
        f"""
### Monitoring Dashboard
{status_line}Source Time: `{dashboard['source_time']}`  
People Detected: `{dashboard['total']}`  
Standing: `{dashboard['standing']}`  
Walking: `{dashboard['walking']}`  
Sitting: `{dashboard['sitting']}`  
Bending: `{dashboard['bending']}`  
Falls: `{dashboard['fall']}`  
Unknown: `{dashboard['unknown']}`  
FPS: `{dashboard['fps']}`  
Latency: `{dashboard['latency_ms']} ms`
"""
    )


def render_logs(target):
    log_html = "<br>".join(st.session_state.logs[-40:]) if st.session_state.logs else "No activity logged yet."
    target.markdown(
        f"""
### Activity Log
<div style='height:340px; overflow-y:auto; border:1px solid #888; border-radius:8px; padding:12px; background:#0f172a; color:#e2e8f0'>
{log_html}
</div>
""",
        unsafe_allow_html=True,
    )


init_state()
if st.session_state.mode is None:
    source_choice = st.radio("Select input source", ["Webcam", "Video Upload"], horizontal=True)

    if source_choice == "Webcam":
        if st.button("Start Monitoring"):
            reset_app_state()
            test_cap = cv2.VideoCapture(0)
            if not test_cap.isOpened():
                st.error("Webcam could not be opened. Check camera permission or whether another app is using it.")
                test_cap.release()
            else:
                ok, test_frame = test_cap.read()
                test_cap.release()
                if not ok or test_frame is None:
                    st.error("Webcam opened, but frames could not be read. The camera may be blocked or unavailable.")
                else:
                    st.session_state.mode = "webcam"
                    st.session_state.running = True
                    st.rerun()
    else:
        uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_video is not None and st.button("Analyze Video"):
            reset_app_state()
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file.write(uploaded_video.read())
            temp_file.flush()
            temp_cap = cv2.VideoCapture(temp_file.name)
            if not temp_cap.isOpened():
                temp_cap.release()
                st.error("Uploaded file could not be opened as a video. Please upload a valid video file.")
            else:
                ok, _ = temp_cap.read()
                temp_cap.release()
                if not ok:
                    st.error("Video file opened but frames could not be read. Please try another video.")
                else:
                    st.session_state.mode = "video"
                    st.session_state.video_path = temp_file.name
                    st.session_state.running = True
                    st.rerun()
else:
    control_col1, control_col2, control_col3 = st.columns(3)
    with control_col1:
        if st.button("Stop"):
            reset_app_state()
            st.rerun()

    if st.session_state.mode == "video":
        with control_col2:
            if st.button("Pause"):
                st.session_state.paused = True
        with control_col3:
            if st.button("Resume"):
                st.session_state.paused = False

    view_col, dashboard_col, log_col = st.columns([3.2, 1.2, 1.4])
    video_slot = view_col.empty()
    dashboard_slot = dashboard_col.empty()
    log_slot = log_col.empty()

    if st.session_state.cap is None:
        if st.session_state.mode == "webcam":
            st.session_state.cap = cv2.VideoCapture(0)
            st.session_state.source_fps = 30.0
        else:
            st.session_state.cap = cv2.VideoCapture(st.session_state.video_path)
            detected_fps = st.session_state.cap.get(cv2.CAP_PROP_FPS)
            st.session_state.source_fps = detected_fps if detected_fps and detected_fps > 1 else 30.0

        if st.session_state.cap is None or not st.session_state.cap.isOpened():
            st.error("Input source could not be initialized.")
            reset_app_state()
            st.stop()

    capture = st.session_state.cap

    while st.session_state.running:
        if st.session_state.paused:
            if st.session_state.last_frame is not None:
                video_slot.image(st.session_state.last_frame, channels="BGR")
            if st.session_state.last_dashboard is not None:
                render_dashboard(dashboard_slot, st.session_state.last_dashboard)
            render_logs(log_slot)
            time.sleep(0.1)
            continue

        success, frame = capture.read()
        if not success:
            st.warning("Video ended or the current frame could not be read.")
            st.session_state.running = False
            break

        if frame is None:
            st.warning("Empty frame received from the source.")
            st.session_state.running = False
            break

        if frame.mean() < 8:
            dashboard = {
                "total": 0,
                "standing": 0,
                "walking": 0,
                "sitting": 0,
                "bending": 0,
                "fall": 0,
                "unknown": 0,
                "fps": 0.0,
                "latency_ms": 0.0,
                "source_time": "N/A",
                "status_message": "Frame is too dark. The camera may be covered or the scene is not visible.",
            }
            st.session_state.last_frame = fit_frame_for_display(frame)
            st.session_state.last_dashboard = dashboard
            video_slot.image(st.session_state.last_frame, channels="BGR")
            render_dashboard(dashboard_slot, dashboard)
            render_logs(log_slot)
            time.sleep(0.05)
            continue

        annotated_frame, dashboard, new_logs = process_frame(
            frame,
            source_type=st.session_state.mode,
            frame_index=st.session_state.frame_index,
            fps=st.session_state.source_fps,
        )

        st.session_state.frame_index += 1
        display_frame = fit_frame_for_display(annotated_frame)
        st.session_state.last_frame = display_frame
        st.session_state.last_dashboard = dashboard
        st.session_state.logs.extend(new_logs)

        video_slot.image(display_frame, channels="BGR")
        render_dashboard(dashboard_slot, dashboard)
        render_logs(log_slot)

        if st.session_state.mode == "video":
            frame_delay = 1.0 / max(st.session_state.source_fps, 1.0)
            remaining_delay = max(frame_delay - (dashboard["latency_ms"] / 1000.0), 0.0)
            if remaining_delay > 0:
                time.sleep(remaining_delay)
        else:
            time.sleep(0.01)

    release_capture()
