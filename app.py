"""
app.py — Streamlit UI for Real-Time Human Activity & Fall Detection
Run with: streamlit run app.py

"""

import os
import tempfile
import time

import cv2
import streamlit as st

from main_backend import (
    process_frame, reset_state, init_session,
    flush_session_summary, get_session_summary,
    MIN_POSE_CONF, VIDEO_FRAME_SKIP,
)
from event_logger import create_session_log

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Activity & Fall Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="column"] { overflow: hidden !important; min-width: 0 !important; }

.main-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.2rem; font-weight: 700;
    letter-spacing: 2px; color: #e2e8f0; margin-bottom: 0;
}
.sub-caption {
    font-size: 0.82rem; color: #64748b;
    letter-spacing: 1px; margin-top: 2px; margin-bottom: 16px;
}
.status-danger {
    background: linear-gradient(135deg,#7f1d1d,#991b1b);
    border:1px solid #ef4444; border-radius:8px; padding:10px 16px;
    color:#fca5a5; font-family:'Rajdhani',sans-serif;
    font-size:1.2rem; font-weight:700; letter-spacing:1.5px;
    text-align:center; animation:pulse-danger 1.2s ease-in-out infinite;
}
.status-normal {
    background: linear-gradient(135deg,#052e16,#14532d);
    border:1px solid #22c55e; border-radius:8px; padding:10px 16px;
    color:#86efac; font-family:'Rajdhani',sans-serif;
    font-size:1.2rem; font-weight:700; letter-spacing:1.5px; text-align:center;
}
@keyframes pulse-danger {
    0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,.4);}
    50%{box-shadow:0 0 0 8px rgba(239,68,68,0);}
}
.metric-card {
    background:#1e293b; border:1px solid #334155; border-radius:10px;
    padding:9px 14px; margin-bottom:7px;
    display:flex; justify-content:space-between; align-items:center;
}
.metric-label{font-size:.77rem;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;}
.metric-value{font-family:'Rajdhani',sans-serif;font-size:1.3rem;font-weight:700;color:#e2e8f0;}
.metric-value.danger {color:#f87171;}
.metric-value.success{color:#4ade80;}
.metric-value.info   {color:#60a5fa;}
.metric-value.warn   {color:#fbbf24;}
.metric-value.orange {color:#fb923c;}
.fall-hist{font-size:.73rem;color:#f87171;font-family:'Courier New',monospace;margin-top:3px;}
.log-box{
    height:590px;max-height:600px;overflow-y:auto;
    background:#0f172a;border:1px solid #1e3a5f;
    border-radius:10px;padding:12px 14px;
    font-family:'Courier New',monospace;font-size:.75rem;line-height:1.6;color:#cbd5e1;
}
.log-fall  {color:#f87171;font-weight:bold;}
.log-normal{color:#86efac;}
.log-sit   {color:#60a5fa;}
.log-warn  {color:#fbbf24;}
div.stButton>button{
    border-radius:8px !important;font-family:'Rajdhani',sans-serif !important;
    font-size:1rem !important;font-weight:600 !important;
    letter-spacing:.8px !important;padding:8px 20px !important;
    transition:all .2s ease !important;
}
div.stButton>button:hover{transform:translateY(-1px);}
[data-testid="stImage"]{margin:0 !important;padding:0 !important;line-height:0;overflow:hidden;}
[data-testid="stImage"] img{
    max-width:100% !important;height:auto !important;width:auto !important;
    border-radius:8px;display:block;object-fit:contain;
    box-shadow:0 4px 24px rgba(0,0,0,.5);
}
.section-title{
    font-family:'Rajdhani',sans-serif;font-size:1rem;font-weight:700;
    letter-spacing:2px;color:#475569;text-transform:uppercase;
    border-bottom:1px solid #1e293b;padding-bottom:5px;margin-bottom:10px;
}
.summary-box{
    background:#1e293b;border:1px solid #334155;border-radius:12px;
    padding:18px 22px;margin-bottom:12px;
}
.summary-title{font-family:'Rajdhani',sans-serif;font-size:1.1rem;font-weight:700;color:#94a3b8;}
.summary-val  {font-family:'Rajdhani',sans-serif;font-size:1.8rem;font-weight:700;color:#e2e8f0;}
</style>
""", unsafe_allow_html=True)

# ─── Fall alert: JS beep + Python system bell fallback ───────────────────────
# JS beep uses Web Audio API (works in Chrome/Firefox with user interaction).
# Python fallback: prints \a (ASCII bell) to terminal — audible on most systems.
# Visual flash banner is shown regardless, as guaranteed fallback.

FALL_ALERT_JS = """
<script>
(function(){
try {
    // Unlock AudioContext on first user gesture if needed
    var ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (ctx.state === 'suspended') { ctx.resume(); }
    function beep(f, t, d) {
        var o = ctx.createOscillator(), g = ctx.createGain();
        o.connect(g); g.connect(ctx.destination);
        o.frequency.value = f; o.type = 'square';
        g.gain.setValueAtTime(0.4, ctx.currentTime + t);
        g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + t + d);
        o.start(ctx.currentTime + t);
        o.stop(ctx.currentTime + t + d + 0.05);
    }
    beep(880, 0.00, 0.20);
    beep(660, 0.25, 0.20);
    beep(880, 0.50, 0.20);
} catch(e) {
    // Silent fallback if Web Audio blocked
}
})();
</script>
"""

def _python_beep():
    """System bell — works on Windows, Linux, Mac terminals."""
    import sys
    try:
        import winsound
        winsound.Beep(880, 200)
        winsound.Beep(660, 200)
        winsound.Beep(880, 200)
    except Exception:
        # Cross-platform fallback: ASCII bell character
        print("\a\a\a", end="", flush=True)

DISPLAY_HEIGHT = 660
DISPLAY_MAX_W  = 820


# ─── State helpers ────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "mode": None, "running": False, "paused": False,
        "cap": None, "logs": [], "last_frame": None,
        "last_dashboard": None, "video_path": None,
        "video_filename": "", "frame_index": 0, "source_fps": 30.0,
        "session_log_path": "", "show_summary": False,
        "last_summary": None, "fall_alert_played": False,
        "conf_thresh": MIN_POSE_CONF,
        "video_skip":  VIDEO_FRAME_SKIP,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def release_capture():
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None


def reset_app_state():
    release_capture()
    reset_state()
    st.session_state.running           = False
    st.session_state.paused            = False
    st.session_state.mode              = None
    st.session_state.logs              = []
    st.session_state.last_frame        = None
    st.session_state.last_dashboard    = None
    st.session_state.video_path        = None
    st.session_state.video_filename    = ""
    st.session_state.frame_index       = 0
    st.session_state.source_fps        = 30.0
    st.session_state.session_log_path  = ""
    st.session_state.show_summary      = False
    st.session_state.last_summary      = None
    st.session_state.fall_alert_played = False


def prepare_display_frame(frame):
    """
    Scale frame to fit inside DISPLAY_HEIGHT x DISPLAY_MAX_W box.
    Uses the SMALLER scale factor so BOTH height AND width are always
    within bounds. Portrait/tall videos stay short enough to sit
    beside the dashboard without pushing it off screen.
    Aspect ratio always preserved — no distortion.
    """
    h, w   = frame.shape[:2]
    scale  = min(
        DISPLAY_HEIGHT / max(h, 1),   # don't exceed height
        DISPLAY_MAX_W  / max(w, 1),   # don't exceed width
    )
    new_w  = max(int(w * scale), 1)
    new_h  = max(int(h * scale), 1)
    frame  = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _log_class(t: str) -> str:
    u = t.upper()
    if "FALL"    in u: return "log-fall"
    if "WALKING" in u or "STANDING" in u: return "log-normal"
    if "SITTING" in u: return "log-sit"
    return "log-warn"


# ─── Dashboard renderer ───────────────────────────────────────────────────────

def render_dashboard(slot, db, conf_threshold: float):
    fall_active = db.get("fall", 0) > 0
    status_html = (
        '<div class="status-danger">🚨 DANGER — FALL DETECTED</div>'
        if fall_active else
        '<div class="status-normal">✅ ALL NORMAL</div>'
    )
    if db.get("status_message"):
        status_html += (
            f'<p style="color:#94a3b8;font-size:.76rem;margin-top:5px">'
            f'{db["status_message"]}</p>'
        )

    # Bending hint: only shown when bending is actively detected
    bending_hint = ""
    if db.get("bending", 0) > 0:
        bending_hint = '<div style="font-size:.68rem;color:#64748b;margin:-5px 0 5px 6px;font-style:italic">pre-fall risk signal</div>'

    metrics = f"""
<div class="metric-card"><span class="metric-label">Total People</span><span class="metric-value">{db['total']}</span></div>
<div class="metric-card"><span class="metric-label">🟢 Standing</span><span class="metric-value success">{db['standing']}</span></div>
<div class="metric-card"><span class="metric-label">🟢 Walking</span><span class="metric-value success">{db['walking']}</span></div>
<div class="metric-card"><span class="metric-label">🔵 Sitting</span><span class="metric-value info">{db['sitting']}</span></div>
<div class="metric-card"><span class="metric-label">🟡 Bending</span><span class="metric-value warn">{db['bending']}</span></div>
{bending_hint}
<div class="metric-card"><span class="metric-label">🔴 Falls</span><span class="metric-value danger">{db['fall']}</span></div>
<div class="metric-card"><span class="metric-label">❓ Unknown</span><span class="metric-value">{db['unknown']}</span></div>
<div class="metric-card"><span class="metric-label">⏱ Source Time</span><span class="metric-value" style="font-size:.93rem">{db['source_time']}</span></div>
<div class="metric-card"><span class="metric-label">⚡ FPS</span><span class="metric-value info">{db['fps']}</span></div>
<div class="metric-card"><span class="metric-label">🕐 Latency</span><span class="metric-value">{db['latency_ms']} ms</span></div>
<div class="metric-card"><span class="metric-label">🎯 Conf filter</span><span class="metric-value orange">{int(conf_threshold*100)}%</span></div>
"""
    slot.markdown(
        f'<div class="section-title">Dashboard</div>{status_html}'
        f'<div style="margin-top:8px">{metrics}</div>',
        unsafe_allow_html=True,
    )


def render_logs(slot):
    lines = st.session_state.logs[-60:]
    items = (
        "".join(f'<div class="{_log_class(l)}">{l}</div>' for l in reversed(lines))
        if lines else
        '<div style="color:#475569">No activity logged yet.</div>'
    )
    slot.markdown(
        f'<div class="section-title">Activity Log</div>'
        f'<div class="log-box">{items}</div>',
        unsafe_allow_html=True,
    )


def render_session_summary(summary: dict):
    st.markdown("---")
    st.markdown(
        '<div class="main-title" style="font-size:1.6rem">📊 Session Summary</div>',
        unsafe_allow_html=True,
    )
    dur = summary.get("duration_seconds", 0)
    mins, sec = divmod(int(dur), 60)
    fc = summary.get("fall_count", 0)

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f'<div class="summary-box"><div class="summary-title">Duration</div>'
        f'<div class="summary-val">{mins}m {sec}s</div></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<div class="summary-box"><div class="summary-title">People Seen</div>'
        f'<div class="summary-val">{summary.get("total_people",0)}</div></div>',
        unsafe_allow_html=True,
    )
    fall_color = "#f87171" if fc > 0 else "#4ade80"
    c3.markdown(
        f'<div class="summary-box"><div class="summary-title">Falls Detected</div>'
        f'<div class="summary-val" style="color:{fall_color}">{fc}</div></div>',
        unsafe_allow_html=True,
    )

    if summary.get("fall_timestamps"):
        st.markdown("**Fall timestamps:**")
        for ts in summary["fall_timestamps"]:
            st.markdown(f"- ⚠️ `{ts}`")

    totals = summary.get("state_totals", {})
    if totals:
        st.markdown("**Activity breakdown per person:**")
        for pid, states in sorted(totals.items()):
            total_f = max(sum(states.values()), 1)
            with st.expander(f"Person {pid}", expanded=True):
                for state, cnt in sorted(states.items(), key=lambda x: -x[1]):
                    pct  = cnt / total_f * 100
                    bar  = int(pct / 5)
                    st.markdown(f"`{state:<12}` {'█'*bar}{'░'*(20-bar)} **{pct:.1f}%**")

    log_path = st.session_state.get("session_log_path", "")
    if log_path and os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            log_content = f.read()
        st.download_button(
            label="📥 Download Session Log (.txt)",
            data=log_content,
            file_name=os.path.basename(log_path),
            mime="text/plain",
            use_container_width=True,
        )


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.markdown("---")

        conf_thresh = st.slider(
            "🎯 Min pose confidence",
            min_value=0.20, max_value=0.85,
            value=float(st.session_state.get("conf_thresh", MIN_POSE_CONF)),
            step=0.05,
            help=(
                "Detections with avg keypoint confidence below this are ignored. "
                "Raise it to reduce false background detections. "
                "Lower it if real people are being missed."
            ),
        )
        st.session_state["conf_thresh"] = conf_thresh

        st.markdown("---")
        st.markdown("**📹 Video analysis only**")
        video_skip = st.slider(
            "⚡ Video frame skip",
            min_value=1, max_value=4,
            value=int(st.session_state.get("video_skip", VIDEO_FRAME_SKIP)),
            step=1,
            help=(
                "Video only — webcam always processes every frame.\n\n"
                "1 = every frame (slowest, most accurate).\n"
                "2 = every other frame (recommended).\n"
                "3-4 = faster, slight accuracy drop."
            ),
        )
        st.session_state["video_skip"] = video_skip

        st.markdown("---")
        st.markdown("**Legend**")
        st.markdown("🟢 Standing / Walking")
        st.markdown("🔵 Sitting")
        st.markdown("🟡 Bending = pre-fall risk")
        st.markdown("🔴 FALL = danger")
        st.markdown("---")
        st.markdown("**Log locations**")
        st.caption("`event_log/webcam/` — webcam sessions")
        st.caption("`event_log/video/`  — video analysis")
        st.caption("`saved_frames/`     — fall snapshots")

    return conf_thresh, video_skip


# ─── App entry ────────────────────────────────────────────────────────────────

init_state()
conf_thresh, video_skip = render_sidebar()

st.markdown('<div class="main-title">🛡️ ACTIVITY & FALL DETECTION</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-caption">REAL-TIME POSE ESTIMATION · MULTI-PERSON TRACKING · FALL ALERT</div>',
    unsafe_allow_html=True,
)

# ── Summary screen ────────────────────────────────────────────────────────────
if st.session_state.show_summary and st.session_state.last_summary:
    render_session_summary(st.session_state.last_summary)
    st.markdown("---")
    if st.button("↩ Start New Session", use_container_width=True):
        st.session_state.show_summary = False
        st.session_state.last_summary = None
        st.rerun()
    st.stop()

# ── Landing ───────────────────────────────────────────────────────────────────
if st.session_state.mode is None:
    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.markdown("### 🎥 Webcam")
        st.caption("Live monitoring. Every frame processed — no skip.")
        if st.button("▶ Start Webcam", use_container_width=True):
            reset_app_state()
            cap_test = cv2.VideoCapture(0)
            if not cap_test.isOpened():
                st.error("Webcam could not be opened. Check camera permissions.")
                cap_test.release()
            else:
                ok, _ = cap_test.read()
                cap_test.release()
                if not ok:
                    st.error("Webcam opened but frames could not be read.")
                else:
                    log_path = create_session_log("webcam")
                    st.session_state.mode             = "webcam"
                    st.session_state.running          = True
                    st.session_state.session_log_path = log_path
                    init_session(log_path)
                    st.rerun()

    with right:
        st.markdown("### 📁 Video Upload")
        st.caption("Analyze a recorded video file (mp4, avi, mov, mkv).")
        uploaded_video = st.file_uploader(
            "Choose a video", type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed",
        )
        if uploaded_video is not None:
            if st.button("▶ Analyze Video", use_container_width=True):
                reset_app_state()
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp.write(uploaded_video.read())
                tmp.flush()
                cap_test = cv2.VideoCapture(tmp.name)
                if not cap_test.isOpened():
                    cap_test.release()
                    st.error("Could not open the video file.")
                else:
                    ok, _ = cap_test.read()
                    cap_test.release()
                    if not ok:
                        st.error("Video opened but frames could not be read.")
                    else:
                        log_path = create_session_log("video", uploaded_video.name)
                        st.session_state.mode             = "video"
                        st.session_state.video_path       = tmp.name
                        st.session_state.video_filename   = uploaded_video.name
                        st.session_state.running          = True
                        st.session_state.session_log_path = log_path
                        init_session(log_path)
                        st.rerun()

# ── Active monitoring ─────────────────────────────────────────────────────────
else:
    ctrl = st.columns(4)
    with ctrl[0]:
        if st.button("⏹ Stop", use_container_width=True):
            flush_session_summary()
            st.session_state.last_summary = get_session_summary()
            st.session_state.show_summary = True
            reset_app_state()
            st.rerun()

    if st.session_state.mode == "video":
        with ctrl[1]:
            if st.button("⏸ Pause", use_container_width=True):
                st.session_state.paused = True
        with ctrl[2]:
            if st.button("▶ Resume", use_container_width=True):
                st.session_state.paused = False

    mode_label = "🎥 WEBCAM — LIVE" if st.session_state.mode == "webcam" else "📁 VIDEO FILE"
    ctrl[3].markdown(
        f'<div style="text-align:right;color:#475569;font-size:.83rem;padding-top:8px">'
        f'{mode_label}</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.session_log_path:
        st.caption(f"📝 Logging → `{st.session_state.session_log_path}`")

    st.markdown('<hr style="border-color:#1e293b;margin:6px 0 10px">', unsafe_allow_html=True)

    vid_col, dash_col, log_col = st.columns([3, 2, 2])
    video_slot     = vid_col.empty()
    dashboard_slot = dash_col.empty()
    log_slot       = log_col.empty()
    alert_slot     = st.empty()

    if st.session_state.cap is None:
        if st.session_state.mode == "webcam":
            st.session_state.cap        = cv2.VideoCapture(0)
            st.session_state.source_fps = 30.0
        else:
            st.session_state.cap = cv2.VideoCapture(st.session_state.video_path)
            detected = st.session_state.cap.get(cv2.CAP_PROP_FPS)
            st.session_state.source_fps = detected if detected and detected > 1 else 30.0

        if not st.session_state.cap.isOpened():
            st.error("Input source could not be initialized.")
            reset_app_state()
            st.stop()

    cap = st.session_state.cap

    while st.session_state.running:

        if st.session_state.paused:
            if st.session_state.last_frame is not None:
                video_slot.image(st.session_state.last_frame)
            if st.session_state.last_dashboard is not None:
                render_dashboard(dashboard_slot, st.session_state.last_dashboard, conf_thresh,
                )
            render_logs(log_slot)
            time.sleep(0.1)
            continue

        ok, frame = cap.read()
        if not ok or frame is None:
            if st.session_state.mode == "video":
                flush_session_summary()
                st.session_state.last_summary = get_session_summary()
                st.session_state.show_summary = True
                reset_app_state()
                st.rerun()
            else:
                st.warning("⚠️ Could not read frame from webcam.")
            st.session_state.running = False
            break

        if frame.mean() < 8:
            dark_db = {
                "total": 0, "standing": 0, "walking": 0, "sitting": 0,
                "bending": 0, "fall": 0, "unknown": 0,
                "fps": 0.0, "latency_ms": 0.0, "source_time": "N/A",
                "status_message": "Frame too dark — camera may be covered.",
            }
            display = prepare_display_frame(frame)
            st.session_state.last_frame     = display
            st.session_state.last_dashboard = dark_db
            video_slot.image(display)
            render_dashboard(dashboard_slot, dark_db, conf_thresh)
            render_logs(log_slot)
            time.sleep(0.05)
            continue

        annotated, dashboard, new_logs = process_frame(
            frame,
            source_type=st.session_state.mode,
            frame_index=st.session_state.frame_index,
            fps=st.session_state.source_fps,
            min_pose_conf=conf_thresh,
            video_frame_skip=video_skip,
        )

        st.session_state.frame_index += 1
        display = prepare_display_frame(annotated)
        st.session_state.last_frame     = display
        st.session_state.last_dashboard = dashboard
        st.session_state.logs.extend(new_logs)

        # Fall alert — JS beep + Python bell + visual banner
        if dashboard.get("fall", 0) > 0:
            if not st.session_state.fall_alert_played:
                alert_slot.markdown(FALL_ALERT_JS, unsafe_allow_html=True)
                _python_beep()   # system bell as reliable fallback
                st.session_state.fall_alert_played = True
        else:
            if st.session_state.fall_alert_played:
                alert_slot.empty()
            st.session_state.fall_alert_played = False

        video_slot.image(display)
        render_dashboard(dashboard_slot, dashboard, conf_thresh,
        )
        render_logs(log_slot)

        if st.session_state.mode == "video":
            frame_delay = 1.0 / max(st.session_state.source_fps, 1.0)
            remaining   = max(frame_delay - dashboard["latency_ms"] / 1000.0, 0.0)
            if remaining > 0:
                time.sleep(remaining)
        else:
            time.sleep(0.01)

    release_capture()
