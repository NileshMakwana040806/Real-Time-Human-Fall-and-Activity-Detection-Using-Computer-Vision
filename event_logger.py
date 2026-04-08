"""
event_logger.py — Organised session logging system.

Log folder structure:
    event_log/
        webcam/
            webcam_2026-03-24_19-30-00.txt
            webcam_2026-03-25_10-15-42.txt
        video/
            my_video_2026-03-24_20-00-00.txt
            hospital_cctv_2026-03-25_09-00-00.txt

"""

import os
from datetime import datetime


# ─── Base directory ───────────────────────────────────────────────────────────
LOG_BASE = "event_log"


def _safe_name(text: str) -> str:
    """Strip characters that are unsafe in filenames."""
    keep = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")
    return "".join(c if c in keep else "_" for c in text)


def create_session_log(source_type: str, video_filename: str = "") -> str:
    """
    Create a new log file for this session and return its full path.

    source_type   : "webcam" or "video"
    video_filename: original uploaded filename (used in log name for video sessions)

    Returns the absolute path of the created log file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if source_type == "webcam":
        folder = os.path.join(LOG_BASE, "webcam")
        filename = f"webcam_{timestamp}.txt"
    else:
        folder = os.path.join(LOG_BASE, "video")
        if video_filename:
            # Remove extension, keep base name
            base = os.path.splitext(os.path.basename(video_filename))[0]
            base = _safe_name(base)[:40]   # cap length
            filename = f"{base}_{timestamp}.txt"
        else:
            filename = f"video_{timestamp}.txt"

    os.makedirs(folder, exist_ok=True)
    log_path = os.path.join(folder, filename)

    # Write session header
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"  SESSION TYPE  : {source_type.upper()}\n")
        if source_type == "video" and video_filename:
            f.write(f"  VIDEO FILE    : {os.path.basename(video_filename)}\n")
        f.write(f"  DATE & TIME   : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

    return log_path


def append_log(log_path: str, text: str):
    """Append a line to the session log file."""
    if not log_path:
        return
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")
            f.flush()
    except Exception:
        pass   # never crash the main pipeline because of logging


def write_session_summary(log_path: str, summary: dict):
    """
    Append a human-readable summary block at the end of the session log.

    summary keys expected:
        duration_seconds, total_people, state_totals (dict),
        fall_count, fall_timestamps (list)
    """
    if not log_path:
        return
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write("  SESSION SUMMARY\n")
            f.write("=" * 60 + "\n")
            dur = summary.get("duration_seconds", 0)
            mins, secs = divmod(int(dur), 60)
            f.write(f"  Duration          : {mins}m {secs}s\n")
            f.write(f"  Total people seen : {summary.get('total_people', 0)}\n")
            f.write(f"  Falls detected    : {summary.get('fall_count', 0)}\n")
            if summary.get("fall_timestamps"):
                for ts in summary["fall_timestamps"]:
                    f.write(f"    -> Fall at {ts}\n")
            totals = summary.get("state_totals", {})
            if totals:
                f.write("  Activity breakdown per person:\n")
                for pid, states in totals.items():
                    f.write(f"    Person {pid}:\n")
                    total_frames = max(sum(states.values()), 1)
                    for state, count in sorted(states.items()):
                        pct = count / total_frames * 100
                        f.write(f"      {state:<12}: {pct:5.1f}%\n")
            f.write("=" * 60 + "\n")
    except Exception:
        pass
