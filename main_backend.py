"""
main_backend.py — Frame processing pipeline for human activity & fall detection.

"""

import os
import time
from collections import defaultdict
from datetime import datetime
from math import sqrt

import cv2
from ultralytics import YOLO

from activity_detector import ActivityAnalyzer, draw_skeleton, ACTIVITY_COLORS
from event_logger import append_log, write_session_summary

# ─── Model & analyzer ─────────────────────────────────────────────────────────
model             = YOLO("yolov8n-pose.pt")
activity_analyzer = ActivityAnalyzer()

# ─── Constants ────────────────────────────────────────────────────────────────
SAVE_INTERVAL        = 5
MAX_LOST_FRAMES      = 45
IOU_MATCH_THRESHOLD  = 0.25
DIST_MATCH_THRESHOLD = 120

MIN_POSE_CONF = 0.45    # default, overridden by sidebar slider
MIN_BOX_AREA  = 3000    # px² — ignore tiny background detections

# Video-only frame-skip default (webcam always processes every frame)
VIDEO_FRAME_SKIP = 2

# ─── Mutable global state ─────────────────────────────────────────────────────
_tracks          = {}
_next_display_id = 1

last_logged_activity = {}
last_fall_log_time   = defaultdict(float)
last_saved_time      = 0.0

# Session tracking
_session_log_path   = ""
_session_start_time = 0.0
_session_state_totals: dict = {}
_session_max_people = 0
_session_fall_count = 0
_session_fall_times: list = []

# Cache for video frame-skip (stores last REAL dashboard + annotated frame)
_last_annotated_frame = None
_last_real_dashboard  = None


# ─── Session management ───────────────────────────────────────────────────────

def reset_state():
    global _next_display_id, last_saved_time
    global _session_log_path, _session_start_time
    global _session_state_totals, _session_max_people
    global _session_fall_count, _session_fall_times
    global _last_annotated_frame, _last_real_dashboard

    activity_analyzer.reset()
    _tracks.clear()
    last_logged_activity.clear()
    last_fall_log_time.clear()
    _next_display_id        = 1
    last_saved_time         = 0.0
    _session_log_path       = ""
    _session_start_time     = 0.0
    _session_state_totals   = {}
    _session_max_people     = 0
    _session_fall_count     = 0
    _session_fall_times     = []
    _last_annotated_frame   = None
    _last_real_dashboard    = None


def init_session(log_path: str):
    global _session_log_path, _session_start_time
    _session_log_path   = log_path
    _session_start_time = time.time()


def get_session_summary() -> dict:
    duration = time.time() - _session_start_time if _session_start_time else 0
    return {
        "duration_seconds": duration,
        "total_people":     _session_max_people,
        "fall_count":       _session_fall_count,
        "fall_timestamps":  list(_session_fall_times),
        "state_totals":     dict(_session_state_totals),
    }


def flush_session_summary():
    if _session_log_path:
        write_session_summary(_session_log_path, get_session_summary())


# ─── Utilities ────────────────────────────────────────────────────────────────

def save_frame(frame, timestamp_label: str):
    os.makedirs("saved_frames", exist_ok=True)
    safe = timestamp_label.replace(":", "-").replace(".", "-")
    cv2.imwrite(f"saved_frames/fall_{safe}.jpg", frame)


def format_timestamp(seconds: float, source_type: str) -> str:
    if source_type == "video":
        total_ms              = int(max(seconds, 0) * 1000)
        minutes, ms_remaining = divmod(total_ms, 60000)
        secs, ms              = divmod(ms_remaining, 1000)
        hours, minutes        = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
    return datetime.now().strftime("%H:%M:%S")


def _to_list(data):
    if data is None:
        return []
    if hasattr(data, "cpu"):
        data = data.cpu()
    if hasattr(data, "tolist"):
        return data.tolist()
    return list(data)


def _centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _dist(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _iou(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    aA = max((ax2 - ax1) * (ay2 - ay1), 1.0)
    aB = max((bx2 - bx1) * (by2 - by1), 1.0)
    return inter / (aA + aB - inter)


def _box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _avg_keypoint_conf(confidences) -> float:
    if not confidences:
        return 1.0
    valid = [c for c in confidences if c > 0]
    return sum(valid) / max(len(valid), 1)


def _predict_box(track, frames_ahead=1):
    b  = track["box"]
    vx = track["velocity"][0] * frames_ahead
    vy = track["velocity"][1] * frames_ahead
    return [b[0]+vx, b[1]+vy, b[2]+vx, b[3]+vy]


def _update_track(track: dict, box, center, frame_index: int):
    alpha = 0.4
    vx = alpha*(center[0]-track["center"][0]) + (1-alpha)*track["velocity"][0]
    vy = alpha*(center[1]-track["center"][1]) + (1-alpha)*track["velocity"][1]
    track["velocity"]    = (vx, vy)
    track["center"]      = center
    track["box"]         = list(box)
    track["lost_frames"] = 0
    track["last_seen"]   = frame_index


def _allocate_display_id() -> int:
    global _next_display_id
    did = _next_display_id
    _next_display_id += 1
    return did


# ─── Stable ID assignment ─────────────────────────────────────────────────────

def assign_stable_ids(boxes_xyxy: list, model_ids: list, frame_index: int) -> list:
    display_ids     = []
    used_track_keys = set()

    for idx, box in enumerate(boxes_xyxy):
        center   = _centroid(box)
        model_id = model_ids[idx] if idx < len(model_ids) else None

        if model_id is not None and model_id in _tracks:
            track = _tracks[model_id]
            _update_track(track, box, center, frame_index)
            display_ids.append(track["display_id"])
            used_track_keys.add(model_id)
            continue

        best_key   = None
        best_score = -1.0

        for key, track in _tracks.items():
            if key in used_track_keys or track["lost_frames"] == 0:
                continue
            predicted = _predict_box(track, track["lost_frames"])
            iou  = _iou(box, predicted)
            dist = _dist(center, (
                track["center"][0] + track["velocity"][0] * track["lost_frames"],
                track["center"][1] + track["velocity"][1] * track["lost_frames"],
            ))
            score = iou * 2.0 + max(0.0, 1.0 - dist / DIST_MATCH_THRESHOLD)
            if (iou >= IOU_MATCH_THRESHOLD or dist < DIST_MATCH_THRESHOLD) and score > best_score:
                best_score = score
                best_key   = key

        if best_key is not None:
            track = _tracks.pop(best_key)
            new_key = model_id if model_id is not None else best_key
            _tracks[new_key] = track
            _update_track(track, box, center, frame_index)
            used_track_keys.add(new_key)
            display_ids.append(track["display_id"])
            continue

        did = _allocate_display_id()
        key = model_id if model_id is not None else -(did)
        _tracks[key] = {
            "display_id":  did, "center": center, "box": list(box),
            "velocity": (0.0, 0.0), "lost_frames": 0, "last_seen": frame_index,
        }
        used_track_keys.add(key)
        display_ids.append(did)

    for key in list(_tracks.keys()):
        if key not in used_track_keys:
            _tracks[key]["lost_frames"] += 1

    for key in [k for k, t in _tracks.items() if t["lost_frames"] > MAX_LOST_FRAMES]:
        disp = _tracks[key]["display_id"]
        del _tracks[key]
        last_logged_activity.pop(disp, None)
        last_fall_log_time.pop(disp, None)

    return display_ids


# ─── Drawing helpers ──────────────────────────────────────────────────────────

def _draw_box_label(frame, box, activity, display_id, confidence: float):
    """Label shows: ID 1: STANDING 87%"""
    if box is None:
        return
    color = ACTIVITY_COLORS.get(activity, (180, 120, 60))
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    label = f"ID {display_id}: {activity} {int(confidence * 100)}%"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thick = 2
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)
    lx = x1
    ly = max(y1 - th - baseline - 8, 0)
    cv2.rectangle(frame, (lx, ly), (lx + tw + 10, ly + th + baseline + 6), color, -1, cv2.LINE_AA)
    cv2.putText(frame, label, (lx + 5, ly + th + 2), font, scale,
                (255, 255, 255), thick, cv2.LINE_AA)


def _draw_fall_warning(frame, box):
    if box is None:
        return
    x1, y1, x2, y2 = [int(v) for v in box]
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 220), -1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, "! FALL !", (x1 + 4, y2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def _draw_fps_overlay(frame, fps: float, latency_ms: float):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (210, 48), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, f"FPS: {fps:.1f}  Lat: {latency_ms:.0f}ms",
                (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 255, 180), 2, cv2.LINE_AA)


# ─── Main processing entry point ──────────────────────────────────────────────

def process_frame(frame, source_type="webcam", frame_index=0, fps=30.0,
                  min_pose_conf: float = MIN_POSE_CONF,
                  video_frame_skip: int = VIDEO_FRAME_SKIP):
    """
    source_type == "webcam"  → ALWAYS processes every frame (no skip).
    source_type == "video"   → skips YOLO on non-keyframes but returns
                               the LAST real dashboard (no flickering zeros).
    """
    global last_saved_time, _last_annotated_frame, _last_real_dashboard
    global _session_max_people, _session_fall_count, _session_state_totals

    timestamp_seconds = (frame_index / max(fps, 1.0)) if source_type == "video" else time.time()
    timestamp_label   = format_timestamp(timestamp_seconds, source_type)

    # ── VIDEO frame-skip: only skip YOLO, never blank the dashboard ──────────
    if source_type == "video" and video_frame_skip > 1 and frame_index % video_frame_skip != 0:
        if _last_annotated_frame is not None and _last_real_dashboard is not None:
            # Update source time on cached dashboard so clock still ticks
            cached_db = dict(_last_real_dashboard)
            cached_db["source_time"] = timestamp_label
            return _last_annotated_frame.copy(), cached_db, []
        # First frame — fall through and process normally

    # ── Full YOLO processing ──────────────────────────────────────────────────
    started_at = time.perf_counter()

    dashboard = {
        "total": 0, "standing": 0, "walking": 0, "sitting": 0,
        "bending": 0, "fall": 0, "unknown": 0,
        "fps": 0.0, "latency_ms": 0.0,
        "source_time": timestamp_label, "status_message": "",
    }
    logs = []

    try:
        results = model.track(frame, persist=True, verbose=False,
                              tracker="bytetrack.yaml")
        result  = results[0]
        annotated_frame = frame.copy()
    except Exception:
        try:
            results = model.track(frame, persist=True, verbose=False)
            result  = results[0]
            annotated_frame = frame.copy()
        except Exception as exc2:
            annotated_frame = frame.copy()
            dashboard["status_message"] = f"Pose estimation error: {exc2}"
            elapsed = time.perf_counter() - started_at
            dashboard["latency_ms"] = round(elapsed * 1000, 2)
            dashboard["fps"]        = round(1.0 / max(elapsed, 1e-6), 2)
            return annotated_frame, dashboard, logs

    if result.keypoints is not None and len(result.keypoints.xy) > 0:
        keypoints_xy   = _to_list(result.keypoints.xy)
        keypoints_conf = _to_list(result.keypoints.conf if result.keypoints.conf is not None else [])
        boxes_xyxy     = _to_list(result.boxes.xyxy  if result.boxes is not None else [])
        raw_ids        = _to_list(
            result.boxes.id if result.boxes is not None and result.boxes.id is not None else []
        )
        model_ids = []
        for i in range(len(boxes_xyxy)):
            model_ids.append(int(raw_ids[i]) if i < len(raw_ids) and raw_ids[i] is not None else None)

        display_ids = assign_stable_ids(boxes_xyxy, model_ids, frame_index)
        valid_count = 0

        for idx, points in enumerate(keypoints_xy):
            if idx >= len(display_ids):
                continue

            box         = boxes_xyxy[idx]     if idx < len(boxes_xyxy)     else None
            confidences = keypoints_conf[idx] if idx < len(keypoints_conf) else None

            # Filter 1: ignore tiny boxes (background furniture, objects)
            if box is not None and _box_area(box) < MIN_BOX_AREA:
                continue

            # Filter 2: ignore low-confidence pose detections
            avg_conf = _avg_keypoint_conf(confidences) if confidences else 1.0
            if avg_conf < min_pose_conf:
                continue

            valid_count  += 1
            display_id    = display_ids[idx]

            activity, features = activity_analyzer.analyze(
                points, display_id, timestamp_seconds, confidences, box
            )
            decision_conf = features.get("decision_confidence", 0.0)

            # Session state accumulation
            if display_id not in _session_state_totals:
                _session_state_totals[display_id] = defaultdict(int)
            if activity != "UNKNOWN":
                _session_state_totals[display_id][activity] += 1

            # Log state transitions
            if last_logged_activity.get(display_id) != activity and activity != "UNKNOWN":
                log_text = (
                    f"[{timestamp_label}] ID {display_id} -> {activity} "
                    f"({int(decision_conf * 100)}%)"
                )
                logs.append(log_text)
                append_log(_session_log_path, log_text)
                last_logged_activity[display_id] = activity

            # Dashboard counters + visuals
            if activity == "FALL":
                dashboard["fall"] += 1
                _draw_fall_warning(annotated_frame, box)

                if timestamp_seconds - last_fall_log_time[display_id] > 1.0:
                    fall_text = (
                        f"[{timestamp_label}] FALL DETECTED | ID {display_id} | Status: DANGER"
                    )
                    logs.append(fall_text)
                    append_log(_session_log_path, fall_text)
                    last_fall_log_time[display_id] = timestamp_seconds
                    _session_fall_count += 1
                    _session_fall_times.append(timestamp_label)

                    if time.time() - last_saved_time > SAVE_INTERVAL:
                        save_frame(frame, timestamp_label)
                        last_saved_time = time.time()

            elif activity == "STANDING": dashboard["standing"] += 1
            elif activity == "WALKING":  dashboard["walking"]  += 1
            elif activity == "SITTING":  dashboard["sitting"]  += 1
            elif activity == "BENDING":  dashboard["bending"]  += 1
            else:                        dashboard["unknown"]   += 1

            draw_skeleton(annotated_frame, points, confidences, activity)
            _draw_box_label(annotated_frame, box, activity, display_id, decision_conf)

        dashboard["total"] = valid_count
        _session_max_people = max(_session_max_people, valid_count)

    else:
        dashboard["status_message"] = "No person detected in this frame."

    elapsed = time.perf_counter() - started_at
    dashboard["latency_ms"] = round(elapsed * 1000, 2)
    dashboard["fps"]        = round(1.0 / max(elapsed, 1e-6), 2)

    _draw_fps_overlay(annotated_frame, dashboard["fps"], dashboard["latency_ms"])

    # Cache for video frame-skip
    _last_annotated_frame = annotated_frame.copy()
    _last_real_dashboard  = dict(dashboard)

    return annotated_frame, dashboard, logs
