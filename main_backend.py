import os
import time
from collections import defaultdict
from datetime import datetime

import cv2
from ultralytics import YOLO

from activity_detector import ActivityAnalyzer


model = YOLO("yolov8n-pose.pt")
activity_analyzer = ActivityAnalyzer()

LOG_FILE = "event_log.txt"
SAVE_INTERVAL = 5
TRACK_MATCH_DISTANCE = 80

track_memory = {}
track_lost_counts = {}
model_track_to_internal = {}
display_id_by_internal = {}
next_internal_track_id = 1
next_display_id = 1
last_logged_activity = {}
last_fall_log_time = defaultdict(float)
last_saved_time = 0.0


if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", encoding="utf-8") as file:
        file.write("=== ACTIVITY LOG START ===\n")


def reset_state():
    global next_internal_track_id, next_display_id, last_saved_time
    activity_analyzer.reset()
    track_memory.clear()
    track_lost_counts.clear()
    model_track_to_internal.clear()
    display_id_by_internal.clear()
    last_logged_activity.clear()
    last_fall_log_time.clear()
    next_internal_track_id = 1
    next_display_id = 1
    last_saved_time = 0.0


def save_log(text):
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(text + "\n")
        file.flush()


def save_frame(frame, timestamp_label):
    os.makedirs("saved_frames", exist_ok=True)
    filename = f"saved_frames/fall_{timestamp_label.replace(':', '-').replace('.', '-')}.jpg"
    cv2.imwrite(filename, frame)


def format_timestamp(seconds, source_type):
    if source_type == "video":
        total_ms = int(max(seconds, 0) * 1000)
        minutes, ms_remaining = divmod(total_ms, 60000)
        secs, ms = divmod(ms_remaining, 1000)
        hours, minutes = divmod(minutes, 60)
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


def _distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _draw_custom_box(frame, box, color):
    if box is None:
        return
    x1, y1, x2, y2 = [int(value) for value in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def _allocate_internal_id():
    global next_internal_track_id
    internal_id = next_internal_track_id
    next_internal_track_id += 1
    return internal_id


def get_display_id(internal_id):
    global next_display_id
    if internal_id not in display_id_by_internal:
        display_id_by_internal[internal_id] = next_display_id
        next_display_id += 1
    return display_id_by_internal[internal_id]


def assign_track_ids(boxes_xyxy, model_ids):
    assigned_ids = []
    active_ids = set()

    for idx, box in enumerate(boxes_xyxy):
        center = _centroid(box)
        resolved_id = None

        if idx < len(model_ids) and model_ids[idx] is not None:
            model_id = int(model_ids[idx])
            if model_id not in model_track_to_internal:
                model_track_to_internal[model_id] = _allocate_internal_id()
            resolved_id = model_track_to_internal[model_id]
        else:
            best_match = None
            best_distance = TRACK_MATCH_DISTANCE
            for existing_id, existing_center in track_memory.items():
                distance = _distance(center, existing_center)
                if distance < best_distance and existing_id not in active_ids:
                    best_match = existing_id
                    best_distance = distance

            if best_match is not None:
                resolved_id = best_match
            else:
                resolved_id = _allocate_internal_id()

        track_memory[resolved_id] = center
        track_lost_counts[resolved_id] = 0
        active_ids.add(resolved_id)
        assigned_ids.append(resolved_id)

    stale_ids = [track_id for track_id in track_memory if track_id not in active_ids]
    for track_id in stale_ids:
        track_lost_counts[track_id] = track_lost_counts.get(track_id, 0) + 1
        if track_lost_counts[track_id] > 30:
            track_memory.pop(track_id, None)
            track_lost_counts.pop(track_id, None)
            last_logged_activity.pop(track_id, None)

    return assigned_ids


def process_frame(frame, source_type="webcam", frame_index=0, fps=30.0):
    global last_saved_time

    started_at = time.perf_counter()
    timestamp_seconds = frame_index / max(fps, 1.0) if source_type == "video" else time.time()
    timestamp_label = format_timestamp(timestamp_seconds, source_type)

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
        "source_time": timestamp_label,
        "status_message": "",
    }
    logs = []

    try:
        results = model.track(frame, persist=True, verbose=False)
        result = results[0]
        annotated_frame = frame.copy()
    except Exception as exc:
        annotated_frame = frame.copy()
        dashboard["status_message"] = f"Pose estimation error: {exc}"
        elapsed = time.perf_counter() - started_at
        dashboard["latency_ms"] = round(elapsed * 1000, 2)
        dashboard["fps"] = round(1.0 / elapsed, 2) if elapsed > 0 else 0.0
        return annotated_frame, dashboard, logs

    if result.keypoints is not None and len(result.keypoints.xy) > 0:
        keypoints_xy = _to_list(result.keypoints.xy)
        keypoints_conf = _to_list(result.keypoints.conf if result.keypoints.conf is not None else [])
        boxes_xyxy = _to_list(result.boxes.xyxy if result.boxes is not None else [])
        model_ids = _to_list(result.boxes.id if result.boxes is not None and result.boxes.id is not None else [])
        track_ids = assign_track_ids(boxes_xyxy, model_ids)

        dashboard["total"] = len(keypoints_xy)

        for idx, points in enumerate(keypoints_xy):
            if idx >= len(track_ids):
                continue

            internal_id = track_ids[idx]
            display_id = get_display_id(internal_id)
            confidences = keypoints_conf[idx] if idx < len(keypoints_conf) else None
            box = boxes_xyxy[idx] if idx < len(boxes_xyxy) else None
            activity, features = activity_analyzer.analyze(points, internal_id, timestamp_seconds, confidences, box)

            if last_logged_activity.get(internal_id) != activity and activity != "UNKNOWN":
                log_text = f"[{timestamp_label}] ID {display_id} -> {activity}"
                logs.append(log_text)
                save_log(log_text)
                last_logged_activity[internal_id] = activity

            if activity == "FALL":
                dashboard["fall"] += 1
                color = (0, 0, 255)
                if timestamp_seconds - last_fall_log_time[internal_id] > 1.0:
                    fall_text = f"[{timestamp_label}] FALL DETECTED | ID {display_id} | Status: DANGER"
                    logs.append(fall_text)
                    save_log(fall_text)
                    last_fall_log_time[internal_id] = timestamp_seconds

                    if time.time() - last_saved_time > SAVE_INTERVAL:
                        save_frame(frame, timestamp_label)
                        last_saved_time = time.time()
            elif activity == "STANDING":
                dashboard["standing"] += 1
                color = (0, 200, 0)
            elif activity == "WALKING":
                dashboard["walking"] += 1
                color = (0, 200, 0)
            elif activity == "SITTING":
                dashboard["sitting"] += 1
                color = (0, 180, 255)
            elif activity == "BENDING":
                dashboard["bending"] += 1
                color = (255, 200, 0)
            else:
                dashboard["unknown"] += 1
                color = (255, 120, 0)

            _draw_custom_box(annotated_frame, box, color)
            label_anchor = (int(features["center"][0]), int(features["shoulder_center"][1]) - 20)
            cv2.putText(
                annotated_frame,
                f"ID {display_id}: {activity}",
                label_anchor,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
    else:
        dashboard["status_message"] = "No person detected in this frame."

    elapsed = time.perf_counter() - started_at
    dashboard["latency_ms"] = round(elapsed * 1000, 2)
    dashboard["fps"] = round(1.0 / elapsed, 2) if elapsed > 0 else 0.0

    return annotated_frame, dashboard, logs
