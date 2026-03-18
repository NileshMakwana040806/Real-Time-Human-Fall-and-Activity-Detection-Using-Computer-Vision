# main_backend.py

import cv2
import os
import time
from datetime import datetime
from ultralytics import YOLO
from anomaly_detector import detect_activity

model = YOLO("yolov8n-pose.pt")

activity_history = {}
last_logged_activity = {}   # 🔥 NEW (avoid repetition)
fall_counter = {}

last_saved_time = 0

FALL_THRESHOLD = 5
SAVE_INTERVAL = 5

LOG_FILE = "event_log.txt"

# -------- INIT LOG FILE --------
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("=== ACTIVITY LOG START ===\n")


# -------- SAVE LOG --------
def save_log(text):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")
        f.flush()


# -------- SAVE FRAME --------
def save_frame(frame):
    os.makedirs("saved_frames", exist_ok=True)
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"saved_frames/fall_{time_str}.jpg"
    cv2.imwrite(filename, frame)


# -------- PROCESS FRAME --------
def process_frame(frame):

    global activity_history, fall_counter, last_saved_time, last_logged_activity

    results = model.track(frame, persist=True)

    annotated_frame = results[0].plot()
    keypoints = results[0].keypoints
    boxes = results[0].boxes

    dashboard = {
        "total": 0,
        "standing": 0,
        "walking": 0,
        "sitting": 0,
        "fall": 0,
        "unknown": 0
    }

    logs = []

    if keypoints is not None and len(keypoints.xy) > 0:

        dashboard["total"] = len(keypoints.xy)

        for i, points in enumerate(keypoints.xy):

            track_id = int(boxes.id[i]) if boxes.id is not None else i
            activity = detect_activity(points, track_id)

            # -------- HISTORY --------
            if track_id not in activity_history:
                activity_history[track_id] = []

            if len(activity_history[track_id]) == 0 or activity_history[track_id][-1] != activity:
                activity_history[track_id].append(activity)

            # -------- SMART LOGGING --------
            if track_id not in last_logged_activity:
                last_logged_activity[track_id] = None

            if last_logged_activity[track_id] != activity:

                log_text = f"[{datetime.now().strftime('%H:%M:%S')}] ID {track_id} → {activity}"

                logs.append(log_text)
                save_log(log_text)

                last_logged_activity[track_id] = activity

            # -------- FALL LOGIC --------
            if track_id not in fall_counter:
                fall_counter[track_id] = 0

            if activity == "FALL":
                fall_counter[track_id] += 1
            else:
                fall_counter[track_id] = 0

            if fall_counter[track_id] >= FALL_THRESHOLD:

                dashboard["fall"] += 1

                history_str = " → ".join(activity_history[track_id])

                log_text = f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 FALL DETECTED ID {track_id} | {history_str}"

                logs.append(log_text)
                save_log(log_text)

                # SAVE FRAME (SMART)
                current_time = time.time()
                if current_time - last_saved_time > SAVE_INTERVAL:
                    save_frame(frame)
                    last_saved_time = current_time

                fall_counter[track_id] = 0

            # -------- COUNT + COLOR --------
            if activity == "STANDING":
                dashboard["standing"] += 1
                color = (0, 255, 0)
            elif activity == "WALKING":
                dashboard["walking"] += 1
                color = (0, 255, 0)
            elif activity == "SITTING":
                dashboard["sitting"] += 1
                color = (0, 255, 0)
            elif activity == "UNKNOWN":
                dashboard["unknown"] += 1
                color = (255, 0, 0)  # Blue
            elif activity == "FALL":
                color = (0, 0, 255)  # Red
            else:
                color = (0, 255, 0)

            # -------- DRAW --------
            x, y = int(points[0][0]), int(points[0][1])

            cv2.putText(
                annotated_frame,
                f"ID {track_id}: {activity}",
                (x, y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    return annotated_frame, dashboard, logs