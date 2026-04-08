"""
Activity Detector — Temporal pose-based human activity classifier.
Classifies: STANDING, WALKING, SITTING, BENDING, FALL, UNKNOWN
"""

from collections import Counter, deque
from math import atan2, degrees, sqrt

import cv2

# ─── COCO keypoint indices ────────────────────────────────────────────────────
KEYPOINT_NAMES = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# Skeleton connections for drawing (pairs of keypoint indices)
SKELETON_CONNECTIONS = [
    (5, 6),   # shoulders
    (5, 7),   # left shoulder-elbow
    (7, 9),   # left elbow-wrist
    (6, 8),   # right shoulder-elbow
    (8, 10),  # right elbow-wrist
    (5, 11),  # left shoulder-hip
    (6, 12),  # right shoulder-hip
    (11, 12), # hips
    (11, 13), # left hip-knee
    (13, 15), # left knee-ankle
    (12, 14), # right hip-knee
    (14, 16), # right knee-ankle
]

# Activity colors (BGR)
ACTIVITY_COLORS = {
    "STANDING": (0, 220, 80),
    "WALKING":  (0, 200, 20),
    "SITTING":  (0, 180, 255),
    "BENDING":  (255, 200, 0),
    "FALL":     (0, 0, 255),
    "UNKNOWN":  (180, 120, 60),
}


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def _point(points, name):
    idx = KEYPOINT_NAMES[name]
    x, y = points[idx]
    return float(x), float(y)


def _confidence(confidences, name):
    if not confidences:
        return 1.0
    idx = KEYPOINT_NAMES[name]
    if idx >= len(confidences):
        return 0.0
    return float(confidences[idx])


def _midpoint(a, b):
    return (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0


def _distance(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _angle_deg(a, b, c):
    """Angle at vertex b formed by a–b–c, in degrees."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag = max(_distance(a, b) * _distance(b, c), 1e-6)
    cosv = max(-1.0, min(1.0, dot / mag))
    import math
    return math.degrees(math.acos(cosv))


# ─── Skeleton overlay ─────────────────────────────────────────────────────────

def draw_skeleton(frame, points, confidences, activity, conf_threshold=0.3):
    """Draw pose skeleton on frame with activity-appropriate colour."""
    color = ACTIVITY_COLORS.get(activity, (200, 200, 200))
    joint_color = tuple(min(255, int(c * 1.3)) for c in color)

    # Draw limb connections
    for start_idx, end_idx in SKELETON_CONNECTIONS:
        if start_idx >= len(points) or end_idx >= len(points):
            continue
        c_start = float(confidences[start_idx]) if confidences and start_idx < len(confidences) else 1.0
        c_end   = float(confidences[end_idx])   if confidences and end_idx   < len(confidences) else 1.0
        if c_start < conf_threshold or c_end < conf_threshold:
            continue
        x1, y1 = int(points[start_idx][0]), int(points[start_idx][1])
        x2, y2 = int(points[end_idx][0]),   int(points[end_idx][1])
        if x1 == 0 and y1 == 0:
            continue
        if x2 == 0 and y2 == 0:
            continue
        cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # Draw joints
    for idx, (x, y) in enumerate(points):
        conf = float(confidences[idx]) if confidences and idx < len(confidences) else 1.0
        if conf < conf_threshold:
            continue
        ix, iy = int(x), int(y)
        if ix == 0 and iy == 0:
            continue
        cv2.circle(frame, (ix, iy), 4, joint_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (ix, iy), 4, (255, 255, 255), 1, cv2.LINE_AA)


# ─── Core analyzer ────────────────────────────────────────────────────────────

class ActivityAnalyzer:
    def __init__(self, history_size=30, state_window=12):
        self.history_size = history_size
        self.state_window = state_window
        self.people = {}

    def reset(self):
        self.people.clear()

    def analyze(self, points, track_id, timestamp_seconds, confidences=None, box=None):
        features = self._extract_features(points, timestamp_seconds, confidences, box)
        person = self.people.setdefault(
            track_id,
            {
                "features": deque(maxlen=self.history_size),
                "raw_states": deque(maxlen=self.state_window),
                "stable_state": "UNKNOWN",
            },
        )

        previous = person["features"][-1] if person["features"] else None
        self._update_temporal_features(features, previous)
        person["features"].append(features)

        raw_state, confidence = self._classify_raw(person["features"], person["stable_state"])
        features["decision_confidence"] = confidence
        person["raw_states"].append(raw_state)

        stable_state = self._stabilize_state(person)
        person["stable_state"] = stable_state
        return stable_state, features

    # ── Feature extraction ────────────────────────────────────────────────────

    def _extract_features(self, points, timestamp_seconds, confidences=None, box=None):
        left_shoulder  = _point(points, "left_shoulder")
        right_shoulder = _point(points, "right_shoulder")
        left_hip       = _point(points, "left_hip")
        right_hip      = _point(points, "right_hip")
        left_knee      = _point(points, "left_knee")
        right_knee     = _point(points, "right_knee")
        left_ankle     = _point(points, "left_ankle")
        right_ankle    = _point(points, "right_ankle")

        shoulder_center = _midpoint(left_shoulder, right_shoulder)
        hip_center      = _midpoint(left_hip, right_hip)
        knee_center     = _midpoint(left_knee, right_knee)
        ankle_center    = _midpoint(left_ankle, right_ankle)

        full_height    = max(_distance(shoulder_center, ankle_center), 1.0)
        vertical_span  = max(abs(ankle_center[1] - shoulder_center[1]), 1.0)
        shoulder_width = max(_distance(left_shoulder, right_shoulder), 1.0)

        body_axis_angle = abs(
            degrees(atan2(hip_center[1] - shoulder_center[1],
                          hip_center[0] - shoulder_center[0]))
        )
        orientation_from_vertical = abs(90.0 - body_axis_angle)

        # Hip-knee-ankle angle for sitting detection
        try:
            hip_knee_ankle_angle = _angle_deg(hip_center, knee_center, ankle_center)
        except Exception:
            hip_knee_ankle_angle = 180.0

        if box is None:
            x1, y1, x2, y2 = 0.0, 0.0, shoulder_width, full_height
        else:
            x1, y1, x2, y2 = [float(v) for v in box]
        box_width  = max(x2 - x1, 1.0)
        box_height = max(y2 - y1, 1.0)

        hip_conf      = min(_confidence(confidences, "left_hip"),      _confidence(confidences, "right_hip"))
        ankle_conf    = min(_confidence(confidences, "left_ankle"),    _confidence(confidences, "right_ankle"))
        shoulder_conf = min(_confidence(confidences, "left_shoulder"), _confidence(confidences, "right_shoulder"))
        knee_conf     = min(_confidence(confidences, "left_knee"),     _confidence(confidences, "right_knee"))

        # Left/right ankle positions for spread variation (gait signature)
        left_ankle_pt  = _point(points, "left_ankle")
        right_ankle_pt = _point(points, "right_ankle")
        ankle_spread   = _distance(left_ankle_pt, right_ankle_pt)

        return {
            "center":                  hip_center,
            "hip_center":              hip_center,
            "shoulder_center":         shoulder_center,
            "ankle_center":            ankle_center,
            "knee_center":             knee_center,
            "left_ankle":              left_ankle_pt,
            "right_ankle":             right_ankle_pt,
            "ankle_spread":            ankle_spread,
            "full_height":             full_height,
            "vertical_span_ratio":     vertical_span / full_height,
            "orientation_from_vertical": orientation_from_vertical,
            "box_aspect_ratio":        box_height / box_width,
            "hip_knee_ankle_angle":    hip_knee_ankle_angle,
            "shoulders_visible":       shoulder_conf >= 0.35,
            "hips_visible":            hip_conf      >= 0.35,
            "ankles_visible":          ankle_conf    >= 0.25,
            "knees_visible":           knee_conf     >= 0.30,
            "center_velocity":         0.0,
            "horizontal_speed":        0.0,   # |dx| only — key for walking
            "ankle_velocity":          0.0,
            "ankle_spread_delta":      0.0,   # change in ankle separation (gait)
            "vertical_velocity":       0.0,
            "timestamp":               timestamp_seconds,
            "decision_confidence":     0.0,
        }

    def _update_temporal_features(self, features, previous):
        if previous is None:
            return
        dt = max(features["timestamp"] - previous["timestamp"], 1 / 30)
        dx = features["center"][0] - previous["center"][0]
        dy = features["center"][1] - previous["center"][1]
        features["center_velocity"]    = _distance(features["center"],       previous["center"])  / dt
        features["horizontal_speed"]   = abs(dx) / dt                    # pure X displacement
        features["ankle_velocity"]     = _distance(features["ankle_center"], previous["ankle_center"]) / dt
        features["ankle_spread_delta"] = abs(features["ankle_spread"] - previous["ankle_spread"])  # gait cue
        features["vertical_velocity"]  = abs(dy) / dt

    # ── Raw classification ────────────────────────────────────────────────────

    def _classify_raw(self, history, stable_state):
        current = history[-1]
        height  = max(current["full_height"], 1.0)

        # Use last 8 frames for motion averaging (more context)
        recent   = list(history)[-8:]
        recent6  = list(history)[-6:]

        # ── Normalised speed signals ──────────────────────────────────────────
        # Total speed (both axes)
        center_speed_norm   = sum(f["center_velocity"]   for f in recent6) / len(recent6) / height
        ankle_speed_norm    = sum(f["ankle_velocity"]    for f in recent6) / len(recent6) / height
        vertical_speed_norm = sum(f["vertical_velocity"] for f in recent6) / len(recent6) / height

        # Horizontal-only speed — the most reliable walking signal
        horiz_speed_norm    = sum(f["horizontal_speed"]  for f in recent)  / len(recent)  / height

        # Ankle spread variation across frames (gait oscillation)
        ankle_spread_variation = sum(f["ankle_spread_delta"] for f in recent) / len(recent) / height

        # ── Count consecutive motion frames (last 8) ─────────────────────────
        # A frame is "moving" if its horizontal speed alone exceeds a low threshold
        motion_frames = sum(
            1 for f in recent
            if f["horizontal_speed"] / height > 0.06          # lenient per-frame check
        )
        # Reliable walking: motion detected in at least 4 of 8 recent frames
        sustained_horizontal_motion = motion_frames >= 4

        # Also accept: total speed sustained, or ankle oscillation present
        moving_by_speed  = center_speed_norm > 0.12 and ankle_speed_norm > 0.08
        moving_by_ankles = ankle_speed_norm  > 0.10 and ankle_spread_variation > 0.03
        is_moving = sustained_horizontal_motion or moving_by_speed or moving_by_ankles

        # ── Posture bucket ────────────────────────────────────────────────────
        ofv = current["orientation_from_vertical"]
        if ofv < 30:
            posture = "UPRIGHT"
        elif ofv > 60:
            posture = "HORIZONTAL"
        else:
            posture = "BENDING"

        compressed         = current["vertical_span_ratio"] < 0.76 or current["box_aspect_ratio"] < 1.7
        lower_body_visible = current["hips_visible"] and current["ankles_visible"]
        upper_body_only    = current["shoulders_visible"] and current["hips_visible"] and not current["ankles_visible"]
        knee_bent          = current["hip_knee_ankle_angle"] < 120 and current["knees_visible"]

        confidence = 0.0
        state = stable_state if stable_state != "UNKNOWN" else "UNKNOWN"

        # ── FALL ─────────────────────────────────────────────────────────────
        if posture == "HORIZONTAL":
            confidence += 0.45
            if vertical_speed_norm > 0.2:
                confidence += 0.35
                if lower_body_visible:
                    confidence += 0.2
                state = "FALL"
            elif stable_state == "FALL" and lower_body_visible:
                confidence += 0.25
                state = "FALL"

        # ── SITTING — must be upright + compressed + static ──────────────────
        elif posture == "UPRIGHT" and compressed and not is_moving:
            confidence += 0.35
            if knee_bent:
                confidence += 0.25
            if upper_body_only or current["box_aspect_ratio"] < 2.2:
                confidence += 0.25
            if center_speed_norm < 0.08:
                confidence += 0.15
            # Close-up webcam override: if box suddenly grew much taller
            # (person stood up into frame), break out of sitting immediately
            if len(recent) >= 4:
                avg_prev_ar = sum(f["box_aspect_ratio"] for f in recent[:-2]) / max(len(recent)-2, 1)
                if current["box_aspect_ratio"] > avg_prev_ar * 1.35:
                    # Box got significantly taller — person is rising/standing
                    confidence = 0.3   # drop below threshold → fallback
            state = "SITTING"

        # ── WALKING — upright + any reliable motion signal ───────────────────
        elif posture == "UPRIGHT" and is_moving:
            confidence += 0.45
            if sustained_horizontal_motion:
                confidence += 0.20          # strongest signal
            if moving_by_ankles:
                confidence += 0.15          # gait oscillation bonus
            if lower_body_visible:
                confidence += 0.15
            if current["vertical_span_ratio"] > 0.75:
                confidence += 0.10
            state = "WALKING"

        # ── STANDING — upright + truly static ────────────────────────────────
        elif posture == "UPRIGHT" and not is_moving:
            confidence += 0.45
            if current["vertical_span_ratio"] > 0.8:
                confidence += 0.25
            if lower_body_visible:
                confidence += 0.10
            # Close-up mode: box taller than typical sitting aspect ratio
            if current["box_aspect_ratio"] >= 1.9:
                confidence += 0.15  # tall box → full body visible → standing
            # Extra penalty if any motion signal present — prevent mis-label
            if horiz_speed_norm > 0.04 or ankle_speed_norm > 0.05:
                confidence -= 0.20
            state = "STANDING"

        # ── BENDING ──────────────────────────────────────────────────────────
        elif posture == "BENDING":
            confidence += 0.50
            if vertical_speed_norm < 0.2:
                confidence += 0.20
            state = "BENDING"

        # ── Visibility guard ──────────────────────────────────────────────────
        if not lower_body_visible and state in {"WALKING", "FALL"}:
            state = stable_state if stable_state != "UNKNOWN" else "UNKNOWN"
            confidence = 0.45

        # ── Sitting → Walking: require sustained evidence ─────────────────────
        if stable_state == "SITTING" and state == "WALKING":
            if motion_frames < 4:
                return "SITTING", 0.7

        # ── Fall recovery guard ───────────────────────────────────────────────
        if stable_state == "FALL" and state in {"STANDING", "BENDING", "WALKING"}:
            recovery_frames = 0
            for f in reversed(list(history)[-5:]):
                ofv_now = f["orientation_from_vertical"]
                if state == "STANDING" and ofv_now < 30:
                    recovery_frames += 1
                elif state == "BENDING" and 30 <= ofv_now <= 60:
                    recovery_frames += 1
                elif state == "WALKING" and ofv_now < 30:
                    recovery_frames += 1
            if recovery_frames < 3:
                return "FALL", 0.75

        if confidence < 0.6:
            fallback = stable_state if stable_state != "UNKNOWN" else "UNKNOWN"
            return fallback, confidence

        return state, confidence

    # ── State stabilization ───────────────────────────────────────────────────

    def _stabilize_state(self, person):
        recent_states = [s for s in person["raw_states"] if s != "UNKNOWN"]
        if not recent_states:
            return person["stable_state"]

        latest = list(person["raw_states"])[-1]
        dominant_state, dominant_count = Counter(recent_states).most_common(1)[0]

        # Fast FALL promotion
        if latest == "FALL":
            fall_recent = list(person["raw_states"])[-4:]
            if fall_recent.count("FALL") >= 2:
                return "FALL"

        # FALL recovery — require sustained upright evidence
        if person["stable_state"] == "FALL" and latest in {"STANDING", "BENDING", "WALKING"}:
            recovery_recent = list(person["raw_states"])[-4:]
            if recovery_recent.count(latest) >= 3:
                return latest
            return "FALL"

        if dominant_count >= 3:
            return dominant_state

        if len(person["raw_states"]) >= 3:
            last_three = list(person["raw_states"])[-3:]
            if last_three[0] == last_three[1] == last_three[2] != "UNKNOWN":
                return last_three[0]

        return person["stable_state"]
