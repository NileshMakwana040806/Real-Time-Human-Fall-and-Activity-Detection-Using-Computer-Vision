from collections import Counter, deque
from math import atan2, degrees, sqrt


KEYPOINT_NAMES = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


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


class ActivityAnalyzer:
    def __init__(self, history_size=25, state_window=15):
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

    def _extract_features(self, points, timestamp_seconds, confidences=None, box=None):
        left_shoulder = _point(points, "left_shoulder")
        right_shoulder = _point(points, "right_shoulder")
        left_hip = _point(points, "left_hip")
        right_hip = _point(points, "right_hip")
        left_ankle = _point(points, "left_ankle")
        right_ankle = _point(points, "right_ankle")

        shoulder_center = _midpoint(left_shoulder, right_shoulder)
        hip_center = _midpoint(left_hip, right_hip)
        ankle_center = _midpoint(left_ankle, right_ankle)

        full_height = max(_distance(shoulder_center, ankle_center), 1.0)
        vertical_span = max(abs(ankle_center[1] - shoulder_center[1]), 1.0)
        shoulder_width = max(_distance(left_shoulder, right_shoulder), 1.0)

        body_axis_angle = abs(
            degrees(atan2(hip_center[1] - shoulder_center[1], hip_center[0] - shoulder_center[0]))
        )
        orientation_from_vertical = abs(90.0 - body_axis_angle)

        if box is None:
            x1, y1, x2, y2 = 0.0, 0.0, shoulder_width, full_height
        else:
            x1, y1, x2, y2 = [float(value) for value in box]
        box_width = max(x2 - x1, 1.0)
        box_height = max(y2 - y1, 1.0)

        hip_conf = min(_confidence(confidences, "left_hip"), _confidence(confidences, "right_hip"))
        ankle_conf = min(_confidence(confidences, "left_ankle"), _confidence(confidences, "right_ankle"))
        shoulder_conf = min(
            _confidence(confidences, "left_shoulder"),
            _confidence(confidences, "right_shoulder"),
        )

        return {
            "center": hip_center,
            "hip_center": hip_center,
            "shoulder_center": shoulder_center,
            "ankle_center": ankle_center,
            "full_height": full_height,
            "vertical_span_ratio": vertical_span / full_height,
            "orientation_from_vertical": orientation_from_vertical,
            "box_aspect_ratio": box_height / box_width,
            "shoulders_visible": shoulder_conf >= 0.35,
            "hips_visible": hip_conf >= 0.35,
            "ankles_visible": ankle_conf >= 0.25,
            "center_velocity": 0.0,
            "ankle_velocity": 0.0,
            "vertical_velocity": 0.0,
            "timestamp": timestamp_seconds,
            "decision_confidence": 0.0,
        }

    def _update_temporal_features(self, features, previous):
        if previous is None:
            return
        dt = max(features["timestamp"] - previous["timestamp"], 1 / 30)
        features["center_velocity"] = _distance(features["center"], previous["center"]) / dt
        features["ankle_velocity"] = _distance(features["ankle_center"], previous["ankle_center"]) / dt
        features["vertical_velocity"] = abs(features["center"][1] - previous["center"][1]) / dt

    def _classify_raw(self, history, stable_state):
        current = history[-1]
        recent = list(history)[-6:]
        height = max(current["full_height"], 1.0)

        center_speed_norm = sum(item["center_velocity"] for item in recent) / len(recent) / height
        ankle_speed_norm = sum(item["ankle_velocity"] for item in recent) / len(recent) / height
        vertical_speed_norm = sum(item["vertical_velocity"] for item in recent) / len(recent) / height

        if current["orientation_from_vertical"] < 30:
            posture = "UPRIGHT"
        elif current["orientation_from_vertical"] > 60:
            posture = "HORIZONTAL"
        else:
            posture = "BENDING"

        moving = center_speed_norm > 0.15 and ankle_speed_norm > 0.12
        motion = "MOVING" if moving else "STATIC"

        compressed = current["vertical_span_ratio"] < 0.76 or current["box_aspect_ratio"] < 1.7
        lower_body_visible = current["hips_visible"] and current["ankles_visible"]
        upper_body_only = current["shoulders_visible"] and current["hips_visible"] and not current["ankles_visible"]

        confidence = 0.0
        state = stable_state if stable_state != "UNKNOWN" else "UNKNOWN"

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

        elif posture == "UPRIGHT" and compressed and motion == "STATIC":
            confidence += 0.4
            if upper_body_only or current["box_aspect_ratio"] < 2.2:
                confidence += 0.3
            if center_speed_norm < 0.1:
                confidence += 0.2
            state = "SITTING"

        elif posture == "UPRIGHT" and motion == "MOVING":
            confidence += 0.45
            if lower_body_visible:
                confidence += 0.3
            if current["vertical_span_ratio"] > 0.8:
                confidence += 0.15
            state = "WALKING"

        elif posture == "UPRIGHT" and motion == "STATIC":
            confidence += 0.45
            if current["vertical_span_ratio"] > 0.8:
                confidence += 0.25
            if lower_body_visible:
                confidence += 0.1
            state = "STANDING"

        elif posture == "BENDING":
            confidence += 0.5
            if vertical_speed_norm < 0.2:
                confidence += 0.2
            state = "BENDING"

        if not lower_body_visible and state in {"WALKING", "FALL"}:
            state = stable_state if stable_state != "UNKNOWN" else "UNKNOWN"
            confidence = 0.45

        if stable_state == "SITTING" and state == "WALKING":
            walking_frames = sum(1 for item in recent if (item["center_velocity"] / max(item["full_height"], 1.0)) > 0.15)
            if walking_frames < 3:
                return "SITTING", 0.7

        if stable_state == "FALL" and state in {"STANDING", "BENDING", "WALKING"}:
            recovery_frames = 0
            for raw_state in reversed(list(history)[-5:]):
                posture_now = raw_state["orientation_from_vertical"]
                if state == "STANDING" and posture_now < 30:
                    recovery_frames += 1
                elif state == "BENDING" and 30 <= posture_now <= 60:
                    recovery_frames += 1
                elif state == "WALKING" and posture_now < 30:
                    recovery_frames += 1
            if recovery_frames < 3:
                return "FALL", 0.75

        if confidence < 0.6:
            fallback = stable_state if stable_state != "UNKNOWN" else "UNKNOWN"
            return fallback, confidence

        return state, confidence

    def _stabilize_state(self, person):
        recent_states = [state for state in person["raw_states"] if state != "UNKNOWN"]
        if not recent_states:
            return person["stable_state"]

        latest = list(person["raw_states"])[-1]
        dominant_state, dominant_count = Counter(recent_states).most_common(1)[0]

        if latest == "FALL":
            fall_recent = list(person["raw_states"])[-4:]
            if fall_recent.count("FALL") >= 2:
                return "FALL"

        if person["stable_state"] == "FALL" and latest in {"STANDING", "BENDING", "WALKING"}:
            recovery_recent = list(person["raw_states"])[-4:]
            if recovery_recent.count(latest) >= 3:
                return latest
            return "FALL"

        if dominant_count >= 4:
            return dominant_state

        if len(person["raw_states"]) >= 3:
            last_three = list(person["raw_states"])[-3:]
            if last_three[0] == last_three[1] == last_three[2] and last_three[0] != "UNKNOWN":
                return last_three[0]

        return person["stable_state"]
