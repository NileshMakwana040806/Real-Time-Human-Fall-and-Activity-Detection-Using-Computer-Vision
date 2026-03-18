# anomaly_detector.py

# This function takes body keypoints and predicts activity
def detect_activity(points, track_id):
    """
    points: keypoints of a single person (17 body points)
    track_id: ID of the person (for tracking)

    Returns:
        activity label (STANDING, SITTING, WALKING, FALL)
    """

    try:
        # Keypoints index reference (YOLO Pose)
        # 0 = nose
        # 11 = left hip
        # 12 = right hip
        # 13 = left knee
        # 14 = right knee
        # 15 = left ankle
        # 16 = right ankle

        # Extract required points
        left_hip = points[11]
        right_hip = points[12]
        left_knee = points[13]
        right_knee = points[14]
        left_ankle = points[15]
        right_ankle = points[16]

        # Average positions (more stable)
        hip_y = (left_hip[1] + right_hip[1]) / 2
        knee_y = (left_knee[1] + right_knee[1]) / 2
        ankle_y = (left_ankle[1] + right_ankle[1]) / 2

        # -------- Activity Logic --------

        # 1. FALL detection (body horizontal)
        if abs(hip_y - ankle_y) < 40:
            return "FALL"

        # 2. SITTING detection
        if abs(hip_y - knee_y) < 40:
            return "SITTING"

        # 3. STANDING detection
        if hip_y < knee_y < ankle_y:
            return "STANDING"

        # 4. Default
        return "UNKNOWN"

    except:
        # If keypoints missing or error occurs
        return "UNKNOWN"