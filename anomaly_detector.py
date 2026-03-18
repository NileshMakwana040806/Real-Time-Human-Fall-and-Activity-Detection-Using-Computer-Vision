# anomaly_detector.py

# Store previous hip positions for each person
previous_positions = {}

def detect_activity(points, track_id):
    """
    points: keypoints of a single person (17 body points)
    track_id: ID of the person (for tracking)

    Returns:
        activity label (STANDING, SITTING, WALKING, FALL)
    """

    try:
        # Keypoints index reference (YOLO Pose)
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
        hip_x = (left_hip[0] + right_hip[0]) / 2
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

        # 3. WALKING detection (movement in X direction)
        if track_id in previous_positions:
            prev_x = previous_positions[track_id]

            # If hip moves significantly → walking
            if abs(hip_x - prev_x) > 10:
                previous_positions[track_id] = hip_x
                return "WALKING"

        # Update position
        previous_positions[track_id] = hip_x

        # 4. STANDING detection
        if hip_y < knee_y < ankle_y:
            return "STANDING"

        # 5. Default
        return "UNKNOWN"

    except:
        return "UNKNOWN"