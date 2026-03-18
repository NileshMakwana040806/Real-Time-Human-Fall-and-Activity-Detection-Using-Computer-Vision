# input_handler.py

import cv2

def get_video_source():
    """
    Allows user to select input source:
    1. Webcam
    2. Video file

    Returns:
        cv2.VideoCapture object
    """

    print("Select Input Source:")
    print("1. Webcam")
    print("2. Video File")

    choice = input("Enter choice (1 or 2): ")

    # Webcam
    if choice == "1":
        print("Opening Webcam...")
        return cv2.VideoCapture(0)

    # Video File
    elif choice == "2":
        path = input("Enter video file path: ")
        return cv2.VideoCapture(path)

    else:
        print("Invalid choice. Defaulting to Webcam.")
        return cv2.VideoCapture(0)