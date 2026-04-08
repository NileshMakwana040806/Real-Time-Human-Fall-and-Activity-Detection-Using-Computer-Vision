# 🛡️ Real-Time Human Fall and Activity Detection

> **AI-powered, camera-only fall detection and activity monitoring — no wearable devices required.**

Built with **YOLOv8 Pose Estimation** + **ByteTrack** + **Streamlit** | Runs in real-time on standard CPU

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Features](#-features)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Activity Detection Logic](#-activity-detection-logic)
- [Configuration](#-configuration)
- [Log File System](#-log-file-system)
- [Performance](#-performance)
- [Known Limitations](#-known-limitations)
- [Future Work](#-future-work)
- [References](#-references)

---

## 🔍 Overview

Falls are the **second leading cause of accidental deaths worldwide** (WHO, 2021 — 684,000 deaths per year). Most existing solutions either require the person to wear a device, manually press a button, or rely on a human operator watching CCTV footage 24/7. None of these are truly automatic or non-intrusive.

This project solves that problem with a **fully automated, camera-based system** that:

- Watches people through a standard webcam or video file
- Understands their posture and movement in real time using AI
- Detects falls **within 1–2 seconds** and triggers an immediate alert
- Requires **no wearable device**, **no internet connection**, and **no GPU**

### Detected Activity States

| State | Description | Alert |
|---|---|---|
| 🟢 **STANDING** | Person upright and still | — |
| 🟢 **WALKING** | Person upright and moving horizontally | — |
| 🔵 **SITTING** | Person in a seated posture | — |
| 🟡 **BENDING** | Body tilted 30–60° — pre-fall risk signal | ⚠️ Visual warning |
| 🔴 **FALL** | Body suddenly becomes horizontal | 🚨 Alert + Beep + Snapshot |

---

## 🎬 Demo

### Interface Layout

```
┌──────────────────────────────────────────────────────────────────┐
│  ⏹ Stop   ⏸ Pause   ▶ Resume               📁 VIDEO FILE        │
├──────────────────────────────────────────────────────────────────┤
│                        │                     │                   │
│   LIVE VIDEO FEED      │    DASHBOARD        │   ACTIVITY LOG    │
│                        │                     │                   │
│  [Skeleton overlay]    │  ✅ ALL NORMAL      │  [19:30:01] ID 1  │
│  [Bounding boxes]      │  ─────────────      │   -> WALKING(81%) │
│  [ID 1: WALKING 81%]   │  Total People:  2   │                   │
│  [ID 2: SITTING 88%]   │  🟢 Standing:   1   │  [19:30:05] ID 2  │
│  [FPS: 9.2  Lat:108ms] │  🟢 Walking:    1   │   -> SITTING(88%) │
│                        │  🔵 Sitting:    0   │                   │
│                        │  🟡 Bending:    0   │  [19:30:12] ID 1  │
│                        │  🔴 Falls:      0   │   -> STANDING(74%)│
│                        │  ⚡ FPS:     9.21   │                   │
│                        │  🕐 Latency: 108ms  │                   │
└──────────────────────────────────────────────────────────────────┘
```

### When a Fall is Detected

```
┌──────────────────────────────────────────────────────────────────┐
│                        │                     │                   │
│  [Red overlay on box]  │  🚨 DANGER —        │  [19:30:22] ID 1  │
│  [! FALL ! text]       │     FALL DETECTED   │   -> FALL (100%)  │
│                        │  ─────────────      │                   │
│                        │  🔴 Falls:  1       │  [19:30:22]       │
│  🔊 Beep alert plays   │  ⚠ Fall at 19:30:22 │  FALL DETECTED    │
│  📸 Snapshot saved     │                     │  Status: DANGER   │
└──────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### Core Detection
- 🎯 **YOLOv8n-pose** detects 17 COCO body keypoints per person per frame
- 👥 **Multi-person support** — all people in frame tracked simultaneously with stable IDs
- 🔢 **Confidence % on every label** — e.g. `ID 1: WALKING 81%`
- ⚡ **Fall detected in 1–2 seconds** via fast 2-of-4-frame promotion rule
- 🟡 **BENDING as pre-fall warning** — body tilt of 30–60° flagged before fall occurs

### 3-Layer Stable ID Tracking
No more IDs changing every few frames:
- **Layer 1 — Direct match:** ByteTrack ID already known → reuse instantly
- **Layer 2 — IoU recovery:** Velocity-predicted position matching for lost tracks
- **Layer 3 — Distance fallback:** 120px Euclidean threshold for fast movers
- **45-frame grace period** before expiring a lost track (~1.5 sec at 30 FPS)

### Alerts on Fall Detection
- 🚨 Red pulsing danger banner in dashboard
- 🔊 Browser beep (Web Audio API) + Python system bell fallback
- 📸 Fall frame snapshot auto-saved to `saved_frames/`
- 📝 Event recorded in session log with precise timestamp

### Interface
- Three-panel layout: **Video | Dashboard | Activity Log**
- Color-coded log: 🟢 movement / 🔵 sitting / 🟡 bending / 🔴 fall
- Real-time FPS + latency HUD on video feed
- Pause / Resume for video analysis
- Adjustable settings in sidebar (no restart needed)

### Session Management
- 📊 **Automatic session summary** — duration, people, falls, activity % per person
- 📥 **One-click log download** from browser after session ends
- 🗂️ **Organized log files** — one unique timestamped file per session

---

## ⚙️ How It Works

### Processing Pipeline (per frame)

```
Camera / Video File
       │
       ▼  cv2.VideoCapture
OpenCV Frame Read
       │
       ▼  model.track(frame, persist=True)
YOLOv8n-pose Inference
  → 17 keypoints (x, y, confidence) per person
  → Bounding boxes
  → ByteTrack IDs
       │
       ▼  confidence ≥ 45%  AND  box area ≥ 3000 px²
Quality Filters
       │
       ▼  assign_stable_ids()
3-Layer Stable ID Assignment
       │
       ▼  _extract_features()
Feature Extraction per person:
  ┌─────────────────────────────────────┐
  │ orientation_from_vertical (angle °) │
  │ box_aspect_ratio          (h / w)   │
  │ horizontal_speed          (px/s)    │
  │ ankle_spread_delta        (gait)    │
  │ vertical_velocity         (px/s)    │
  │ hip_knee_ankle_angle      (°)       │
  └─────────────────────────────────────┘
       │
       ▼  _classify_raw() + _stabilize_state()
12-Frame Temporal Classification
       │
       ├── Normal → update dashboard + log
       └── FALL   → red banner + beep + snapshot + log
```

### Classification Decision Tree

```
Body angle from vertical:
  < 30°  →  UPRIGHT    bucket
  30-60° →  BENDING    bucket  ← pre-fall warning
  > 60°  →  HORIZONTAL bucket

UPRIGHT  + no motion                   →  STANDING
UPRIGHT  + motion in ≥ 4 of 8 frames  →  WALKING
UPRIGHT  + compressed box + static    →  SITTING
BENDING                                →  BENDING
HORIZONTAL + high vertical velocity   →  FALL
```

---

## 📁 Project Structure

```
project/
│
├── app.py                ← Streamlit UI, session control, main loop
├── main_backend.py       ← YOLO inference, ID tracking, drawing, alerts
├── activity_detector.py  ← Feature extraction, classification, smoother
├── event_logger.py       ← Session log creation and writing
│
├── yolov8n-pose.pt       ← Pre-trained YOLOv8 nano pose model (6.6 MB)
├── requirements.txt      ← pip dependencies
│
├── event_log/            ← AUTO-CREATED on first run
│   ├── webcam/
│   │   └── webcam_2026-03-25_19-30-00.txt
│   └── video/
│       └── myvideo_2026-03-25_20-00-00.txt
│
└── saved_frames/         ← AUTO-CREATED on first fall detection
    └── fall_19-30-22.jpg
```

### File Responsibilities

| File | Role | Key Functions |
|---|---|---|
| `app.py` | UI, session state, main loop | `render_dashboard()`, `render_logs()`, `render_session_summary()` |
| `main_backend.py` | YOLO + tracking + drawing | `process_frame()`, `assign_stable_ids()` |
| `activity_detector.py` | Classification engine | `analyze()`, `_classify_raw()`, `_stabilize_state()` |
| `event_logger.py` | Log file management | `create_session_log()`, `append_log()`, `write_session_summary()` |

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- A webcam **or** a video file to analyze
- No GPU needed — runs on CPU

### Step 1 — Clone

```bash
git clone https://github.com/yourusername/fall-detection.git
cd fall-detection
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

**What gets installed:**

| Package | Version | Purpose |
|---|---|---|
| `ultralytics` | ≥ 8.0.0 | YOLOv8 model + ByteTrack |
| `streamlit` | ≥ 1.32.0 | Web interface |
| `opencv-python` | ≥ 4.9.0 | Video capture + drawing |
| `torch` | ≥ 2.0.0 | Neural network inference |

> **First run:** `yolov8n-pose.pt` (~6.6 MB) is downloaded automatically if missing.

### Step 3 — Run

```bash
streamlit run app.py
```

Open → `http://localhost:8501`

---

## 🖥️ Usage

### Live Webcam Monitoring

1. Click **▶ Start Webcam** on the landing page
2. Grant camera access if prompted
3. System begins monitoring immediately
4. Click **⏹ Stop** → session summary appears

### Video Analysis

1. Click **📁 Choose a video** — supports MP4, AVI, MOV, MKV
2. Click **▶ Analyze Video**
3. Use **⏸ Pause** / **▶ Resume** at any time
4. Video ends → session summary appears automatically

### Session Summary

```
📊 Session Summary
─────────────────────────────
Duration      4m 32s
People Seen   2
Falls         1  ← shown in red if > 0

Fall timestamps:
  ⚠️  19:30:22

Activity breakdown per person:
  Person 1:
    WALKING    ████████████░░░░░░░░  60.2%
    STANDING   ████████░░░░░░░░░░░░  38.1%
    FALL       █░░░░░░░░░░░░░░░░░░░   1.7%

  📥  Download Session Log (.txt)
```

---

## 🧠 Activity Detection Logic

### 12-Frame State Stabilizer Rules

| Priority | Rule | Condition | Purpose |
|---|---|---|---|
| 1 | **FALL fast-promote** | ≥ 2 FALL votes in last 4 frames | Detect falls quickly (safety-critical) |
| 2 | **FALL recovery guard** | ≥ 3 non-FALL in 4 frames to exit FALL | Prevent false recovery from rolling |
| 3 | **Majority voting** | ≥ 3 same state in 12-frame window | Normal activity stabilization |
| 4 | **Consecutive triple** | Last 3 frames identical | Quick confirmation of clear transitions |
| 5 | **Fallback** | None of above triggered | Hold previous state — ignore single-frame noise |

### Features Explained

| Feature | How Computed | What It Detects |
|---|---|---|
| `orientation_from_vertical` | `|90° - atan2(hip_y−shoulder_y, hip_x−shoulder_x)|` | Upright / Bending / Fallen posture |
| `horizontal_speed` | `|hip_x_now − hip_x_prev| / dt` | Walking (horizontal movement only) |
| `ankle_spread_delta` | `|spread_now − spread_prev|` | Gait oscillation — walking signature |
| `vertical_velocity` | `|hip_y_now − hip_y_prev| / dt` | Sudden downward drop (fall) |
| `box_aspect_ratio` | `box_height / box_width` | Compressed box → sitting posture |
| `hip_knee_ankle_angle` | Dot product at knee vertex | Knee bend → sitting confirmation |

---

## 🔧 Configuration

Settings are live in the **sidebar** — no restart needed.

### Min Pose Confidence *(default: 0.45)*

Controls which detections are processed. Average keypoint confidence below this threshold is discarded.

```
Lower value (0.20)  →  More detections, including background noise
Default (0.45)      →  Good balance for most environments
Higher value (0.80) →  Only very clear, close detections
```

### Video Frame Skip *(default: 2, video only)*

Every Nth frame is sent to YOLO. Skipped frames reuse the last real dashboard (no flickering).

```
Skip = 1  →  Every frame processed  (~7 FPS,  most accurate)
Skip = 2  →  Every other frame     (~14 FPS, recommended)
Skip = 3  →  Every 3rd frame       (~20 FPS, slight accuracy drop)
Skip = 4  →  Every 4th frame       (~25 FPS, for slow machines)
```

> Webcam always processes every frame regardless of this setting.

### Recommended Settings by Environment

| Environment | Confidence | Skip |
|---|---|---|
| Good lighting, close camera, 1 person | 0.50 | 1 |
| Normal indoor, 1–2 people | 0.45 | 2 |
| Large room, 3+ people | 0.40 | 2 |
| Busy background, clutter | 0.55 | 1 |
| Older/slower machine | 0.45 | 3 |

---

## 🗂️ Log File System

### Naming Convention

```
event_log/
├── webcam/
│   └── webcam_YYYY-MM-DD_HH-MM-SS.txt
└── video/
    └── <videoname>_YYYY-MM-DD_HH-MM-SS.txt
```

One file per session. Never mixed. Immediately tells you: *what type of session, which video, exact date and time*.

### Log File Example

```
============================================================
  SESSION TYPE  : VIDEO
  VIDEO FILE    : hospital_corridor.mp4
  DATE & TIME   : 2026-03-25  19:30:00
============================================================

[00:00:02.035] ID 1 -> WALKING (81%)
[00:00:08.266] ID 1 -> STANDING (74%)
[00:00:15.301] ID 2 -> SITTING (88%)
[00:00:22.603] ID 1 -> FALL (100%)
[00:00:22.603] FALL DETECTED | ID 1 | Status: DANGER
[00:00:23.604] FALL DETECTED | ID 1 | Status: DANGER
[00:00:31.112] ID 1 -> STANDING (69%)

============================================================
  SESSION SUMMARY
============================================================
  Duration          : 4m 32s
  Total people seen : 2
  Falls detected    : 1
    -> Fall at 00:00:22.603
  Activity breakdown per person:
    Person 1:
      STANDING    :  48.2%
      WALKING     :  32.1%
      FALL        :   4.3%
      SITTING     :  15.4%
    Person 2:
      SITTING     :  91.7%
      STANDING    :   8.3%
============================================================
```

---

## 📊 Performance

Tested on **Intel Core i5 8th Gen · 8 GB RAM · CPU only (no GPU)**:

| Metric | Result |
|---|---|
| FPS — Webcam (every frame) | **7–10 FPS** |
| FPS — Video (skip = 2) | **14–18 FPS** |
| Fall detection latency | **1.0–1.8 seconds** |
| Per-frame latency | **100–130 ms** |
| ID stability through occlusion | **up to 1.5 seconds** |
| Max people tracked simultaneously | **tested up to 4** |
| Model file size | **6.6 MB** |

---

## ⚠️ Known Limitations

| Issue | When It Happens | Workaround |
|---|---|---|
| Walking-in-place not detected | No horizontal movement (treadmill) | Person must move across the frame |
| Reduced accuracy near camera | Only upper body visible | Position camera to capture waist-up |
| Low FPS on weak hardware | CPU-only, older i3 / Atom | Increase video frame-skip to 3–4 |
| Very slow falls may be missed | 10+ second descent to floor | BENDING state shown as visual warning |
| Darkness reduces keypoint confidence | Poor lighting | Ensure adequate room illumination |

---

## 🔮 Future Work

- [ ] **SMS / WhatsApp alert** — Twilio API integration for instant caregiver notification
- [ ] **Multi-camera dashboard** — monitor multiple rooms simultaneously
- [ ] **Cloud deployment** — remote monitoring from any browser via AWS / GCP
- [ ] **Night vision support** — IR camera integration for 24/7 low-light operation
- [ ] **Custom model fine-tuning** — improve edge-case accuracy on fall-specific datasets
- [ ] **Long-term analytics** — weekly / monthly activity trend reports
- [ ] **Mobile app** — companion app for real-time caregiver alerts

---

## 📚 References

1. Jocher, G., Chaurasia, A., Qiu, J. (2023). *Ultralytics YOLOv8*. https://github.com/ultralytics/ultralytics
2. Zhang, Y. et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*. ECCV 2022.
3. Bradski, G. (2000). *The OpenCV Library*. Dr. Dobb's Journal. https://opencv.org
4. Lin, T.Y. et al. (2014). *Microsoft COCO: Common Objects in Context*. ECCV 2014.
5. World Health Organization (2021). *Falls — Key Facts*. https://www.who.int/news-room/fact-sheets/detail/falls
6. Streamlit Inc. (2024). *Streamlit Documentation*. https://docs.streamlit.io

---

## 📄 License

Developed for academic submission — Mini Project, AI & Data Science, ADIT CVM University (A.Y. 2025-26).

---

<div align="center">

**Made for safer living 🛡️**

`YOLOv8` · `ByteTrack` · `OpenCV` · `Streamlit` · `PyTorch`

</div>
