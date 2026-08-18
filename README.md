# Adaptive Object Localization and Tracking Across Scales and Occlusions

<div align="center">
  <img src="Report/Working Detection.png" alt="Working Detection" width="800"/>
</div>

## Overview
This project provides robust computer vision scripts for detecting and tracking sports balls in video feeds. It is highly optimized for **Basketballs** and **Footballs**, utilizing a hybrid approach of color-based segmentation, motion detection, shape analysis, and Kalman filtering to maintain accurate tracking even during occlusions.

---

## Features

- **Multi-Sport Support**: Specialized profiles and color ranges for basketballs and footballs.
- **Robust Tracking**: Kalman filtering combined with appearance modeling for smooth trajectory estimation.
- **Occlusion Handling**: The tracker is designed to maintain track of the ball even when it is temporarily blocked by players or objects.
- **Real-Time Visualization**: Watch the tracking happen live with bounding boxes, trajectory lines, and ID labels.

### Basketball Tracking
<div align="center">
  <img src="Report/Basketball Tracking.png" alt="Basketball Tracking" width="800"/>
</div>

### Football Tracking
<div align="center">
  <img src="Report/Football Tracking.png" alt="Football Tracking" width="800"/>
</div>

### Occlusion Handling
The system elegantly handles situations where the ball is occluded.
<div align="center">
  <img src="Report/Occlusion (1).png" alt="Occlusion 1" width="400"/>
  <img src="Report/Occlusion (2).png" alt="Occlusion 2" width="400"/>
</div>

---

## Requirements & Installation

The project relies on standard computer vision libraries in Python.

1. Install the required dependencies:
```bash
pip install opencv-python numpy scipy
```

2. Clone this repository or download the scripts to your local machine.

---

## Usage Guide

The project includes two main scripts: `detection_tracker.py` for general detection and `Ball_Tracker.py` which is highly specialized for specific sports.

### 1. Using `Ball_Tracker.py` (Recommended for Sports)

This script provides the most optimized experience for tracking basketballs and footballs.

**Command Syntax:**
```bash
python Ball_Tracker.py [VIDEO_PATH] [VIDEO_TYPE] [options]
```
- `VIDEO_PATH`: Path to the input video.
- `VIDEO_TYPE`: The sport type (`basketball` or `football`).

**Options:**
- `-o`, `--output_path`: Save the result to a specified path.
- `-d`, `--display`: Enable the visualization window while processing.
- `-n`, `--top_n_balls`: Maximum number of balls to track simultaneously (default: 1).

**Examples:**
```bash
# Track a basketball with visualization
python Ball_Tracker.py basketball_game.mp4 basketball --display

# Track up to 2 footballs and save the result
python Ball_Tracker.py football_match.mp4 football -n 2 -o tracked_football.mp4
```

### 2. Using `detection_tracker.py`

A more generalized tracking script for various objects or general ball detection.

**Command Syntax:**
```bash
python detection_tracker.py --video [VIDEO_PATH] --mode [MODE] --output [OUTPUT_PATH]
```
- `MODE`: `all` (general objects) or `balls` (general ball detection).
- Use `--show_mask` to view the detection mask for debugging.

---

## How It Works

Our tracking system uses a multi-stage pipeline:

1. **Color-Based Detection**: Uses carefully tuned HSV color ranges to segment out potential ball regions, with sport-specific profiles.
2. **Motion Detection**: Implements background subtraction and frame differencing to capture rapid movement while suppressing court reflections.
3. **Shape Analysis**: Filters candidates using circularity, aspect ratio, and solidity (area vs. convex hull area) metrics.
4. **Kalman Filtering Tracking**: Predicts the ball's next position based on its velocity and trajectory, allowing the system to handle missed detections and occlusions effectively.

---

## Customization & Troubleshooting

If you are testing on your own footage and facing issues, consider adjusting these parameters in `Ball_Tracker.py`:

- **HSV Color Ranges (`COLOR_RANGES`)**: Tweak these if your ball is a slightly different shade or if lighting conditions are unusual.
- **Size Constraints (`BALL_FILTER_PARAMS`)**: Modify `min_radius` and `max_radius` depending on the resolution of your video and how far away the camera is.
- **Shape Constraints**: Adjust `min_circularity` if motion blur is heavily distorting the ball's shape.