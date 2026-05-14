# Real-Time People Tracking & Zone-Based Occupancy Analytics

## Overview
This project uses YOLOv8 and DeepSORT to perform real-time people detection, tracking, and zone-based occupancy analytics on surveillance footage.

## Features
- Real-time person detection using YOLOv8
- Multi-object tracking using DeepSORT
- Persistent tracking IDs
- Zone-based occupancy counting
- Unique visitor counting
- Peak occupancy analytics
- Occupancy log CSV generation
- Annotated output video generation

## Technologies Used
- Python
- OpenCV
- YOLOv8
- DeepSORT
- Pandas

## Project Structure

people_tracking_project/
│
├── input/
├── output/
├── src/
├── README.md

## Installation

```bash
pip install ultralytics
pip install deep-sort-realtime
pip install opencv-python
pip install pandas
```

## Run

```bash
python main.py
```

## Zone Definitions
The video frame was divided into 4 logical zones based on observed store layout and movement patterns.

## Analytics Generated
- Live occupancy count
- Unique visitors per zone
- Peak occupancy
- Occupancy timestamp
- Frame-level occupancy logs

## Challenges Faced
- ID switches during occlusion
- Zone boundary overlap
- Maintaining stable tracking

## Solutions
- Increased DeepSORT max_age
- Used bottom-center foot position for zone assignment
- Used confirmed tracks only

## Output Files
- output_video.mp4
- occupancy_log.csv
