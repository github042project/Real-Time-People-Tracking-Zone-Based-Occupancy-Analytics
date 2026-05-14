from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import pandas as pd

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Video path
video_path = "../input/video.mp4"

# Open video
cap = cv2.VideoCapture(video_path)

# Get original FPS
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 30

# Frame size
frame_width = 640
frame_height = 480

# Video codec
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Output video writer
out = cv2.VideoWriter(
    "../output/output_video.avi",
    fourcc,
    fps,
    (frame_width, frame_height)
)

# Define zones
zones = {
    "Entrance": [(0, 0), (320, 240)],
    "Checkout": [(320, 0), (640, 240)],
    "Aisle_Left": [(0, 240), (320, 480)],
    "Aisle_Right": [(320, 240), (640, 480)]
}

# Unique visitors
unique_visitors = {
    zone: set() for zone in zones
}

# Peak occupancy
peak_occupancy = {
    zone: 0 for zone in zones
}

peak_time = {
    zone: 0 for zone in zones
}

# Occupancy log
occupancy_log = []

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (640, 480))

    frame_number += 1

    timestamp = round(frame_number / fps, 2)

    # Zone counts
    zone_counts = {
        zone: 0 for zone in zones
    }

    # Draw zones
    for zone_name, points in zones.items():

        (zx1, zy1), (zx2, zy2) = points

        cv2.rectangle(
            frame,
            (zx1, zy1),
            (zx2, zy2),
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            zone_name,
            (zx1 + 10, zy1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    # YOLO detection
    results = model(frame, classes=[0], verbose=False)

    detections = []

    for result in results:

        for box in result.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            conf = float(box.conf[0])

            detections.append(
                ([x1, y1, x2 - x1, y2 - y1], conf, "person")
            )

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Process tracks
    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id

        ltrb = track.to_ltrb()

        x1, y1, x2, y2 = map(int, ltrb)

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Draw tracking ID
        cv2.putText(
            frame,
            f"ID: {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Foot position
        center_x = int((x1 + x2) / 2)
        center_y = int(y2)

        # Draw foot point
        cv2.circle(
            frame,
            (center_x, center_y),
            4,
            (0, 0, 255),
            -1
        )

        # Check zones
        for zone_name, points in zones.items():

            (zx1, zy1), (zx2, zy2) = points

            if zx1 <= center_x <= zx2 and zy1 <= center_y <= zy2:

                zone_counts[zone_name] += 1

                unique_visitors[zone_name].add(track_id)

    # Update peak occupancy
    for zone in zones:

        if zone_counts[zone] > peak_occupancy[zone]:

            peak_occupancy[zone] = zone_counts[zone]

            peak_time[zone] = timestamp

    # Display occupancy counts
    y_position = 30

    for zone, count in zone_counts.items():

        cv2.putText(
            frame,
            f"{zone}: {count}",
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        y_position += 35

        # Save occupancy log
        occupancy_log.append({
            "frame": frame_number,
            "time_sec": timestamp,
            "zone": zone,
            "occupancy": count
        })

    # Save video frame
    out.write(frame)

    # Bigger display window
    display_frame = cv2.resize(frame, (1000, 700))

    # Show frame
    cv2.imshow(
        "People Tracking and Zone Analytics",
        display_frame
    )

    # Press q to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save occupancy CSV
df = pd.DataFrame(occupancy_log)

df.to_csv(
    "../output/occupancy_log.csv",
    index=False
)

# Print final analytics
print("\n========== FINAL ANALYTICS ==========\n")

for zone in zones:

    print(f"Zone: {zone}")

    print(f"Unique Visitors: {len(unique_visitors[zone])}")

    print(f"Peak Occupancy: {peak_occupancy[zone]}")

    print(f"Peak Time: {peak_time[zone]} sec")

    print("-----------------------------------")

# Release everything properly
cap.release()
out.release()
cv2.destroyAllWindows()