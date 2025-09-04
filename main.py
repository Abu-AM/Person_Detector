import cv2
import os
import json
from detector import YOLOv11Detector
from config import VIDEO_SOURCE, OUTPUT_DIR, DEVICE, TARGET_CLASSES
from utils import draw_detections, timestamp_filename


OUTSIDE_DOCKER = True

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize detector
print(f"Loading YOLOv11 detector...")
detector = YOLOv11Detector(device=DEVICE, target_classes=TARGET_CLASSES)

# Open video source
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Error: Cannot open video source {VIDEO_SOURCE}")
    exit(1)

# Get video properties for output video
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None:
    fps = 25  # fallback default
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_video_path = os.path.join(OUTPUT_DIR, 'detections_with_boxes.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

log_path = os.path.join(OUTPUT_DIR, 'detections.json')
log_file = open(log_path, 'a')

print("Starting video stream. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    detections = detector.detect(frame)
    frame_disp = draw_detections(frame.copy(), detections)
    out.write(frame_disp)  # Save frame with boxes to video

    if detections:
        # Save frame as JPEG
        filename = timestamp_filename(prefix='det', ext='jpg')
        filepath = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(filepath, frame)
        # Log detections
        log_entry = {
            'file': filename,
            'detections': detections
        }
        log_file.write(json.dumps(log_entry) + '\n')
        log_file.flush()

    if OUTSIDE_DOCKER:
        cv2.imshow('Drone Security - YOLOv11', frame_disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
log_file.close()

if OUTSIDE_DOCKER:
    cv2.destroyAllWindows() 