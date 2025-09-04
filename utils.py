import cv2
import os
from datetime import datetime

def draw_detections(frame, detections):
    """Draw bounding boxes and labels on the frame."""
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        label = f"{det['label']} {det['score']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def timestamp_filename(prefix='det', ext='jpg'):
    """Generate a timestamped filename."""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    return f"{prefix}_{ts}.{ext}"

def safe_save(path, data):
    """Safely save data to a file, creating directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(data) 