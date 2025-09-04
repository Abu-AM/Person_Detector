import numpy as np
from typing import List, Dict
from ultralytics import YOLO
import cv2
from config import CONFIDENCE_THRESHOLD

class YOLOv11Detector:
    """
    YOLOv11Detector wraps Ultralytics YOLOv11 for object detection.
    Provides a simple interface for real-time object detection with automatic resolution optimization.
    """
    def __init__(self, device='auto', target_classes=None, max_process_size=640):
        """
        Initialize YOLOv11 detector.
        Args:
            device: 'auto', 'cpu', 'cuda', or specific device
            target_classes: List of class names to detect (e.g., ['person']). If None, detects all classes.
            max_process_size: Maximum dimension for processing (maintains aspect ratio)
        """
        # Load the YOLOv11 model file (assumed to be yolo11n.pt in the working directory)
        self.model = YOLO('yolo11n.pt')
        self.conf_threshold = CONFIDENCE_THRESHOLD
        self.target_classes = target_classes
        self.max_process_size = max_process_size
        self.frame_size = None  # Will be set dynamically
        print(f"YOLOv11 model loaded successfully")
        if target_classes:
            print(f"Filtering for classes: {target_classes}")
        print(f"Auto-scaling enabled (max process size: {max_process_size})")

    def _calculate_optimal_size(self, orig_width, orig_height):
        """
        Calculate optimal processing size while maintaining aspect ratio.
        """
        # Calculate scaling factor to fit within max_process_size
        scale = min(self.max_process_size / orig_width, self.max_process_size / orig_height)
        
        # Calculate new dimensions
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        
        # Ensure dimensions are even (some models prefer even dimensions)
        new_width = new_width if new_width % 2 == 0 else new_width + 1
        new_height = new_height if new_height % 2 == 0 else new_height + 1
        
        return new_width, new_height

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run detection on a frame with automatic resolution optimization.
        Returns: List of detections: {label, score, bbox}
        """
        # Get original frame dimensions
        orig_h, orig_w = frame.shape[:2]
        
        # Calculate optimal processing size for this frame
        proc_w, proc_h = self._calculate_optimal_size(orig_w, orig_h)
        
        # Resize frame for processing
        frame_resized = cv2.resize(frame, (proc_w, proc_h))
        
        # Run inference on resized frame
        results = self.model(frame_resized, conf=self.conf_threshold, verbose=False)
        
        detections = []
        if results and len(results) > 0:
            result = results[0]  # Get first result
            
            # Extract detections
            if result.boxes is not None:
                boxes = result.boxes
                for box in boxes:
                    # Get coordinates from resized frame
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Scale coordinates back to original frame size
                    x1_scaled = x1 * orig_w / proc_w
                    y1_scaled = y1 * orig_h / proc_h
                    x2_scaled = x2 * orig_w / proc_w
                    y2_scaled = y2 * orig_h / proc_h
                    
                    # Get confidence score
                    score = float(box.conf[0].cpu().numpy())
                    
                    # Get class label
                    class_id = int(box.cls[0].cpu().numpy())
                    label = self.model.names[class_id]
                    
                    # Filter by target classes if specified
                    if self.target_classes is None or label in self.target_classes:
                        det = {
                            'label': label,
                            'score': score,
                            'bbox': [float(x1_scaled), float(y1_scaled), float(x2_scaled), float(y2_scaled)]
                        }
                        detections.append(det)
        
        return detections

    def get_processing_info(self, frame: np.ndarray) -> Dict:
        """
        Get information about how the frame will be processed.
        """
        orig_h, orig_w = frame.shape[:2]
        proc_w, proc_h = self._calculate_optimal_size(orig_w, orig_h)
        
        return {
            'original_size': (orig_w, orig_h),
            'processing_size': (proc_w, proc_h),
            'scale_factor': (orig_w / proc_w, orig_h / proc_h),
            'memory_usage_ratio': (proc_w * proc_h) / (orig_w * orig_h)
        }

# Keep the old class name for backward compatibility
RTMDetDetector = YOLOv11Detector 