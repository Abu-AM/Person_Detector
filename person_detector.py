#!/usr/bin/env python3
"""
Person Detector - Command Line Tool
Analyzes video files and reports timestamps when people are detected.
Enhanced for handling very long videos (days of footage).
"""

import argparse
import cv2
import json
import os
import sys
import time
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from ultralytics import YOLO
import numpy as np


def download_model(model_name: str = 'yolo11n.pt', force_download: bool = False) -> str:
    """
    Download YOLOv11 model if not present.
    
    Args:
        model_name: Name of the model to download
        force_download: Force download even if file exists
        
    Returns:
        Path to the downloaded model file
    """
    model_path = model_name
    
    # Check if model already exists
    if os.path.exists(model_path) and not force_download:
        print(f"✓ Model already exists: {model_path}")
        return model_path
    
    print(f"📥 Downloading {model_name}...")
    print("   This may take a few minutes depending on your internet connection.")
    
    try:
        # Use ultralytics to download the model
        model = YOLO(model_name)
        print(f"✓ Model downloaded successfully: {model_path}")
        return model_path
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        print("   Please check your internet connection and try again.")
        sys.exit(1)


class PersonDetector:
    """Detects people in video frames using YOLOv11."""
    
    def __init__(self, model_path='yolo11n.pt', confidence_threshold=0.3, auto_download=True):
        """
        Initialize the person detector.
        
        Args:
            model_path: Path to YOLOv11 model file
            confidence_threshold: Minimum confidence for detection
            auto_download: Automatically download model if not found
        """
        # Check if model exists, download if needed
        if not os.path.exists(model_path):
            if auto_download:
                model_path = download_model(model_path)
            else:
                print(f"✗ Model file '{model_path}' not found")
                print("   Use --auto-download to automatically download the model")
                sys.exit(1)
        
        try:
            self.model = YOLO(model_path)
            self.confidence_threshold = confidence_threshold
            print(f"✓ YOLOv11 model loaded from {model_path}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            sys.exit(1)
    
    def detect_people(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect people in a single frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of detections with bbox coordinates and confidence scores
        """
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    # Get class ID and check if it's a person (class 0 in COCO dataset)
                    class_id = int(box.cls[0].cpu().numpy())
                    if class_id == 0:  # Person class
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': confidence
                        })
        
        return detections


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    return str(timedelta(seconds=int(seconds)))


def format_duration(seconds: float) -> str:
    """Convert seconds to a more readable format for long durations."""
    if seconds < 3600:  # Less than 1 hour
        return format_timestamp(seconds)
    elif seconds < 86400:  # Less than 1 day
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
    else:  # Days or more
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        return f"{days}d {hours}h"


def estimate_processing_time(total_frames: int, fps: float, sample_time: float = 10.0) -> str:
    """Estimate total processing time based on a sample."""
    if sample_time <= 0:
        return "Unknown"
    
    frames_per_second = fps
    total_seconds = total_frames / frames_per_second
    estimated_hours = total_seconds / 3600
    
    if estimated_hours < 1:
        return f"{estimated_hours * 60:.1f} minutes"
    elif estimated_hours < 24:
        return f"{estimated_hours:.1f} hours"
    else:
        days = estimated_hours / 24
        return f"{days:.1f} days"


def analyze_video(video_path: str, detector: PersonDetector, output_json: str = None, 
                 save_interval: int = 1000, resume_from: int = None) -> List[Dict]:
    """
    Analyze video file and detect people with timestamps.
    Enhanced for very long videos with periodic saving and progress tracking.
    
    Args:
        video_path: Path to input video file
        detector: PersonDetector instance
        output_json: Optional path to save detailed results as JSON
        save_interval: How often to save progress (frames)
        resume_from: Frame number to resume from (for interrupted processing)
        
    Returns:
        List of detection events with timestamps
    """
    print(f"📹 Analyzing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ Error: Cannot open video file {video_path}")
        sys.exit(1)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"📊 Video info: {total_frames:,} frames, {fps:.2f} FPS")
    print(f"📅 Duration: {format_duration(duration)} ({duration:,.0f} seconds)")
    
    # Estimate processing time
    if duration > 3600:  # More than 1 hour
        print(f"⏱️  Estimated processing time: {estimate_processing_time(total_frames, fps)}")
    
    detections = []
    frame_count = 0
    last_detection_time = -1  # Track last detection to avoid spam
    start_time = time.time()
    last_save_time = start_time
    
    # Resume from specific frame if requested
    if resume_from and resume_from > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, resume_from)
        frame_count = resume_from
        print(f"🔄 Resuming from frame {resume_from:,}")
    
    print("🔍 Processing frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        current_time = frame_count / fps
        
        # Detect people in current frame
        people = detector.detect_people(frame)
        
        if people:
            # Only log if we haven't detected people recently (within 1 second)
            if current_time - last_detection_time > 1.0:
                timestamp = format_timestamp(current_time)
                print(f"👤 Person detected at {timestamp} ({len(people)} people)")
                
                detections.append({
                    'timestamp': timestamp,
                    'time_seconds': current_time,
                    'frame_number': frame_count,
                    'people_count': len(people),
                    'detections': people
                })
                
                last_detection_time = current_time
        
        # Progress indicator with time estimates
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            elapsed_time = time.time() - start_time
            
            if elapsed_time > 0:
                frames_per_second = frame_count / elapsed_time
                remaining_frames = total_frames - frame_count
                eta_seconds = remaining_frames / frames_per_second
                
                if eta_seconds > 3600:
                    eta_str = f"{eta_seconds/3600:.1f}h"
                elif eta_seconds > 60:
                    eta_str = f"{eta_seconds/60:.1f}m"
                else:
                    eta_str = f"{eta_seconds:.0f}s"
                
                print(f"   Progress: {progress:.1f}% ({frame_count:,}/{total_frames:,}) - ETA: {eta_str}")
        
        # Periodic saving for very long videos
        if output_json and frame_count % save_interval == 0:
            current_time_elapsed = time.time()
            if current_time_elapsed - last_save_time > 300:  # Save every 5 minutes
                save_progress(detections, video_path, output_json, frame_count, total_frames, fps, duration)
                last_save_time = current_time_elapsed
    
    cap.release()
    
    # Final save
    if output_json:
        save_progress(detections, video_path, output_json, frame_count, total_frames, fps, duration, is_final=True)
    
    return detections


def save_progress(detections: List[Dict], video_path: str, output_json: str, 
                 current_frame: int, total_frames: int, fps: float, duration: float, is_final: bool = False):
    """Save progress to JSON file."""
    progress = (current_frame / total_frames) * 100 if total_frames > 0 else 0
    
    data = {
        'video_path': video_path,
        'analysis_date': datetime.now().isoformat(),
        'progress': {
            'current_frame': current_frame,
            'total_frames': total_frames,
            'progress_percent': progress,
            'is_complete': is_final
        },
        'video_info': {
            'total_frames': total_frames,
            'fps': fps,
            'duration_seconds': duration,
            'duration_formatted': format_duration(duration)
        },
        'detections': detections,
        'summary': {
            'total_detection_events': len(detections),
            'total_people_detected': sum(d['people_count'] for d in detections) if detections else 0
        }
    }
    
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=2)
    
    if is_final:
        print(f"📄 Final results saved to: {output_json}")
    else:
        print(f"💾 Progress saved ({progress:.1f}% complete)")


def print_summary(detections: List[Dict], video_path: str):
    """Print a summary of detection results."""
    print("\n" + "="*60)
    print("📋 DETECTION SUMMARY")
    print("="*60)
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Total detection events: {len(detections)}")
    
    if detections:
        total_people = sum(d['people_count'] for d in detections)
        print(f"Total people detected: {total_people}")
        print(f"Average people per detection: {total_people/len(detections):.1f}")
        
        # Group detections by hour for very long videos
        if len(detections) > 50:  # If many detections, show hourly summary
            print("\n📅 Hourly Summary:")
            hourly_counts = {}
            for det in detections:
                hour = int(det['time_seconds'] // 3600)
                hourly_counts[hour] = hourly_counts.get(hour, 0) + det['people_count']
            
            for hour in sorted(hourly_counts.keys()):
                print(f"  Hour {hour}: {hourly_counts[hour]} people detected")
        
        print("\n📅 Detection Timeline:")
        # Show first 10 and last 10 detections for very long videos
        if len(detections) > 20:
            for i, det in enumerate(detections[:10], 1):
                print(f"  {i:2d}. {det['timestamp']} - {det['people_count']} person(s)")
            print(f"  ... ({len(detections) - 20} more detections) ...")
            for i, det in enumerate(detections[-10:], len(detections) - 9):
                print(f"  {i:2d}. {det['timestamp']} - {det['people_count']} person(s)")
        else:
            for i, det in enumerate(detections, 1):
                print(f"  {i:2d}. {det['timestamp']} - {det['people_count']} person(s)")
    else:
        print("❌ No people detected in this video.")
    
    print("="*60)


def main():
    """Main function to handle command line interface."""
    parser = argparse.ArgumentParser(
        description="Person Detector - Analyze video files for person detection (Enhanced for long videos)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  Basic usage:
    python person_detector.py -i video.mp4
  
  Save detailed results:
    python person_detector.py -i video.mp4 -o results.json
  
  Adjust detection sensitivity:
    python person_detector.py -i video.mp4 --confidence 0.5
  
  Resume interrupted processing:
    python person_detector.py -i long_video.mp4 --resume 50000
  
  Force re-download model:
    python person_detector.py -i video.mp4 --force-download
  
  Note: Models are automatically downloaded if not found (no flag needed)
  
  Use different model:
    python person_detector.py -i video.mp4 --model yolo11s.pt
  
  Adjust save interval for long videos:
    python person_detector.py -i long_video.mp4 --save-interval 500

SUPPORTED VIDEO FORMATS:
  MP4, AVI, MOV, MKV, WebM, FLV, WMV, and other OpenCV-compatible formats

MODEL OPTIONS:
  yolo11n.pt (default) - Nano model, fastest processing
  yolo11s.pt          - Small model, good balance
  yolo11m.pt          - Medium model, higher accuracy
  yolo11l.pt          - Large model, best accuracy
  yolo11x.pt          - Extra large model, highest accuracy

For more information, visit: https://github.com/yourusername/person-detector
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Path to input video file (MP4, AVI, MOV, WebM, etc.)'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Save detailed results to JSON file (includes timestamps, coordinates, confidence scores)'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.3,
        help='Detection confidence threshold (0.0-1.0). Lower = more detections but more false positives (default: 0.3)'
    )
    
    parser.add_argument(
        '--model',
        default='yolo11n.pt',
        help='YOLOv11 model to use: yolo11n.pt (fastest), yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt (most accurate)'
    )
    
    parser.add_argument(
        '--resume',
        type=int,
        help='Resume processing from specific frame number (useful for interrupted long videos)'
    )
    
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1000,
        help='Save progress every N frames for long videos (default: 1000, lower = more frequent saves)'
    )
    
    parser.add_argument(
        '--auto-download',
        action='store_true',
        help='Automatically download model if not found locally (DEFAULT BEHAVIOR - this flag is optional)'
    )
    
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Force re-download model even if it already exists (useful for updating models)'
    )
    
    # Check if no arguments provided and show help
    if len(sys.argv) == 1:
        parser.print_help()
        print("\n" + "="*60)
        print("💡 QUICK START")
        print("="*60)
        print("To analyze a video file, use:")
        print("  python person_detector.py -i your_video.mp4")
        print("\nFor more examples, see the EXAMPLES section above.")
        print("="*60)
        sys.exit(0)
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"✗ Error: Input file '{args.input}' does not exist")
        sys.exit(1)
    
    # Initialize detector with auto-download option
    print("🚀 Initializing Person Detector...")
    detector = PersonDetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
        auto_download=True  # Always auto-download by default
    )
    
    # Analyze video
    detections = analyze_video(
        args.input, 
        detector, 
        args.output, 
        args.save_interval,
        args.resume
    )
    
    # Print summary
    print_summary(detections, args.input)


if __name__ == "__main__":
    main()
