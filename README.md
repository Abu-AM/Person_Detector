# Person Detector - Command Line Tool

A command-line tool that analyzes video files and reports when people are detected with timestamps. **Enhanced for handling very long videos spanning several days of footage.**

## Features

- 🎯 **Person Detection**: Uses YOLOv11 to detect people in video files
- ⏰ **Timestamp Reporting**: Shows exact timestamps when people are detected
- 📊 **Progress Tracking**: Real-time progress updates during analysis
- 📄 **JSON Export**: Optional detailed results export
- 🎛️ **Configurable**: Adjustable confidence threshold and model path
- ⏱️ **Time Estimates**: ETA calculations for long video processing
- 💾 **Periodic Saving**: Automatic progress saving for very long videos
- 🔄 **Resume Capability**: Resume interrupted processing from any frame
- 📅 **Smart Summaries**: Hourly summaries for multi-day videos
- 📥 **Auto-Download**: Automatically download model if not found

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. **Optional**: The tool can automatically download the YOLOv11 model on first use.

## Usage

### Basic Usage
```bash
python person_detector.py -i input.mp4
```

### Save Detailed Results
```bash
python person_detector.py -i input.mp4 -o results.json
```

### Adjust Detection Sensitivity
```bash
python person_detector.py -i input.mp4 --confidence 0.5
```

### Resume Interrupted Processing
```bash
python person_detector.py -i long_video.mp4 --resume 50000
```

### Auto-Download Model
```bash
python person_detector.py -i input.mp4 --auto-download
```

### Force Re-Download Model
```bash
python person_detector.py -i input.mp4 --force-download
```

## Command Line Options

- `-i, --input`: Input video file path (required)
- `-o, --output`: Output JSON file for detailed results (optional)
- `--confidence`: Detection confidence threshold (default: 0.3)
- `--model`: Path to YOLOv11 model file (default: yolo11n.pt)
- `--resume`: Resume processing from specific frame number (optional)
- `--save-interval`: Save progress every N frames (default: 1000)
- `--auto-download`: Automatically download model if not found
- `--force-download`: Force download model even if it exists

## Output

The tool provides:

1. **Real-time Detection**: Shows timestamps as people are detected
2. **Progress Updates**: Displays processing progress
3. **Summary Report**: Final summary with detection timeline
4. **JSON Export** (optional): Detailed results with metadata

### Example Output for Long Video
```
🚀 Initializing Person Detector...
✓ YOLOv11 model loaded from yolo11n.pt
📹 Analyzing video: security_camera_3days.mp4
📊 Video info: 259,200 frames, 30.00 FPS
📅 Duration: 2d 12h (216,000 seconds)
⏱️  Estimated processing time: 2.4 days
🔍 Processing frames...
👤 Person detected at 0:00:15 (1 people)
   Progress: 0.1% (259/259,200) - ETA: 2.4d
👤 Person detected at 0:02:30 (2 people)
   Progress: 0.2% (518/259,200) - ETA: 2.3d
💾 Progress saved (0.5% complete)
   Progress: 0.5% (1,296/259,200) - ETA: 2.2d

============================================================
📋 DETECTION SUMMARY
============================================================
Video: security_camera_3days.mp4
Total detection events: 47
Total people detected: 89
Average people per detection: 1.9

📅 Hourly Summary:
  Hour 0: 12 people detected
  Hour 1: 8 people detected
  Hour 2: 15 people detected
  Hour 3: 3 people detected
  ...

📅 Detection Timeline:
   1. 0:00:15 - 1 person(s)
   2. 0:02:30 - 2 person(s)
   3. 0:05:45 - 1 person(s)
   ... (27 more detections) ...
   45. 2d 11:30:15 - 2 person(s)
   46. 2d 11:45:30 - 1 person(s)
   47. 2d 11:58:45 - 3 person(s)
============================================================
```

## Supported Video Formats

The tool supports all video formats that OpenCV can read, including:
- MP4, AVI, MOV, MKV
- WebM, FLV, WMV
- And many others

## Performance Notes

- Processing speed depends on your hardware and video resolution
- GPU acceleration is automatically used if available
- Lower confidence thresholds may detect more people but with more false positives
- The tool processes frames sequentially for accurate timestamp reporting
- **For very long videos (days):**
  - Progress is automatically saved every 5 minutes
  - ETA calculations help estimate completion time
  - Processing can be resumed from any frame if interrupted
  - Memory usage is optimized for long-running operations

## Troubleshooting

**Model not found error:**
- Use `--auto-download` to automatically download the model
- Or manually download `yolo11n.pt` and place it in the current directory
- Check your internet connection if auto-download fails

**Video file not found:**
- Check the file path is correct
- Ensure the video file exists and is readable

**Memory issues:**
- Try processing shorter videos first
- Close other applications to free up memory
- For very long videos, consider using `--save-interval` to save more frequently

**Processing interrupted:**
- Use `--resume <frame_number>` to continue from where you left off
- Check the JSON output file for the last processed frame
