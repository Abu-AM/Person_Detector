# Configuration for drone_security system

# Video source: 0 for webcam, or RTSP stream URL
VIDEO_SOURCE = 2 #"sample_feeds/4feeds.webm"

# Test mode: Set to True to use a test image instead of webcam
TEST_MODE = False  # Set to True if you want to test without webcam

# Device for inference: 'auto', 'cpu', 'cuda', or specific device
DEVICE = 'auto'

# Minimum confidence for detection
CONFIDENCE_THRESHOLD = 0.3  # Reverted to default

# Auto-scaling settings
MAX_PROCESS_SIZE = 640  # Maximum dimension for processing (maintains aspect ratio)
# Higher values = better accuracy but slower processing
# Lower values = faster processing but potentially lower accuracy

# Target classes to detect (None = detect all classes)
# Common options: ['person'], ['person', 'car'], ['person', 'car', 'truck']
TARGET_CLASSES = ['person']  # Only detect people

# Output directory for saved images and logs
OUTPUT_DIR = 'output/' 