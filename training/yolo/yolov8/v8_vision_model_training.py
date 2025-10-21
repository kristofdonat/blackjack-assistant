import os
import subprocess

# Configuration
DATASET_DIR = r"G:\ddc_blackjack\cards_withbg"  # note the raw string for backslashes
DATA_YAML_PATH = os.path.join(DATASET_DIR, "data.yaml")
MODEL = "yolov8m.pt"  # Choose from yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt
IMG_SIZE = 640
BATCH_SIZE = 16
EPOCHS = 100
RUN_NAME = "cards_yolov8_blackjack"

# Step 1: Install Ultralytics (YOLOv8)
subprocess.run(["pip", "install", "ultralytics"], check=True)

# Step 2: Train the model
command = [
    "yolo", "detect", "train",
    f"data={DATA_YAML_PATH}",
    f"model={MODEL}",
    f"epochs={EPOCHS}",
    f"imgsz={IMG_SIZE}",
    f"batch={BATCH_SIZE}",
    f"name={RUN_NAME}"
]

subprocess.run(command, check=True)

print("✅ Training completed. Check runs/detect/train/ for results.")
