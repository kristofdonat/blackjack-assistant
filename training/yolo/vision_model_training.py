# train_yolov5_cards.py

import os
import subprocess

# Configuration
DATASET_DIR = "G:\\ddc_blackjack\cards_withbg"  #Adjust path if necessary
DATA_YAML_PATH = os.path.join(DATASET_DIR, "data.yaml")
WEIGHTS = "yolov5m.pt"  #Choose from yolov5n.pt, yolov5s.pt, yolov5m.pt, yolov5l.pt
IMG_SIZE = 416
BATCH_SIZE = 32
EPOCHS = 70
RUN_NAME = "cards_yolov5s"

#Step 1: Clone YOLOv5 if not already done
if not os.path.exists("yolov5"):
    subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5"])

#Step 2: Install requirements
subprocess.run(["pip", "install", "-r", "yolov5/requirements.txt"])

#Step 3: Run training
command = [
    "python", "yolov5/train.py",
    "--img", str(IMG_SIZE),
    "--batch", str(BATCH_SIZE),
    "--epochs", str(EPOCHS),
    "--data", DATA_YAML_PATH,
    "--weights", WEIGHTS,
    "--name", RUN_NAME
]

subprocess.run(command)

print("Training completed. Check runs/train/ for results.")