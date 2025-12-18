# blackjack-assistant
An AI assistant trained for the soul purpose of bringing you towards the winning steps in blackjack

This project combines two ML components:
1. **YOLO (YOLOv5 / YOLOv8)** for real-time playing-card detection from a webcam/video stream.
2. **PyTorch policy model** that predicts whether you should **draw another card (HIT)** based on:
   - `x1 = number of cards in hand`
   - `x2 = total Blackjack value of the hand`
   - `y ∈ {True, False}` (draw vs. stop)

> **Exit:** press `Q` anytime.

---

## What the watcher does (runtime)
- Reads **live video** from a webcam (or a video file).
- If a card appears, it **detects and draws a bounding box** around it.
- Maintains a “hand state” by **accumulating cards** while they remain visible.
- Prints to console when a **new card is added**:
  - number of cards
  - hand total
  - “Draw?” decision (policy model)
- Detects “new deal / new hand” when:
  - the camera sees **no cards for a sustained time**, then
  - cards appear again → **new hand starts**
- Writes a **CSV logfile** at the end of the run.

## Recommended running
Paste this into the terminal, or just run blackjack_watcher.py with the required arguments
`python blackjack_watcher.py --source 0`

The project can be run with the following command-line arguments:

- `--source` (str, default: `"0"`)
  - **What it does:** Sets the video input source.
  - **Default behavior:** `"0"` uses the default camera/webcam. You can also pass a video file path (e.g., `"video.mp4"`).
  - **Sample videos:** `"training/sample_videos"` we have some sample videos to try out with, if you don't have a default camera (e.g., `"cards1.mp4"`).

- `--weights` (str, default: `"training/yolo/yolov8/yolov8m_e100.pt"`)
  - **What it does:** Path to the YOLO model weights used for detection.
  - **Default behavior:** Loads YOLO weights from `training/yolo/yolov8/yolov8m_e100.pt`.

- `--policy` (str, default: `"training/torch/blackjack_model.pth"`)
  - **What it does:** Path to the PyTorch policy model used to decide **draw / no draw**.
  - **Default behavior:** Loads the policy model from `training/torch/blackjack_model.pth`.

- `--logdir` (str, default: `"logs"`)
  - **What it does:** Directory where the run will write log output (CSV).
  - **Default behavior:** Saves logs into the `logs/` folder.


## Dataset

The YOLOv5 & later YOLOv8 model was trained on the [Cards Image Dataset-Classification](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification/data) from Kaggle.

- Source: [Kaggle](https://www.kaggle.com/)
- Dataset Author: gpiosenka
- License: Open for public use (as per Kaggle page)

The Torch model was trained on the [900,000 Hands of BlackJack Results](https://www.kaggle.com/datasets/mojocolors/900000-hands-of-blackjack-results/data) from Kaggle.

- Source: [Kaggle](https://www.kaggle.com/)
- Dataset Author: mojocolors
- License: Open for public use (as per Kaggle page)

## Project Structure & File Roles

- `blackjack_watcher.py`: Main runtime script. Runs webcam/video inference with YOLO, tracks cards, computes hand total, predicts HIT/stand, and writes the final CSV logfile.
- `logs/`: Output folder for generated CSV logfiles (one or more rows per hand depending on logging mode).
- `training/`
  - `sample_videos/`
    - `cards1.mp4`, `cards2.MP4`, `cards3.MP4`: Sample videos for quick testing without a webcam.
  - `torch/`
    - `blackjackmodel_training.py`: Trains the blackjack decision (HIT/stand) model using hand-level data.
    - `blackjackmodel_training_v2.py`: Improved training script (includes feature normalization and a checkpoint-style save format).
    - `blackjackmodel_evaltest.py`: Loads and tests the trained blackjack decision model for predictions.
    - `blackjack_model.pth`: Saved weights/checkpoint of the trained blackjack decision model (current).
    - `blackjack_model_old.pth`: Older saved model weights (legacy).
    - `data/`
      - `blkjckhands.csv`: Dataset of blackjack hands used for training the decision model.
  - `yolo/`
    - `yolov5/`
      - `v5_vision_model_training.py`: Script to train a YOLOv5 model for playing-card detection (and to bootstrap/clone YOLOv5 if needed).
      - `v5_video_processor.py`: Runs YOLOv5 inference on a video for quick validation/debugging.
      - `yolov5m_e100.pt`: Trained YOLOv5 weights (epoch 100).
      - `yolov5s.pt`: Pretrained YOLOv5s weights (baseline).
      - `yolov5/`: Placeholder folder for the Ultralytics YOLOv5 source repo (often cloned/downloaded by the training script).
    - `yolov8/`
      - `v8_vision_model_training.py`: Script to train a YOLOv8 model for playing-card detection.
      - `v8_video_processor.py`: Runs YOLOv8 inference on a video for quick validation/debugging.
      - `yolov8m_e100.pt`: Trained YOLOv8 weights (epoch 100).
- `README.md`: Project overview and documentation.