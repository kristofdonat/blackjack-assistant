# blackjack-assistant
An AI assistant trained for the soul purpose of bringing you towards the winning steps in blackjack

## Dataset

The YOLOv5 model was trained on the [Cards Image Dataset-Classification](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification/data) from Kaggle.

- Source: [Kaggle](https://www.kaggle.com/)
- Dataset Author: gpiosenka
- License: Open for public use (as per Kaggle page)

The Torch model was trained on the [900,000 Hands of BlackJack Results](https://www.kaggle.com/datasets/mojocolors/900000-hands-of-blackjack-results/data) from Kaggle.

- Source: [Kaggle](https://www.kaggle.com/)
- Dataset Author: mojocolors
- License: Open for public use (as per Kaggle page)

## Project Structure & File Roles

- `training/`
  - `torch/`
    - `blackjackmodel_training.py`: Trains the blackjack decision model using card hand data.
    - `blackjackmodel_evaltest.py`: Loads and tests the trained blackjack model for predictions.
    - `blackjack_model.pth`: Saved weights of the trained blackjack model.
    - `data/`
      - `blkjckhands.csv`: Dataset of blackjack hands used for training.
  - `yolo/`
    - `vision_model_training.py`: Script to train a YOLOv5 model for card image detection/classification.
    - `yolov5m.pt`, `yolov5s.pt`: Pretrained YOLOv5 weights.
    - `yolov5/`: YOLOv5 source code and utilities (training, validation, export, etc.).
      - `train.py`: Main training script for YOLOv5 models.
      - `detect.py`: Script for running inference with trained YOLOv5 models.
      - `requirements.txt`: Python dependencies for YOLOv5.
      - *(other supporting files for YOLOv5 pipeline)*
- `README.md`: Project overview and documentation.