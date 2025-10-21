import cv2
import argparse
from ultralytics import YOLO

def main(video_source, model_weights):
    #Load pre-trained YOLOv8 model
    model = YOLO(model_weights)
    model.conf = 0.5  # Confidence threshold (optional in YOLOv8)

    # Open video source
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Unable to open video source: {video_source}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #YOLOv8 detection (stream=true for faster processing)
        results = model(frame, stream=True, verbose=False)

        #annotate and display results
        for r in results:
            annotated_frame = r.plot() 
            cv2.imshow("Blackjack card detection (YOLOv8), model: " + model_weights, annotated_frame)

        #exit with 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0",
                        help="Video source (pl. 0 = camera, 'video.mp4' = file)")
    parser.add_argument("--weights", type=str, default="yolov8_30.pt",
                        help="Path to YOLOv8 model weights file")
    args = parser.parse_args()

    model_weights = args.weights
    #if its int, convert to int (for camera index)
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    main(source, model_weights)

#usage example:
#python video_processor.py --source cards1.mp4 --weights yolov8_100.pt
#or
#python video_processor.py --source 0 --weights yolov8_100.pt  #for webcam