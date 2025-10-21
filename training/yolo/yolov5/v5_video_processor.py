import cv2
import torch
import argparse

def main(video_source):
    #loading yolo5 model
    model = torch.hub.load("training/yolo/yolov5/yolov5", "custom", path="training/yolo/yolov5/yolov5m_e100.pt", source="local")
    model.conf = 0.5
    #opening video source
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Unable to open video source: {video_source}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #detecting with yolo5
        results = model(frame)

        #Render the results on the frame with yolo5 built-in method
        annotated_frame = results.render()[0]

        #Display the frame with annotations
        cv2.imshow("Blackjack card detection", annotated_frame)

        #exit with 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Video source (pl. 0 = camera, 'video.mp4' = file)")

    args = parser.parse_args()

    #if its int, convert to int (for camera index)
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    main(source)

#usage example:
#python video_processor.py --source cards1.mp4
#or
#python video_processor.py --source 0  #for webcam