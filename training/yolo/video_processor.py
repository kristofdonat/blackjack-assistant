import cv2
import torch
import argparse

def main(video_source):
    #loading yolo5 model
    model = torch.hub.load("training/yolo/yolov5", "custom", path="training/yolo/yolov5m.pt", source="local")
    model.conf = 0.5
    #opening video source
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Nem sikerült megnyitni a videó forrást: {video_source}")
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
        cv2.imshow("Blackjack kártya detektálás", annotated_frame)

        #exit with 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Video forrás (pl. 0 = kamera, 'video.mp4' = fájl)")

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