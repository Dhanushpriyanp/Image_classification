# from yolov5 import YOLO
import time
from ultralytics import YOLO
import ultralytics
import cv2, os
import numpy as np

model = YOLO("/Users/dhanushpriyan/Downloads/v13v9.pt")

cap = cv2.VideoCapture(
    "/Users/dhanushpriyan/Downloads/WhatsApp Video 2024-05-12 at 12.33.50.mp4"
)
index = 0
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_video_path = "/Users/dhanushpriyan/Desktop/runs/testing/predicted_video.mp4"
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # codec for MP4 video format
out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_width, frame_height))

while True:
    index += 1
    # Capture frame-by-frame
    ret, frame = cap.read()

    a = model.predict(source=frame, imgsz=640, conf=0.5)[0]
    # a.plot(show=True)

    boxes = a.boxes
    # print(boxes)
    high_confidence_color = (0, 0, 255)  # Green

    if len(boxes.cls[(boxes.cls == 1)]) > 0:
        bead_indices = boxes.cls == 1
        bead_xys = boxes.xyxy[bead_indices]

        conf = boxes.conf[bead_indices]
        print(a.verbose(), conf)
        # Loop through high-confidence bead boxes
        for i in range(len(bead_xys)):

            x1, y1, x2, y2 = bead_xys[i].cpu().numpy().astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), high_confidence_color, 2)
            cv2.putText(
                frame,
                f"Beads {conf[i]:.2f}",
                (int(x1), int(y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                high_confidence_color,
                2,
            )
    out.write(frame)

    cv2.imshow("Predicting video", frame)
    # cv2.imwrite(
    #     output_video_path, frame
    # )
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close OpenCV windows
cap.release()
out.release()
cv2.destroyAllWindows()
