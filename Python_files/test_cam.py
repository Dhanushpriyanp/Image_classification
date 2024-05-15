# from yolov5 import YOLO
import time
from ultralytics import YOLO
import ultralytics
import cv2, os
import numpy as np

model = YOLO("/Users/dhanushpriyan/Downloads/v18.pt")

cap = cv2.VideoCapture(0)
index = 0
while True:
    index += 1
    # Capture frame-by-frame
    ret, frame = cap.read()

    a = model.predict(source=frame, imgsz=640, conf=0.5, show=True)[0]
    # a.plot(show=True)

    boxes = a.boxes
    # print(boxes)
    high_confidence_color = (0, 0, 255)  # Green

    if len(boxes.cls[(boxes.cls == 0)]) > 0:
        bead_indices = boxes.cls == 0
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

    cv2.imshow("Predicting", frame)
    cv2.imwrite(
        f"/Users/dhanushpriyan/Desktop/runs/testing/predicted_image{index}.jpg", frame
    )
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
