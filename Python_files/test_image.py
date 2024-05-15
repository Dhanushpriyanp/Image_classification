# from yolov5 import YOLO
import time
from ultralytics import YOLO
import ultralytics
import cv2, os
import numpy as np

model_bead = YOLO("/Users/dhanushpriyan/Downloads/v18.pt")
model_stone = YOLO("/Users/dhanushpriyan/Downloads/v13v9.pt")
model_tassel = YOLO("/Users/dhanushpriyan/Downloads/v18v9.pt")
# v18 is better for beads
# v13v9 is good for stones
# v13v9 la tassel mattum not good remaining good
# v18v9 and v18 la stones is not good
# v17 beads is over trained
# dp1 tassel over trained
# yolov9 average
path = "/Users/dhanushpriyan/Downloads/Jwellery_project/test/SSN_COLLEAGE"
# path = "/Users/dhanushpriyan/Downloads/Jwellery_project/testing"
# # path = "/Users/dhanushpriyan/Downloads/Homogeneous21/train/images"
# path = "/Users/dhanushpriyan/Downloads/testing/gold_chain_stones_beads_tassel"


# file_list = os.listdir(path)

# # Filter out only the image files (assuming images have extensions like .jpg, .png, etc.)
# image_files = [
#     file
#     for file in file_list
#     if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
# ]
image_c = 0
bead_c = 0
stone_c = 0
tassel_c = 0


# Release capture and close windows

# # Iterate over the image files
# for image_file in image_files:
#     image_c += 1

#     image_path = os.path.join(path, image_file)

results_stone = model_stone.predict(source=path, imgsz=640, conf=0.5)
results_bead = model_bead.predict(source=path, imgsz=640, conf=0.5)
results_tassel = model_tassel.predict(source=path, imgsz=640, conf=0.5)
    # a.plot(show=True)
# print(results_stone,results_bead,results_tassel)
for i in range(len(results_stone)):
    # print(results_stone[i],results_bead[i],results_tassel[i])
    boxes_stone = results_stone[i].boxes
    boxes_bead = results_bead[i].boxes
    boxes_tassel = results_tassel[i].boxes
    # print(boxes)

    image = cv2.imread(results_stone[i].path)

    # Define colors for different classes (adjust as needed)
    stone_color = (0, 255, 0)  # Green
    bead_color = (0, 0, 255)
    tassel_color = (255, 0, 0)
    # bead_confidences = boxes.conf[boxes.cls == 1]
    # stone_beads = bead_confidences[bead_confidences >= 0.3]

    if len(boxes_stone.cls[(boxes_stone.cls == 1)]) > 0:
        stone_indices = boxes_stone.cls == 1
        stone_xys = boxes_stone.xyxy[stone_indices]

        conf = boxes_stone.conf[stone_indices]
        stone_c += 1
        for s1 in results_stone[i].verbose().split(","):
            if "stone" in s1:
                print(s1, end=" ")
        # Loop through high-confidence stone boxes
        for i in range(len(stone_xys)):

            x1, y1, x2, y2 = stone_xys[i].cpu().numpy().astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), stone_color, 2)
            cv2.putText(
                image,
                f"Stones {conf[i]:.2f}",
                (int(x1), int(y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                stone_color,
                2,
            )
    if len(boxes_bead.cls[(boxes_bead.cls == 0)]) > 0:
        bead_indices = boxes_bead.cls == 0
        bead_xys = boxes_bead.xyxy[bead_indices]

        conf = boxes_bead.conf[bead_indices]
        bead_c += 1
        for b1 in results_bead[i].verbose().split(","):
            if "bead" in b1:
                print(b1)
        # Loop through high-confidence bead boxes
        for i in range(len(bead_xys)):

            x1, y1, x2, y2 = bead_xys[i].cpu().numpy().astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), bead_color, 2)
            cv2.putText(
                image,
                f"Beads {conf[i]:.2f}",
                (int(x1), int(y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                bead_color,
                2,
            )

    if len(boxes_tassel.cls[(boxes_tassel.cls == 2)]) > 0:
        tassel_indices = boxes_tassel.cls == 2
        tassel_xys = boxes_tassel.xyxy[tassel_indices]

        conf = boxes_tassel.conf[tassel_indices]
        tassel_c += 1
        for s1 in results_tassel[i].verbose().split(","):
            if "tassel" in s1:
                print(s1, end=" ")
        # Loop through high-confidence tassel boxes
        for i in range(len(tassel_xys)):

            x1, y1, x2, y2 = tassel_xys[i].cpu().numpy().astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), tassel_color, 2)
            cv2.putText(
                image,
                f"Tassels {conf[i]:.2f}",
                (int(x1), int(y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                tassel_color,
                2,
            )
    cv2.imshow(results_stone[i].path.split("/")[-1], image)
    cv2.waitKey(1)

    while cv2.getWindowProperty(results_stone[i].path.split("/")[-1], cv2.WND_PROP_VISIBLE) >= 1:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break  # Wait for 100 milliseconds (0.1 second)
    cv2.destroyAllWindows()
# print(image_c, bead_c, stone_c)
# print(bead_c / image_c)
# print(stone_c / image_c)
# print(tassel_c / image_c)
