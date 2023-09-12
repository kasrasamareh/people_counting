
import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Using GPU or CPU

cap = cv2.VideoCapture("video.mp4")  # Loading Video

model = YOLO("yolov8n.pt").to(device) # Download or use YOLO Weights

class_name = ["person"]

original_mask = cv2.imread("mask.png") # Loading The Mask


tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3) # Tracking

# Initializing Variables
horizental_limits = [181, 480, 700, 406]
vertical_limits = [250, 0, 700, 800]
total_counts_id = []
total_count_counter = 0

# 
while True:
    success, img = cap.read() # Reading each frame
    if (not success) : break

    # Applying the mask over each frame
    mask = cv2.resize(original_mask, (img.shape[1], img.shape[0]))
    img_region = cv2.bitwise_and(img, mask)
    
    results = model(img_region, stream=True) # Applying model over each frame

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            current_class = class_name[cls]

            if current_class == "person" and conf > 0.3:
                current_array = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_array))

    results_tracker = tracker.update(detections)

    cv2.line(img, (horizental_limits[0], horizental_limits[1]), (horizental_limits[2], horizental_limits[3]), (0, 0, 255), 5)
    cv2.line(img, (vertical_limits[0], vertical_limits[1]), (vertical_limits[2], vertical_limits[3]), (0, 0, 255), 5)
    for result in results_tracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        nx, ny = x1 + w // 2, y1 + h // 2

        if horizental_limits[0] < nx < horizental_limits[2] and horizental_limits[1] - 100 < ny < horizental_limits[1] + 100:
            if total_counts_id.count(id) == 0:
                total_counts_id.append(id)
                if nx <= (vertical_limits[0]+vertical_limits[2])//2 : 
                    total_count_counter += 1
                else : 
                    total_count_counter -= 1
                cv2.line(img, (horizental_limits[0], horizental_limits[1]), (horizental_limits[2], horizental_limits[3]), (0, 255, 0), 5)

    cv2.putText(img,str(total_count_counter),(700,200),cv2.FONT_HERSHEY_PLAIN,10,(255,255,255),12)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
