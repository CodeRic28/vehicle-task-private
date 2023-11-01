import os
import time
from ultralytics import YOLO
import cv2
import numpy as np
import cvzone

VIDEOS_DIR = os.path.join('.','videos')

video_path = os.path.join(VIDEOS_DIR,'Vehicle_count_test.mp4')
video_path_out = f"{video_path}_out.mp4"

model_path = os.path.join('.','model','runs','detect','train','weights','last.pt')
# model_path = os.path.join('model','yolov8n.pt')
# Load the model
model = YOLO(model_path)  # load a custom model

cap = cv2.VideoCapture(video_path)
cap.set(3,720)
cap.set(4,480)


while True:
    # success, img = cap.read()
    # if not success:
    #     break
    # results = model(img, stream=True)
    #
    # for r in results:
    #     boxes = r.boxes
    #     for box in boxes:
    #         x1, y1, x2, y2 = box.xywh[0]
    #         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    #
    #         w, h = x2-x1, y2-y1
    #         cvzone.cornerRect(img, (x1,y1,w,h))
    #
    # cv2.imshow("Image",img)
    # cv2.waitKey(1)

    ret, frame = cap.read()
    H, W, _ = frame.shape
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        print(result)
    #
    #     if score > .1:
    #         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
    #         cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
    #
    # cv2.imshow("Image",frame)
    # cv2.waitKey(1)
# cap.release()
# cv2.destroyAllWindows()

import os

# from ultralytics import YOLO
# import cv2
#
#
# VIDEOS_DIR = os.path.join('videos')
#
# video_path = os.path.join(VIDEOS_DIR, 'Vehicle_count_test.mp4')
# video_path_out = '{}_out.avi'.format(video_path)
#
# cap = cv2.VideoCapture(video_path)
#
# model_path = os.path.join('.', 'model','runs', 'detect', 'train2', 'weights', 'last.pt')
#
# # Load a model
# model = YOLO(model_path)  # load a custom model
#
# threshold = 0.5
# success = True
# while success:
#
#     success, frame = cap.read()
#     H, W, _ = frame.shape
#     out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MJPG'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))
#     results = model(frame)[0]
#
#     for result in results.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = result
#
#         if score > threshold:
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
#             cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
#     if H <= 0 or W<=0:
#         break
#     out.write(frame)
#     ret, frame = cap.read()
#
# cap.release()
# cv2.destroyAllWindows()