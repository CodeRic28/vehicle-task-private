import os

from ultralytics import YOLO
import cv2


VIDEOS_DIR = os.path.join('.', 'videos')


video_path = os.path.join(VIDEOS_DIR, 'Vehicle_count_test.mp4')
cap = cv2.VideoCapture(video_path)


model_path = os.path.join('.', 'model','runs', 'detect', 'train', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)
print(os.getcwd())
img = os.path.join('.','data','images','Auto','Datacluster Auto (1).jpg')
results = model(img,show=True)
cv2.waitKey(0)



# ret=1
# while ret:
#     ret, frame = cap.read()
#     H, W, _ = frame.shape
#
#     results = model(frame)[0]
#     print(results)

    # cv2.imshow('img',frame)
    # cv2.waitKey(1)