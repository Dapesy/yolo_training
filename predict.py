import os 
from ultralytics import YOLO as yolo
import cv2 as cv
import numpy as np

# # load a my yolov8 model 

# model = yolo('best.pt','v8')
# video_dir = '/Users/mypc/Desktop/yolo/car_annotation/slow-traffic-small.mp4'
# cap =  cv.VideoCapture(video_dir)
# # print(cap.get(cv.CAP_PROP_FRAME_WIDTH))
# # print(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
# # fourcc = cv.VideoWriter_fourcc(*'XVID')
# # out = cv.VideoWriter('video.avi', fourcc, 20.0, (680, 480))
# # # all prop ar assgined to a numerical value, as 3 is to frame width and 4 is to frame height 
# # cap.set(3,1208)
# # cap.set(4,720)
# # print(cap.get(3))
# # print(cap.get(4))
# while cap.isOpened():
#     ret , frame = cap.read()
#     if ret == True : 
#         detect_params = model.predict(source=[frame], conf=0.45, save=True)

#         # out.write(frame)
#         cv.imshow('frame', frame)
#         if cv.waitKey(1) & 0xFF == ord('q') :
#            break
#     else:
#         break
# cap.release()
# cv.destroyAllWindows()

# load a pretrained model 
model = yolo('best.pt', 'v8')


# predict on an image
detection_output = model.predict(source='/Users/mypc/Desktop/yolo/car_annotation/slow-traffic-small.mp4', conf=0.25, save=True)

# display tensor array
print(detection_output)

# dispaly numpy array
print(detection_output[0].numpy())

