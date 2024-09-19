import torch 
from ultralytics import YOLO
import cv2

print("hello world")

model = YOLO("yolov8n.pt")

results = model.predict(source ='LeFish.png')

img = cv2.imread("LeFish.png")
results = model.predict(source=img, show=True)

while(True):
    pass 