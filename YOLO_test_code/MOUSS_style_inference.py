'''
Spec: a single script that runs inference as similar
to MOUSS_FNF would, while still running on a PC. 

Reference: 
 - Inference options:
 https://docs.ultralytics.com/modes/predict/#inference-arguments
 -
'''

import torch 
from ultralytics import YOLO
# import cv2
import os

cwd = os.getcwd()

model_path = os.path.join(cwd, 'models/yolov8n_fish_trained_lgds.pt')

print("hello world")

model = YOLO(model_path)

inference_image = os.path.join(cwd, "LeFish.png")
results = model(source=inference_image) # save_txt=True 
                                         # Save output classes as class, x_center, y_center, width, height, confidence


for result in results:
    boxes = result.boxes
    probs = result.probs
    oriented_bounding_boxes = result.obb
    result.show()
