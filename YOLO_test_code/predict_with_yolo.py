import torch 
from ultralytics import YOLO
import os 


cwd = os.getcwd()

model_path = os.path.join(cwd, 'models/yolov8n_fish_trained_lgds.pt')

model = YOLO(model_path)



results = model.predict(source ='/media/haksel/KINGSTON/20240831_172402/20240831.172835.545.001356.jpg')
#results = model.predict(source ='/media/haksel/KINGSTON/20240831_172402/20240831.172450.455.000005.jpg')

for result in results:
    boxes = result.boxes
    result.show()

