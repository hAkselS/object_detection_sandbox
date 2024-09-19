# https://docs.ultralytics.com/datasets/detect/ 

from ultralytics import YOLO 


model = YOLO("yolov8n.pt") # Load the pretrained model 

# Note: ultralytics have optimized for m1 chip
# Use: train(..., device="mps") 
