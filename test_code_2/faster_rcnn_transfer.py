# https://www.youtube.com/watch?v=qC4yEiJOJtM
# Use this video to write a script ^ 
# https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline
# Reference this as well ^

# 6:32 - list of all imports 
import torchvision 
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor 
#from torchvision import transforms as T 

import pandas as pd 


# Select and easy object detection model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
print(model)

num_classes = 3 # fish, bait_arm, background
in_features = model.roi_heads.box_predictor.cls_score.in_features   # Number of inputs to bbox and classifier layers (that we will train)
print('in_features = ', in_features)

# Create the new box predictor with my custom number of classes 
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Create training data variable (horrible name I know)
#train = pd.read_csv()

# Pause here to create a small dataset