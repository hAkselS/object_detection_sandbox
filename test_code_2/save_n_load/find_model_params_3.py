'''
Spec: Find and print the model parameters for future use. 
'''

import torch 
#from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#from torchsummary import summary
from torchinfo import summary 


#model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101", revision="no_timm")
model = torch.jit.load('test_code_2/model_scripted.pt')

#print(model)
# summary(model, (3,800,800))
summary(model, input_size=(1, 3, 1024, 1024))

# labels = model.config.id2label
# print("Model classes:")
# for idx, label in labels.items():
#     print(f"Class {idx}: {label}")