'''
Spec: when you run this program you get a fresh
DETR model in this directory.
'''
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection


model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101", revision="no_timm")

torch.save(model.cpu(), 'DETR.pt' ) # comment for safety
# torch.save(model.cpu(), 'detr_test_code/DETR_custom_out.pt' )