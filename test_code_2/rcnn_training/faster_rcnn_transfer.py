# https://www.youtube.com/watch?v=qC4yEiJOJtM
# Use this video to write a script ^ 
# https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline
# Reference this as well ^
# https://github.com/harshatejas/pytorch_custom_object_detection/tree/main
# And this 

import matplotlib.pyplot as plt

# 6:32 - list of all imports 
import torch 
import torchvision 
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor 
from torch.utils.data import DataLoader # Used to translate my dataset to torch 
#from torchvision import transforms as T 

import create_dataset as mydata     # Helper functions by Aks


# Select and easy object detection model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

num_classes = 3 # fish, bait_arm, background
in_features = model.roi_heads.box_predictor.cls_score.in_features   # Number of inputs to bbox and classifier layers (that we will train)
#print('in_features = ', in_features)

# Create the new box predictor with my custom number of classes 
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

#print(model)

# pull in my dataset    
mydataset = mydata.MyFishDataset() # Comes with file pointer 


# Collate function to handle varying sizes of bboxes and labels
def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = []
    for item in batch:
        labels = torch.tensor(item[1], dtype=torch.int64)
        bboxes = torch.tensor(item[2], dtype=torch.float32)
        targets.append({"boxes": bboxes, "labels": labels})
    return images, targets

# Use the collate_fn in your DataLoader
train_dataloader = DataLoader(mydataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader

import create_dataset as mydata     # Helper functions by Aks

# Select an object detection model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

num_classes = 3  # fish, bait_arm, background
in_features = model.roi_heads.box_predictor.cls_score.in_features

# Create the new box predictor with custom number of classes
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Pull in your dataset
mydataset = mydata.MyFishDataset()

# Collate function to handle varying sizes of bboxes and labels
def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = []
    for item in batch:
        labels = torch.tensor(item[1], dtype=torch.int64)
        bboxes = torch.tensor(item[2], dtype=torch.float32)
        targets.append({"boxes": bboxes, "labels": labels})
    return images, targets

# Use the collate_fn in your DataLoader
train_dataloader = DataLoader(mydataset, batch_size=3, shuffle=False, collate_fn=collate_fn)

# # Display image with bounding boxes and labels
# def display_image_with_boxes(image, boxes, labels, label_dict):
#     fig, ax = plt.subplots(1, figsize=(12, 9))
#     ax.imshow(image)
#     for box, label in zip(boxes, labels):
#         box = box.tolist()
#         # Convert box to format (x0, y0, x1, y1)
#         x0, y0, x1, y1 = box
#         width = x1 - x0
#         height = y1 - y0 

#         rect = patches.Rectangle((x0, y0), width, height, linewidth=1, edgecolor='r', facecolor='none')
#         ax.add_patch(rect)
#         plt.text(x0, y0, label_dict[label.item()], color='red', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
#     plt.show()

# # Fetch a batch of data
# images, targets = next(iter(train_dataloader))

# # Display the first image with its bounding boxes and labels
# image = images[0].permute(1, 2, 0).numpy()  # Convert image to HWC format and numpy array
# boxes = targets[0]['boxes'].numpy()  # Convert to numpy array
# labels = targets[0]['labels']  # Tensor of labels

# Display image with bounding boxes and labels
# def display_image_with_boxes(image, boxes, labels, label_dict):
#     fig, ax = plt.subplots(1, figsize=(12, 9))
#     ax.imshow(image)
#     for box, label in zip(boxes, labels):
#         box = box.tolist()
#         # Convert box to format (x0, y0, x1, y1)
#         x0, y0, x1, y1 = box
#         width = x1 - x0
#         height = y1 - y0 

#         rect = patches.Rectangle((x0, y0), width, height, linewidth=1, edgecolor='r', facecolor='none')
#         ax.add_patch(rect)
#         plt.text(x0, y0, label_dict[label.item()], color='red', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
#     plt.show()

# # Fetch a batch of data
# images, targets = next(iter(train_dataloader))

# # Display the first image with its bounding boxes and labels
# image = images[0].permute(1, 2, 0).numpy()  # Convert image to HWC format and numpy array
# boxes = targets[0]['boxes'].numpy()  # Convert to numpy array
# labels = targets[0]['labels']  # Tensor of labels

# Display image with bounding boxes and labels
def display_image_with_boxes(dataloader, idx, label_dict):
    images, targets = next(iter(dataloader))
    # Display the first image with its bounding boxes and labels
    image = images[idx].permute(1, 2, 0).numpy()  # Convert image to HWC format and numpy array
    boxes = targets[idx]['boxes'].numpy()  # Convert to numpy array
    labels = targets[idx]['labels']  # Tensor of labels
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    for box, label in zip(boxes, labels):
        box = box.tolist()
        # Convert box to format (x0, y0, x1, y1)
        x0, y0, x1, y1 = box
        width = x1 - x0
        height = y1 - y0 

        rect = patches.Rectangle((x0, y0), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x0, y0, label_dict[label.item()], color='red', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.show()

def main():
    # display_image_with_boxes(image, boxes, labels, mydata.labels_dict)
    print('bob le sponge ')
    display_image_with_boxes(train_dataloader, 4, mydata.labels_dict) # only 0 through 2????f
    


if __name__ == '__main__':
    main()