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
import time 
import csv

cwd = os.getcwd() # Fish_no_fish project root

model_path = os.path.join(cwd, 'models/yolov8n_fish_trained_lgds.pt')

print("hello world")

model = YOLO(model_path)

images_dir = os.path.join(cwd, 'test_code_2/rcnn_training/fish_data/fish_images')



def process_images(images_path):
    count = 0 
    processed_imgs = set() 
    meta_data = []

    while(True): 
        all_images = os.listdir(images_path)
        num_images = len(all_images)

        # Get all the unprocessed images
        images = [cur_image for cur_image in all_images if cur_image.endswith(('.jpg', '.png')) and cur_image not in processed_imgs]
        
        for image in images:
            image_path = os.path.join(images_path, image)

            # Run inference and store number of detections
            inference = model(image_path)
            detection_count = len(inference.boxes)
            meta_data.append([image, detection_count])
            


            processed_imgs.add(image)
            print(count)
            # print(processed_imgs)
            count = count + 1 

        # Wait a second and check if any new images were added
        time.sleep(10)
        all_images = os.listdir(images_path)
        new_num_images = len(all_images)
        if(new_num_images == num_images):
            print("No new images, exiting...")
            print(meta_data)
            exit()
        


process_images(images_dir)

# Run inference on every image 
# Store the image name and number of detections in csv file 
# Parse the csv, find max number of detections during each interval (maybe separate program)
# Save best images with inference data on them
# Display to user 
