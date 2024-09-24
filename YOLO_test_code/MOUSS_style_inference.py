'''
Spec: a single script that runs inference as similar
to MOUSS_FNF would, while still running on a PC. 

Reference: 
 - Inference options:
 https://docs.ultralytics.com/modes/predict/#inference-arguments
 -
'''

import torch 
from ultralytics import YOLO, solutions
# import cv2
import os
import time 
import csv
import pandas as pd 


class FishDetector:
    def __init__ (self):
        self.cwd = os.getcwd() # Fish_no_fish project root

        ## Specify path to model
        self.model_path = os.path.join(self.cwd, 'models/yolov8n_fish_trained_lgds.pt')

        self.model = YOLO(self.model_path)

        ## Specify path to images
        self.images_dir = os.path.join(self.cwd, 'test_code_2/rcnn_training/fish_data/fish_images')

        self.wait_for_new_images_time = 4 # Seconds, before the program starts analyzing metrics
        self.need_header = True

        self.analyze_every_x_frame = 1  # Analyze every (2nd, 3rd, 4th,... ect) frame
                                        # This must be a tunable parameter

        self.metrics_chunk_size = (6 * 60) / self.analyze_every_x_frame  # 6 frames per second * 60 frames in a minute / how many we actually analyze 
                                        # batch size MUST represent 1 minute of data 
                                        # for statistical accuracy. 
                                        # Max = 360 (frames in one minute of data)

        self.num_images = 0 
        
        print("image batch size = ", self.metrics_chunk_size)
    def write_line(self, directory, image_name, num_detections):
        '''
        Write one line in the image/detection CSV.
        Args: - (string) directory where csv was created [should be same as above]
        - (string) name of image as it will appear in the csv
        - (int) number of detections associated with current image
        '''
        df = pd.DataFrame([[image_name, num_detections]], columns=['Image', 'Num_detections'])
        df.to_csv(directory, mode='a', header=self.need_header, index=False) 
        self.need_header = False

    def process_images(self, images_path):
        '''
        Run inference on every image in the current directory.
        Wait ten seconds (change for MOUSS), and run inference on
        any new images or exit. Add image name and num detections to
        the CSV.
        Args: - (string) path to image folder
        '''
        count = 0 
        processed_imgs = set() 

        while(True): 
            all_images = os.listdir(images_path)
            self.num_images = len(all_images)

            # Get all the unprocessed images
            images = [cur_image for cur_image in all_images if cur_image.endswith(('.jpg', '.png')) and cur_image not in processed_imgs]
            
            for image in images:
                image_path = os.path.join(images_path, image)

                # Run inference and store number of detections
                inference = self.model(image_path)

                for inference in inference:
                    boxes = inference.boxes
                    detection_count = len(boxes)

                # meta_data.append([image, detection_count]) # write as you go, instead of at the end 
                self.write_line(os.path.join(self.cwd, 'detections.csv'), image, detection_count)
                


                processed_imgs.add(image)
                print(count)
                # print(processed_imgs)
                count = count + 1 

            # Wait a second and check if any new images were added
            time.sleep(self.wait_for_new_images_time)
            all_images = os.listdir(images_path)
            new_num_images = len(all_images)
            if(new_num_images == self.num_images):
                print("No new images, exiting...")
                break

    def get_metrics(self, path_to_csv):
        chunk_size = self.metrics_chunk_size

        for i in range(int(self.num_images / self.metrics_chunk_size) + 1): 
            '''
            Analyze all the data in chunks that equate to one minute of images 
            '''
            tmp_table = []
            for df in pd.read_csv(path_to_csv, chunksize=chunk_size):
                # tmp_table.append([df.Image, df.Num_detections])# TODO: this is coming off as two separate tables but should be one table. 
                pass 
            # print("temp table = ")
            # print(tmp_table)
            # print("table shape = ")
            # print(len(tmp_table))

def main():

    fishDetector = FishDetector() 
    # create_csv(os.path.join(cwd, 'detections.csv')) # Use current directory for now
    fishDetector.process_images(fishDetector.images_dir)    # TODO: maybe move 
    csv_path = os.path.join(fishDetector.cwd, 'detections.csv')
    fishDetector.get_metrics(csv_path)



if __name__ == '__main__': 
    main()

# Run inference on every image 
# Store the image name and number of detections in csv file 
# Parse the csv, find max number of detections during each interval (maybe separate program)
# Save best images with inference data on them
# Display to user 
