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
import pandas as pd # Handle CSV file 
import numpy as np # TODO: remove, used to find table shape
import streamlit as st 
import plotly.express as px

class FishDetector:
    def __init__ (self):
        self.cwd = os.getcwd() # Fish_no_fish project root

        ## Specify path to model
        self.model_path = os.path.join(self.cwd, 'models/yolov8n_fish_trained_lgds.pt')

        self.model = YOLO(self.model_path)

        ## Specify path to images
        self.images_dir = os.path.join(self.cwd, 'test_code_2/rcnn_training/fish_data/fish_images')

        self.wait_for_new_images_time = 1 # Seconds, before the program starts analyzing metrics
        self.need_header = True

        self.analyze_every_x_frame = 1  # Analyze every (2nd, 3rd, 4th,... ect) frame
                                        # This must be a tunable parameter

        self.metrics_chunk_size = (6 * 60) / self.analyze_every_x_frame  # 6 frames per second * 60 frames in a minute / how many we actually analyze 
                                        # batch size MUST represent 1 minute of data 
                                        # for statistical accuracy. 
                                        # Max = 360 (frames in one minute of data)

        self.num_images = 0 

        self.stats_dict = {'Minutes': [],'Maxes': [], 'Indexes': [], 'Means': []}

        
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
        '''
        Analyze all the data in chunks that equate to one minute of images 
        Each statistic is calculated over 1 single chunk.
        Under the hood, dataframes are loaded in chunks, so
        when I ask for max, I get the max from the chunk, 
        not the overall. 
        '''
        num_rows = int(self.metrics_chunk_size) # num rows = 360
        print('num rows = ', num_rows)

        max_of_chunk = []
        idx_of_chunk_max = []  
        mean_of_chunk = [] 

        for df in pd.read_csv(path_to_csv, chunksize=num_rows):
            max = df['Num_detections'].max()
            max_of_chunk.append(max)
            max_idx = df['Num_detections'].idxmax()
            idx_of_chunk_max.append(max_idx)
            mean = df['Num_detections'].mean()
            mean_of_chunk.append(mean)

        print('Number of chunks =', len(max_of_chunk))

        for i in range(len(max_of_chunk)):
            self.stats_dict['Minutes'].append(i)
            self.stats_dict['Maxes'].append(max_of_chunk[i])
            self.stats_dict['Indexes'].append(idx_of_chunk_max[i])
            self.stats_dict['Means'].append(mean_of_chunk[i])
        print(self.stats_dict)


    def visualize_stats (self, stats=None):
        if (stats is None):
            # Default to self.stats_dict
            stats = self.stats_dict
        # translate python dictionary to pandas dataframe
        data = pd.DataFrame.from_dict(stats)
        # st.bar_chart(data, y = ['Maxes','Means']) 

        # Plotly 
        # Define custom colors for the bars
        custom_colors = {
            'Maxes': 'orange',     # Set 'Maxes' to orange
            'Means': 'rgb(0, 119, 190)'  # Set 'Means' to ocean blue (using RGB format)
        }

        figure = px.bar(
            data,
            x = 'Minutes',
            y = ['Maxes', 'Means'],
            labels={
            'Minute': 'Time (Minutes)',  # X-axis label
            'value': 'Detections',  # Custom Y-axis label
            'variable': 'Metrics'  # Legend label for grouping 'Maxes' and 'Means'
            },
            title='FNF Detections',
            barmode='overlay',
            color_discrete_map=custom_colors,
            opacity=1
        )

        st.plotly_chart(figure)

def main():

    fishDetector = FishDetector() 
    #fishDetector.process_images(fishDetector.images_dir)    # TODO: maybe move 
    csv_path = os.path.join(fishDetector.cwd, 'long_detections.csv')
    fishDetector.get_metrics(csv_path)
    fishDetector.visualize_stats(fishDetector.stats_dict)



if __name__ == '__main__': 
    main()

# Run inference on every image 
# Store the image name and number of detections in csv file 
# Parse the csv, find max number of detections during each interval (maybe separate program)
# Save best images with inference data on them
# Display to user 
