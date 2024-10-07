'''
Spec: a single script that runs inference as similar
to MOUSS_FNF would, while still running on a PC. 

Reference: 
 - Inference options:
 https://docs.ultralytics.com/modes/predict/#inference-arguments
 -
'''

from ultralytics import YOLO
# import cv2
import os
import time 
import pandas as pd # Handle CSV file 
import numpy as np # TODO: remove, used to find table shape
import streamlit as st # display metrics on webpage 
import plotly.express as px # for metrics 
from PIL import Image, ImageDraw



class FishDetector:
    def __init__ (self):
        self.cwd = os.getcwd() # Fish_no_fish project root

        ## Specify path to model
        self.model_path = os.path.join(self.cwd, 'models/yolov8n_fish_trained_lgds.pt')

        self.model = YOLO(self.model_path)

        ## Specify path to images ##
        self.images_dir = os.path.join(self.cwd, 'test_code_2/rcnn_training/fish_data/fish_images')
        # self.images_dir = os.path.join(self.cwd, '/Volumes/KINGSTON/20240831_172402')


        self.wait_for_new_images_time = 1 # Seconds, before the program starts analyzing metrics
        self.need_header = True

        self.analyze_every_x_frame = 1  # Analyze every (2nd, 3rd, 4th,... ect) frame
                                        # This must be a tunable parameter

        self.metrics_chunk_size = (6 * 60) / self.analyze_every_x_frame  # 6 frames per second * 60 frames in a minute / how many we actually analyze 
                                        # batch size MUST represent 1 minute of data 
                                        # for statistical accuracy. 
                                        # Max = 360 (frames in one minute of data)

        self.num_images = 0 

        self.stats_dict = {'Minutes': [],'Maxes': [], 'Indexes': [], 'Names': [], 'Means': []}

        
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
            
            for image in images[::self.analyze_every_x_frame]:
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

    def get_metrics(self, path_to_csv, write_stats_csv=False):
        '''
        Analyze all the data in chunks that equate to one minute of images.
        Each statistic is calculated over 1 single chunk.
        Under the hood, dataframes are loaded in chunks, so
        when I ask for max, I get the max from the chunk, 
        not the overall. 
        '''
        num_rows = int(self.metrics_chunk_size) # num rows = 360
        print('num rows = ', num_rows)

        max_of_chunk = []
        idx_of_chunk_max = []  
        name_of_image = [] 
        mean_of_chunk = [] 

        for df in pd.read_csv(path_to_csv, chunksize=num_rows):
            max = df['Num_detections'].max()
            max_of_chunk.append(max)
            
            max_idx = df['Num_detections'].idxmax()
            idx_of_chunk_max.append(max_idx)
            
            name_string = df['Image'][max_idx]
            name_of_image.append(name_string)
            
            mean = df['Num_detections'].mean()
            mean_of_chunk.append(mean)

        print('Number of chunks =', len(max_of_chunk))

        for i in range(len(max_of_chunk)):
            self.stats_dict['Minutes'].append(i)
            self.stats_dict['Maxes'].append(max_of_chunk[i])
            self.stats_dict['Indexes'].append(idx_of_chunk_max[i])
            self.stats_dict['Names'].append(name_of_image[i])
            self.stats_dict['Means'].append(mean_of_chunk[i])
        ## Print for troubleshooting 
        print(self.stats_dict)

        # Write to CSV if desired 
        if (write_stats_csv == True): 
            stats_df = pd.DataFrame.from_dict(self.stats_dict)
            stats_df.to_csv('stats.csv', index=False, header=True)



    def visualize_stats(self, stats=None):
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

    def inference_best_images(self, path_to_csv=None):
        '''
        Run inference on the images that were selected as
        'best' by get_metrics. Images should be saved to
        a directory called 'labeled_data' and should have 
        bounding boxes (but not lables) stored AS PART OF
        THE IMAGE. 
        '''
        # Create a place to save images (make if it doesn't exit yet)
        image_output_dir = os.path.join(self.cwd, 'labeled_data')
        os.makedirs(image_output_dir, exist_ok=True)
        
        # Return a list of image names, either from local memory of a csv
        if (path_to_csv is not None): # CSV case
            # TODO: if using csv method, fill 'images_to_inference" list with string names of all revelvant images
            pass 
        else:
            images_to_inference = [] 
            images_to_inference.append(self.stats_dict['Names'])
        
        idx = 0
        for name in self.stats_dict['Names']:
            path_to_best_image = os.path.join(self.images_dir, name)
            second_inference = self.model(path_to_best_image)
            
            image = Image.open(path_to_best_image)
            draw = ImageDraw.Draw(image)
            for result in second_inference:
                boxes = result.boxes

                for box in boxes: 
                    box_coords = box.xyxy[0]
                    # print(box_coords)
                    # TODO: Scaling is wrong here
                    x, y, w, h = box_coords
                    top_left = (x, y)
                    # bottom_right = (x+w, y+h)
                    width_height = (w, h)
                    draw.rectangle([top_left, width_height], outline='red', width=3)
            # Save each annotated image
            # image_name = os.path.join('prediction_', str(idx)) # name the image 'prediction_2' for ex. 
            # image_name = os.path.join(image_name, '.png')
            image_name = 'prediction_' + str(idx) + '.png'
            image_save_path = os.path.join(image_output_dir, image_name)
            image.save(image_save_path)
            idx = idx + 1


def main():

    fishDetector = FishDetector() 
    #fishDetector.process_images(fishDetector.images_dir)    
    csv_path = os.path.join(fishDetector.cwd, 'detections.csv')
    fishDetector.get_metrics(csv_path, True) # True means write stats to a file
    fishDetector.inference_best_images()
    #fishDetector.visualize_stats(fishDetector.stats_dict)
    

if __name__ == '__main__': 
    main()

# Run inference on every image 
# Store the image name and number of detections in csv file 
# Parse the csv, find max number of detections during each interval (maybe separate program)
# Save best images with inference data on them
# Display to user 

# TODO: run inference on best images, save them into a new directory WITH bboxes included in the jpg
