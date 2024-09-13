'''
Spec: This program is designed to properly format a dataset
for use as training and validation for faster_rcnn. Note, that 
pytorch data classes have certain requirements outlined in the 
reference material. 

Reference: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
           https://github.com/harshatejas/pytorch_custom_object_detection/blob/main/train.py
'''

import os # Open files 
import json # Read Dataset files 
import matplotlib.pyplot as plt 
import torch
import torchvision.io as io
import torchvision.transforms as T 
# import torchvision.utils as utils


cwd = os.getcwd()
# Path to dataset 
data_dir_path = os.path.join(cwd, 'test_code_2/rcnn_training/fish_data')
images_dir_path = os.path.join(data_dir_path, 'fish_images')
train_json_file_name = 'train.json'
resize_dims = (600,800)
resize_transform = T.Resize(resize_dims)

labels_dict = {
    0: 'fish',
    1: 'bait_arm',
    2: 'background'
}
labels_to_index = {v: k for k, v in labels_dict.items()}


def transform_filename(filename):
    # Split at the first hyphen to remove the prefix
    parts = filename.split('-', 1)
    # Take the part after the hyphen and replace underscores with spaces
    # Only replace underscores before ".png"
    transformed_name = parts[1].replace('_', ' ')
    return transformed_name

def read_json(path, file):
    '''
    Open & return the json file associated with the dataset
    '''
    json_file_path = os.path.join(path,file)
    with open(json_file_path,'r') as file:
        json_annotation_file = json.load(file)
        return json_annotation_file
    
def get_image(json_file, idx):
    '''
    Return an image via index 
    '''
    long_img_name = json_file[idx]['file_upload']
    image_name = transform_filename(long_img_name)
    image_path = os.path.join(images_dir_path, image_name)
    #image_tensor = io.read_image('/Users/akselsloan/Desktop/test_dataset/Fith/Screen Shot 2024-09-06 at 12.13.06 PM.png')
    image_tensor = io.read_image(image_path)
    image_tensor = resize_transform(image_tensor) # TODO: I think this just removes extra pixels, come back to this!!!
    # plt.imshow(image_tensor.permute(1,2,0))
    # plt.show()
    return image_tensor

def get_labels(json_file, idx):
    '''
    Return a list of all class labels in an annotated image
    '''
    labels = []
    for annotation in json_file[idx]['annotations'][0]['result']:
        label_str = annotation['value']['rectanglelabels'][0]
        if label_str in labels_to_index:
            labels.append(labels_to_index[label_str])
        else:
            raise ValueError(f"Label '{label_str}' not found in labels_dict")
    return labels

def get_bboxes(json_file, idx): # TODO: bounding boxes are coming out WRONG (may be a precentage thing instead of pixel number)
    # Bounding boxes are a percentage (can be negative some how)
    # The x y refers to the top left corner of the box. (0,0) is the top left corner of the image
    # Adjust as necessary. 
    '''
    Returns a list of lists with all the bboxes in them. Format is [x0,y0, x1,y1,] 
    '''
    bboxes = []
    for i in range(len(json_file[idx]['annotations'][0]['result'])):
        # Need to pass x_min, x_max... to faster_rcnn 
        norm_x = resize_dims[1] * (1 / 100) # dimensions bbox dimensions are given as percentage values
        norm_y = resize_dims[0] * (1 / 100)
        curr_box = [] 

        x_min = json_file[idx]['annotations'][0]['result'][i]['value']['x']
        x_width = json_file[idx]['annotations'][0]['result'][i]['value']['width']
        y_min = json_file[idx]['annotations'][0]['result'][i]['value']['y']
        y_height = json_file[idx]['annotations'][0]['result'][i]['value']['height']

        x_min = x_min * norm_x
        x_max = (x_width * norm_x) + x_min
        y_min = y_min * norm_y 
        y_max = (y_height * norm_y) + y_min 

        curr_box = [x_min, y_min, x_max, y_max]
        bboxes.append(curr_box)
    return bboxes


    

class MyFishDataset(torch.utils.data.Dataset):

    def __init__(self):
        '''
        Fill in data_dir_path and train_json_file_name as needed
        '''
        self.json_file = read_json(data_dir_path, train_json_file_name)
        
    def __len__(self):
        length = len(self.json_file)
        return length

    def __getitem__(self, idx):
        '''
        get item via index
        strip off any weirdness with a function 
        find the image 
        apply necessary transforms
        
        return image and associated labels (json entry)
        '''
        # print(self.json_file[idx])
        image = get_image(self.json_file, idx)
        bbox = get_bboxes(self.json_file, idx)
        label = get_labels(self.json_file, idx)
        return image, label, bbox # Not certain on formatting here
    

def main():
    print ('hello world')
    #json_file = read_json(data_dir_path, train_json_file_name)
    #get_image(json_file, 5)
    data = MyFishDataset()

    data.__getitem__(0)
    # print("data.__getitem__(0) yields:", data.__getitem__(4))
    # print(" length =", data.__len__())
    # for i in range(12):
    #     image = data.__getitem__(i)[0]
    #     print("image size =", image.size())

    # print(resize_dims[1])
if __name__ == '__main__':
    main()