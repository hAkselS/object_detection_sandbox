'''
Spec: This program is designed to show me whats in the Francesco Fish Market dataset.

Notes: There are 20 data classes. 0 is fish 
                '0': fish
                '1': aair
                '2': boal
                '3': chapila
                '4': deshi puti
                '5': foli
                '6': ilish
                '7': kal baush
                '8': katla
                '9': koi
                '10': magur
                '11': mrigel
                '12': pabda
                '13': pangas
                '14': puti
                '15': rui
                '16': shol
                '17': taki
                '18': tara baim
                '19': telapiya

'''
from datasets import load_dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision.transforms as transforms

# Load the dataset
ds = load_dataset("rafaelpadilla/coco2017")
# View information on the dataset
print(ds)

# Isolate the training portion of the dataset
train_ds = ds['train']
# Verify that we have split the training section 
print(train_ds)


# Fill in manually with fish names
label_map = {0: 'Class0', 1: 'Class1', 2: 'Class2', 3: 'Class3', 4: 'Class4', 5: 'Class5', 6: 'Class6'} 

# Function to display images with bounding boxes and labels
def show_image_with_bbox(image, bboxes, categories):
    # Create a plot
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Draw bounding boxes
    for bbox, category in zip(bboxes, categories):
        x, y, width, height = bbox
        
        # Create a rectangle patch
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # Add the class label
        plt.text(x, y, category, color='white', fontsize=12, backgroundcolor='red')

    plt.show()

# Display a small subset of images with bounding boxes and labels
# for i in range(1):  # Display the first 5 images
#     image = ds['train'][i]['image']
#     bboxes = ds['train'][i]['objects']['bbox']
#     categories = ds['train'][i]['objects']['category']
    
#     show_image_with_bbox(image, bboxes, categories)