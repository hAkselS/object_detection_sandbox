import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os 

cwd = os.getcwd()
# Path to dataset 
model_path = os.path.join(cwd, 'test_code_2/rcnn_training/custom_resnet_fasterrcnn_2.pt')
# Load the model weights
model = torch.load(model_path)
model.eval()

# Define the label dictionary
labels_dict = {
    2: 'fish',
    1: 'bait_arm', 
    0: 'background'
} # TODO: try switching fish and background 

# Preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.ToTensor()  # Convert to tensor and normalize
    ])
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor, img

# Postprocess and visualize the results
def visualize_predictions(image, outputs, threshold=0.8):
    # Get bounding boxes, labels, and scores
    boxes = outputs[0]['boxes'].detach().cpu().numpy()
    labels = outputs[0]['labels'].detach().cpu().numpy()
    scores = outputs[0]['scores'].detach().cpu().numpy()

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Draw bounding boxes
    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:  # Only show predictions above the threshold
            xmin, ymin, xmax, ymax = box
            width, height = xmax - xmin, ymax - ymin
            rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            # Add text without background
            plt.text(xmin, ymin, f'{labels_dict[label]}: {score:.2f}', color='white', fontsize=8, bbox=dict(facecolor='none', edgecolor='none', pad=0))

    plt.show()

# Load and preprocess the image
image_tensor, image = preprocess_image('test_code_2/rcnn_training/LeFish.png')

# Run the model on the image
with torch.no_grad():
    predictions = model(image_tensor)

# Visualize the predictions
visualize_predictions(image, predictions)

