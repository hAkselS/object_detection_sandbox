import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
from tempfile import TemporaryDirectory
from datasets import load_dataset
from transformers import DetrImageProcessor, DetrForObjectDetection  # Import the model

# Use CUDA if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device = {device}')

# Load the locally saved model and move it to the GPU
custom_detr_model = torch.load("detr_test_code/DETR.pt", weights_only=False)
custom_detr_model = custom_detr_model.to(device)  # Move the model to GPU
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101", revision="no_timm")

# Freeze the existing model parameters
for param in custom_detr_model.parameters():
    param.requires_grad = False

# Modify the classification head of the model
num_ftrs = custom_detr_model.class_labels_classifier.in_features
# print("Number of input features to the last classification layer:", num_ftrs)
num_outputs = 21  # 20 classes in the Francesco fish market dataset 

# Create new layer and move to GPU 
custom_detr_model.class_labels_classifier = nn.Linear(num_ftrs, num_outputs)
custom_detr_model.class_labels_classifier = custom_detr_model.class_labels_classifier.to(device)  # Move new head to GPU

# Load the dataset
ds = load_dataset("Francesco/fish-market-ggjso")

# Get dataset sizes for training and validation
dataset_sizes = {x: len(ds[x]) for x in ['train', 'validation']} 

# Helper function for training the model
def train_model(model, criterion_cat, criterion_bb, optimizer, scheduler, num_epochs):
    since = time.time()
    '''
    Number of queries refers to the amount of predictions the model will make on a given image. 
    In this case, number of queries is 100, hence, for every image the model will output 100 category
    and bounding box predictions. Most of these will end up being 'background'. This effects us in the case of 
    categories because we need to pass in 100 categories as our ground truth with only 1 (because this 
    dataset only has one fish per image) actual category [17 for example]. 
    '''

    # Create a temporary directory to save training checkpoints
    
    # torch.save(model, 'DETR_custom_out.pt' )

    idx = 0 
    for epoch in range(num_epochs):

        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data
            for sample in ds[phase]:
                idx = idx + 1 
                if ((idx % 100) == 0):
                    print("procesed images in epoch = ", idx)

                if (idx > 200):
                    '''
                    While messsing with variables, it's not worth training on the whole dataset. 
                    '''
                    break

                # Process image
                image = processor(images=sample['image'], return_tensors="pt").to(device)
                input_tensor = image['pixel_values']

                # Find number of objects in a givin sample
                num_objects = len(sample['objects']['category'])
                ## Categories ## 
                # Back fill target categories with all 0 (background)
                background_class = num_outputs -1  ## TODO: this may be the ousrce of my issues (in testing)
                target_categories = torch.full((100,), background_class, dtype=torch.long).to(device)
                # Manually insert the real categories into the 0s tensor
                for i in range(num_objects):
                    target_categories[i] = sample['objects']['category'][i] # Assign scalar values (the category number to category[i]
                # DEBUG: print target categories
                # print('target categories = ', target_categories)
                
                
                # Create the right shape for the target bboxes
                num_queries = 100  # We got this number via the output logits (I think)
                target_bboxes = torch.zeros((1, num_queries, 4)).to(device)
                # Insert the bounding boxes into their corresponding position in the 0s tensor
                for i in range(num_objects):
                    target_bboxes[0,i] = torch.tensor(sample['objects']['bbox'][i]).to(device)
                # print('target bboxes size = ', target_bboxes.size())

                # Zero the parameter gradients
                optimizer.zero_grad()

                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(input_tensor)  # Model output includes logits and bounding boxes
                    logits = outputs.logits  # [batch_size, num_queries, num_classes]
                    bboxes = outputs.pred_boxes  # [batch_size, num_queries, 4]
                    # DEBUG: print statements 
                    #print('size of logits = ', logits.size())
                    #print('size of bboxes = ', bboxes.size())

                    # Reshape logits and target_categorys to match for CrossEntropyLoss
                    # Assuming logits is [batch_size, num_queries, num_classes]
                    # Flatten logits and targets for loss calculation
                    logits = logits.view(-1, logits.shape[-1])  # [batch_size * num_queries, num_classes]
                    target_categories = target_categories.view(-1)  # [batch_size * num_queries]
                    # DEBUG: print statements
                    #print('new size of logits = ', logits.size())
                    #print('size of target catagories = ', target_categories.size())

                    # Calculate classification loss
                    loss_cat = criterion_cat(logits, target_categories)

                    # Calculate bounding box regression loss
                    loss_bb = criterion_bb(bboxes, target_bboxes)

                    # Combine losses with equal weighting
                    loss = loss_cat + loss_bb

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * input_tensor.size(0)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f}')

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    torch.save(model.cpu(), 'detr_test_code/DETR_custom_out.pt' )
    return model

# Define the criterion 
criterion_category = nn.CrossEntropyLoss()
criterion_bbox = nn.SmoothL1Loss()

# Define optimizer
optimizer = optim.SGD(custom_detr_model.class_labels_classifier.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Train the model
model = train_model(custom_detr_model, criterion_category, criterion_bbox, optimizer, exp_lr_scheduler, num_epochs=2)

