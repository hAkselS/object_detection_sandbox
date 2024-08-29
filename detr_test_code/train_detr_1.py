# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# import time
# import os
# from tempfile import TemporaryDirectory
# from datasets import load_dataset
# from transformers import DetrImageProcessor, DetrForObjectDetection  # Import the model

# # Use CUDA if available
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f'device = {device}')

# # Load the locally saved model and move it to the GPU
# custom_detr_model = torch.load("detr_test_code/DETR_custom.pt", weights_only=False)
# custom_detr_model = custom_detr_model.to(device)  # Move the model to GPU
# processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101", revision="no_timm")

# # Freeze the existing model parameters
# for param in custom_detr_model.parameters():
#     param.requires_grad = False

# # Modify the classification head of the model
# num_ftrs = custom_detr_model.class_labels_classifier.in_features
# print("Number of input features to the last classification layer:", num_ftrs)
# num_outputs = 21  # 20 classes in the Francesco fish market dataset 

# # Create new layer and move to GPU 
# custom_detr_model.class_labels_classifier = nn.Linear(num_ftrs, num_outputs)
# custom_detr_model.class_labels_classifier = custom_detr_model.class_labels_classifier.to(device)  # Move new head to GPU

# # Load the dataset
# ds = load_dataset("Francesco/fish-market-ggjso")

# # Get dataset sizes for training and validation
# dataset_sizes = {x: len(ds[x]) for x in ['train', 'validation']} 

# # Helper function for training the model
# def train_model(model, criterion_cat, criterion_bb, optimizer, scheduler, num_epochs=7):
#     since = time.time()

#     # Create a temporary directory to save training checkpoints
#     with TemporaryDirectory() as tempdir:
#         best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
#         torch.save(model.state_dict(), best_model_params_path)
#         best_acc = 0.0

#         for epoch in range(num_epochs):
#             print(f'Epoch {epoch}/{num_epochs - 1}')
#             print('-' * 10)

#             # Each epoch has a training and validation phase
#             for phase in ['train', 'validation']:
#                 if phase == 'train':
#                     model.train()  # Set model to training mode
#                 else:
#                     model.eval()  # Set model to evaluate mode

#                 running_loss = 0.0

#                 # Iterate over data
#                 for sample in ds[phase]:
#                     # Process image
#                     image = processor(images=sample['image'], return_tensors="pt").to(device)
#                     input_tensor = image['pixel_values'] #.squeeze(0)  # Remove batch dimension if present

#                     # Assuming 'objects' contain class labels, convert them to tensors
#                     target_categorys = torch.tensor([sample['objects']['category']]).to(device) # Tensor != tensor 
#                     # BUG: current theory, need to give category AAAND bbox to train
#                     target_bboxes = torch.tensor(sample['objects']['bbox']).to(device)

#                     # Zero the parameter gradients
#                     optimizer.zero_grad()

#                     # Forward pass
#                     with torch.set_grad_enabled(phase == 'train'):
#                         outputs = model(input_tensor)  # Add batch dimension if needed
#                         logits = outputs.logits  # Extract the logits from the DetrObjectDetectionOutput
                        
#                         # Assuming logits are in shape [batch_size, num_queries, num_classes]
#                         # For CrossEntropyLoss, we need [batch_size * num_queries, num_classes]
#                         # And targets should be [batch_size * num_queries]
#                         loss_cat = criterion_cat(logits.view(-1, logits.shape[-1]), target_categorys.view(-1)) # TODO: may need to add anothe target here
#                         loss_bb = criterion_bb(logits.view(-1, logits.shape[-1]), target_bboxes.view(-1))
#                         loss = loss_cat + (1 * loss_bb) # Equally weighted losses 

#                         # Backward pass + optimize only if in training phase
#                         if phase == 'train':
#                             loss.backward()
#                             optimizer.step()

#                     # Statistics
#                     running_loss += loss.item() * input_tensor.size(0)

#                 if phase == 'train':
#                     scheduler.step()

#                 epoch_loss = running_loss / dataset_sizes[phase]
#                 print(f'{phase} Loss: {epoch_loss:.4f}')

#         time_elapsed = time.time() - since
#         print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
#         print(f'Best val Acc: {best_acc:4f}')

#     return model

# # Define the criterion 
# # Cross entropy for classification
# criterion_category = nn.CrossEntropyLoss()
# # L1 smooth sample loss for bbox
# criterion_bbox = nn.SmoothL1Loss()

# # Define optimizer
# optimizer = optim.SGD(custom_detr_model.class_labels_classifier.parameters(), lr=0.001, momentum=0.9)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# # Train the model
# model = train_model(custom_detr_model, criterion_category, criterion_bbox, optimizer, exp_lr_scheduler, num_epochs=7)
