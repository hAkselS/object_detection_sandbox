import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Determine is cuda (gpu) is available
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Create the neural network
class NeuralNetwork(nn.Module): # Inherit from nn.Module (true for most models)
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential( # Define a 5 layer neutal network model
            nn.Linear(28*28, 512),  # Linear layers are what we thing of as neurons 
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    def forward(self, x):   # Forward is called when we pass the model data (no need to call directly)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

# Create random dataset 
X = torch.rand(1, 28, 28, device=device)
logits = model(X) # Pass data into the model 
pred_probab = nn.Softmax(dim=1)(logits) 
y_pred = pred_probab.argmax(1) # Get the prediction (most activated neuron)
print(f"Predicted class: {y_pred}")

