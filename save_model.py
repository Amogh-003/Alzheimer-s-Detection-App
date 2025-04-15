import torch
from main import AlzheimerCNN  # This works only if the model class is in main.py

# Instantiate the model and load trained weights
model = AlzheimerCNN()
model.load_state_dict(torch.load("alzheimer_model.pth"))
torch.save(model.state_dict(), "alzheimer_model.pth")

print("Model saved successfully!")
