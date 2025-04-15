import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from main import AlzheimerCNN, train_loader, device, labels_map  # make sure main.py is set up correctly

# Load the trained model
model = AlzheimerCNN().to(device)
model.load_state_dict(torch.load("alzheimer_model.pth"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Print evaluation metrics
print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=labels_map.keys()))
