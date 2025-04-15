if __name__=="main_":
    print("Starting evaluation...")  # Add this as the first line
import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from main import AlzheimerCNN, test_loader, device, labels_map  # make sure main.py has these defined

# Load the model
model = AlzheimerCNN().to(device)
model.load_state_dict(torch.load("alzheimer_model.pth"))
model.eval()

# Collect predictions
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Print results
print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=labels_map.keys()))
