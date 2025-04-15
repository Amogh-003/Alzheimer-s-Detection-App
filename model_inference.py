import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Define the same model architecture used during training
class AlzheimerModel(nn.Module):
    def __init__(self):
        super(AlzheimerModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)  # Adjusted for two poolings from 128x128 input
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 128 → 64
        x = self.pool(F.relu(self.conv2(x)))  # 64 → 32
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define class labels
class_names = ['Mild Dementia', 'Moderate Dementia', 'Non Demented', 'Very Mild Dementia']

# Prediction function
def predict_image(image_path):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize for model compatibility
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Load model
    model = AlzheimerModel()
    model.load_state_dict(torch.load(
        r'C:\Users\amogh\OneDrive\Desktop\alzhemeizer detection\alzheimers_app\model\alzheimer_model.pth',
        map_location=torch.device('cpu')
    ))
    model.eval()

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]
