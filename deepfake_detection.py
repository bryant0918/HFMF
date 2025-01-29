import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np
from ultralytics import YOLO 
import mediapipe as mp
from Module1 import Module1_feature_extraction as fe
from Ensembling import ensemble
from torchvision import transforms

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Device: ", str(device))


# Load models
resnet_model = fe.load_saved_resnet_model('checkpoints/resnet50_model_wildrf.pth')
vit_model = fe.load_vit_model('checkpoints/WildRF_vit_state_dict_.pth')

# Set up Mediapipe for facial landmarks extraction
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
# Load YOLOv8 model
yolo_model = YOLO("checkpoints/yolov8n.pt").to(device)  # Ensure YOLO runs on GPU if available)  # Choose the YOLOv8 model variant based on resources



frames_dir = "/Users/bryantmcarthur/Addavox/mnt/image_data/gop_frames"
csv_path = "/Users/bryantmcarthur/Addavox/mnt/image_data/gop_frames/features.csv"

# Data transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Custom dataset class
class CustomDatasetInfer(Dataset):
    def __init__(self, root_dir, transform=None, limit=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_files = []
        # self.labels = []

        for idx, file_name in enumerate(os.listdir(root_dir)):
            if limit and idx >= limit:
                break  # Limit the number of files loaded
            if file_name.endswith(('.jpg', '.png', '.jpeg')):  # Ensure image files
                self.image_files.append(os.path.join(root_dir, file_name))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]  # This is a string path
        image = Image.open(img_path).convert("RGB")  # Ensure image is 3 channels
        # label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, img_path  # Ensure that the path returned is a string (not a tensor)



# Extract Features
train_dataset = CustomDatasetInfer(root_dir=frames_dir, transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

print("Processing Train Dataset:")
fe.save_features_to_csv(resnet_model, vit_model, train_loader, save_path=csv_path)

# Initialize models
module1 = ensemble.DeepfakeClassifier().to(device)
module1.initialize_sobel_linear(input_shape=(3, 299, 299))
module1.load_state_dict(torch.load("checkpoints/best_model_module1_WildRF.pth", map_location=device))

module2 = ensemble.DNN(input_dim=768, hidden_dim_1=128, hidden_dim_2=256, output_dim=2).to(device)
module2.load_state_dict(torch.load("checkpoints/best_model_module2_WildRF.pth", map_location=device))

# Freeze weights of module1 and module2
for param in module1.parameters():
    param.requires_grad = False

for param in module2.parameters():
    param.requires_grad = False


# Transforms for images
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Prepare the test dataset and DataLoader
test_dataset = ensemble.prepare_ensemble_data(
    csv_path, frames_dir, transform, module1, module2, labels=False
)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# Ensemble model
class EnsembleModel(nn.Module):
    def __init__(self, module1_dim, module2_dim, output_dim):
        super(EnsembleModel, self).__init__()
        self.fc1 = nn.Linear(module1_dim + module2_dim, output_dim)  # Combine module1 and module2 outputs

    def forward(self, x1_logits, x2_logits):
        # Apply softmax to logits for probabilities
        x1_probs = torch.softmax(x1_logits, dim=1)
        x2_probs = torch.softmax(x2_logits, dim=1)
        # Concatenate probabilities
        combined_probs = torch.cat((x1_probs, x2_probs), dim=1)
        # Pass through the fully connected layer
        output = self.fc1(combined_probs)
        return output


# Initialize the EnsembleModel
module1_output_dim = 2  # Output size of module1
module2_output_dim = 2  # Output size of module2
ensemble_model = EnsembleModel(module1_output_dim, module2_output_dim, output_dim=2).to(device)


# Load the best saved model
ensemble_model.load_state_dict(torch.load("checkpoints/best_ensemble_model_WildRF.pth", map_location=device))
ensemble_model.eval()  # Set the model to evaluation mode

preds_sum = 0
with torch.no_grad():
    for module1_inputs, module2_inputs in tqdm(test_loader, desc=f"Running Ensemble Inference"):
        module1_inputs, module2_inputs = module1_inputs.to(device), module2_inputs.to(device)

    # Forward pass through ensemble model
    outputs = ensemble_model(module1_inputs, module2_inputs)

    # Get predictions
    _, preds = torch.max(outputs, 1)
    preds_sum += preds.sum().item()

    print("Predictions: ", preds.cpu().numpy())


print("Odds that video is a deepfake: ", preds_sum / len(test_dataset))

