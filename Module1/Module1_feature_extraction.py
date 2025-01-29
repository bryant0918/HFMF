from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm 
import ssl
import os
import csv

# image_path = Path(__file__).resolve().parent.parent / "WildRF"
# train_dir = image_path / "train"
# val_dir = image_path / "val"
# test_dir = image_path / "test"

# # Print paths to verify
# print(f"Train directory: {train_dir}")
# print(f"Validation directory: {val_dir}")
# print(f"Test directory: {test_dir}")


# # Custom dataset class
# class CustomDataset(Dataset):
#     def __init__(self, root_dir, transform=None, limit=None):
#         self.root_dir = root_dir
#         self.transform = transform

#         self.image_files = []
#         self.labels = []
#         for label_folder in tqdm(['0_real', '1_fake'], desc="Loading dataset"):
#             full_path = os.path.join(root_dir, label_folder)
#             for idx, file_name in enumerate(os.listdir(full_path)):
#                 if limit and idx >= limit:
#                     break  # Limit the number of files loaded
#                 if file_name.endswith(('.jpg', '.png', '.jpeg')):  # Ensure image files
#                     self.image_files.append(os.path.join(full_path, file_name))
#                     if 'real' in label_folder:
#                         self.labels.append(0)  # Label 0 for real images
#                     else:
#                         self.labels.append(1)  # Label 1 for fake images

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_path = self.image_files[idx]
#         image = Image.open(img_path).convert("RGB")  # Ensure image is 3 channels
#         label = self.labels[idx]

#         if self.transform:
#             image = self.transform(image)

#         return image, label

# # Data transformations
# data_transforms = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# # Load datasets
# train_dataset = CustomDataset(root_dir=train_dir, transform=data_transforms)
# val_dataset = CustomDataset(root_dir=val_dir, transform=data_transforms)
# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
# val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

ssl._create_default_https_context = ssl.create_default_context
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Device: ", str(device))

# Custom dataset class
class CustomDatasetNew(Dataset):
    def __init__(self, root_dir, transform=None, limit=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_files = []
        self.labels = []
        for label_folder in tqdm(['0_real', '1_fake'], desc="Loading dataset"):
            full_path = os.path.join(root_dir, label_folder)
            for idx, file_name in enumerate(os.listdir(full_path)):
                if limit and idx >= limit:
                    break  # Limit the number of files loaded
                if file_name.endswith(('.jpg', '.png', '.jpeg')):  # Ensure image files
                    self.image_files.append(os.path.join(full_path, file_name))
                    if 'real' in label_folder:
                        self.labels.append(0)  # Label 0 for real images
                    else:
                        self.labels.append(1)  # Label 1 for fake images

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]  # This is a string path
        image = Image.open(img_path).convert("RGB")  # Ensure image is 3 channels
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, img_path  # Ensure that the path returned is a string (not a tensor)


# Load ResNet model and capture features
def load_saved_resnet_model(model_path):
    model = torchvision.models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Binary classification (real/fake)

    model.load_state_dict(torch.load(model_path, map_location=device))  # Load the saved model
    model = model.to(device)

    # Hook functions to capture low, mid, and high-level features
    model.layer1[0].register_forward_hook(lambda m, i, o: hook_fn(m, i, o, low_level_features))
    model.layer3[0].register_forward_hook(lambda m, i, o: hook_fn(m, i, o, mid_level_features))
    model.layer4[0].register_forward_hook(lambda m, i, o: hook_fn(m, i, o, high_level_features))

    model.eval()
    return model

# Hook functions to capture ResNet features
low_level_features, mid_level_features, high_level_features = [], [], []

def hook_fn(module, input, output, storage_list):
    storage_list.append(output.clone().detach())

# Define linear layers to convert ResNet features to 768 dimensions
# Define linear layers to convert ResNet features to 768 dimensions
low_to_768 = nn.Linear(256, 768).to(device)   # For low-level features
mid_to_768 = nn.Linear(1024, 768).to(device)  # For mid-level features
high_to_768 = nn.Linear(2048, 768).to(device) # For high-level features

def extract_resnet_features(model, image):
    low_level_features.clear()
    mid_level_features.clear()
    high_level_features.clear()

    with torch.no_grad():
        image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device
        model(image)

    # Pool ResNet features and map to 768 dimensions
    low_pooled = F.adaptive_avg_pool2d(low_level_features[-1].to(device), (1, 1)).squeeze()
    mid_pooled = F.adaptive_avg_pool2d(mid_level_features[-1].to(device), (1, 1)).squeeze()
    high_pooled = F.adaptive_avg_pool2d(high_level_features[-1].to(device), (1, 1)).squeeze()

    low_768 = low_to_768(low_pooled)   # Shape [1, 768]
    mid_768 = mid_to_768(mid_pooled)   # Shape [1, 768]
    high_768 = high_to_768(high_pooled) # Shape [1, 768]

    return low_768, mid_768, high_768

# Function to preprocess the image using ViT's transforms
def pipeline_preprocessor():
    vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    return vit_weights.transforms()

# Function to extract ViT embeddings
def get_vit_embedding(vit_model, image_path):
    preprocessing = pipeline_preprocessor()  # Preprocessing from ViT
    img = Image.open(image_path).convert("RGB")  # Ensure we load image by path (string)
    img = preprocessing(img).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        feats = vit_model._process_input(img)
        batch_class_token = vit_model.class_token.expand(img.shape[0], -1, -1)
        feats = torch.cat([batch_class_token, feats], dim=1)
        feats = vit_model.encoder(feats)
        vit_hidden = feats[:, 0]  # CLS token
    return vit_hidden

# Load ViT model
def load_vit_model(pretrained_weights_path):
    vit_model = torchvision.models.vit_b_16(pretrained=False).to(device)
    pretrained_vit_weights = torch.load(pretrained_weights_path, map_location=device)
    vit_model.load_state_dict(pretrained_vit_weights, strict=False)
    vit_model.eval()  # Set to evaluation mode
    return vit_model

# Add a sequence dimension (if missing) before applying attention
def ensure_correct_shape(tensor):
    if len(tensor.shape) == 2:  # If shape is [batch_size, embedding_dim]
        tensor = tensor.unsqueeze(1)  # Add a sequence dimension: [batch_size, 1, embedding_dim]
    elif len(tensor.shape) == 1:  # If shape is [embedding_dim]
        tensor = tensor.unsqueeze(0).unsqueeze(1)  # Add batch and sequence dimensions: [1, 1, embedding_dim]
    return tensor

# Scaled dot product attention function
def scaled_dot_product_attention(Q, K, V):
    # Ensure Q, K, and V have the correct shapes
    Q = ensure_correct_shape(Q)  # Should be [batch_size, 1, embedding_dim]
    K = ensure_correct_shape(K)  # Should be [batch_size, 1, embedding_dim]
    V = ensure_correct_shape(V)  # Should be [batch_size, 1, embedding_dim]

#     print(f"Q shape after unsqueeze: {Q.shape}, K shape after unsqueeze: {K.shape}, V shape after unsqueeze: {V.shape}")  # Debugging
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32).to(Q.device))
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output

# Save features for each dataset (train/val/test) as CSV
def save_features_to_csv(model, vit_model, data_loader, save_path, labels=True):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write the CSV header
        writer.writerow(["image_name", "features", "label"])

        for images, img_paths in tqdm(data_loader, desc="Extracting features"):
            for i in range(len(images)):
                image = images[i].to(device)  # Move image to the correct device
                img_path = img_paths[i]  # Image path

                # Ensure img_path is a string
                if isinstance(img_path, torch.Tensor):
                    img_path = img_path.item() if img_path.dim() == 0 else str(img_path)

                # Extract ResNet features
                try:
                    low_768, mid_768, high_768 = extract_resnet_features(model, image)
                except Exception as e:
                    print(f"Error extracting ResNet features for {img_path}: {e}")
                    continue

                # Extract ViT features
                try:
                    vit_hidden = get_vit_embedding(vit_model, img_path)  # img_path should be a string
                except Exception as e:
                    print(f"Error extracting ViT features for {img_path}: {e}")
                    continue

                # Apply attention between ResNet and ViT features
                try:
                    output_1 = scaled_dot_product_attention(vit_hidden, low_768, low_768)
                    output_2 = scaled_dot_product_attention(output_1, mid_768, mid_768)
                    final_output = scaled_dot_product_attention(output_2, high_768, high_768)
                except Exception as e:
                    print(f"Error applying attention for {img_path}: {e}")
                    continue

                # Convert features to a flattened list
                features = final_output.detach().cpu().numpy().flatten().tolist()

                if labels:
                    # Extract label from the image path
                    label = 0 if "real" in img_path else 1

                    # Write the row to the CSV
                    writer.writerow([os.path.basename(img_path), features, label])
                else:
                    # Write the row to the CSV without label
                    writer.writerow([os.path.basename(img_path), features, None])

    print(f"Features saved to {save_path}")

# # Load models
# resnet_model = load_saved_resnet_model('checkpoints/resnet50_model_wildrf.pth')
# vit_model = load_vit_model('checkpoints/WildRF_vit_state_dict_.pth')



# train_dataset = CustomDatasetNew(root_dir=train_dir, transform=data_transforms)
# val_dataset = CustomDatasetNew(root_dir=val_dir, transform=data_transforms)

# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
# # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# print("Processing Train Dataset:")
# save_features_to_csv(resnet_model, vit_model, train_loader, save_path="WildRF/train/features.csv")

# print("Processing Validation Dataset:")
# save_features_to_csv(resnet_model, vit_model, val_loader, save_path="WildRF/val/features.csv")

# print("Processing Test Dataset:")
# Now handling test directory structure with subfolders (twitter, facebook, reddit)
# for test_subdir in ['twitter', 'facebook', 'reddit']:
#     test_dataset = CustomDatasetNew(root_dir=os.path.join(test_dir, test_subdir), transform=data_transforms)
#     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
#     print(f"Processing Test Dataset: {test_subdir}")
#     save_features_to_csv(resnet_model, vit_model, test_loader, save_path=f"WildRF/test/{test_subdir}/features.csv")

import shutil
import os

def folder_to_zip(folder_path, zip_name):
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        print("The folder does not exist.")
        return

    # Create a zip file from the folder
    shutil.make_archive(zip_name, 'zip', folder_path)
    print(f"Folder '{folder_path}' has been successfully zipped to '{zip_name}.zip'.")

# Example usage
folder_to_zip('/kaggle/working/features', 'features')
