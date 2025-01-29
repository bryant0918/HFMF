import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
import cv2
from ultralytics import YOLO  # For YOLOv8 face detection
from tqdm import tqdm
import pretrainedmodels  # For Xception model
import numpy as np
import mediapipe as mp  # For facial landmark extraction
from torch.cuda.amp import autocast, GradScaler
import ssl
from pathlib import Path
import matplotlib.pyplot as plt


# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Device: ", str(device))

# # Paths
# image_path = Path(__file__).resolve().parent.parent / "WildRF"
# train_folder = image_path / "train"
# val_folder = image_path / "val"
# test_dir = image_path / "test"

# train_csv = os.path.join(train_folder, "features.csv")
# val_csv = os.path.join(val_folder, "features.csv")

# Transforms for images
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class DeepfakeClassifier(torch.nn.Module):
    def __init__(self):
        super(DeepfakeClassifier, self).__init__()
        self.xception = xception_model  # Outputs 128 features
        self.sobel_cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.sobel_linear = None  # Will initialize dynamically
        self.fc_landmarks = nn.Linear(936, 128).to(device)  # 936 = flattened landmarks
        self.fc_yolo = nn.Linear(80, 64).to(device)  # Adjust YOLO features to 64
        self.fc1 = None  # To be initialized dynamically
        self.fc2 = nn.Linear(128, 2).to(device)

    def initialize_sobel_linear(self, input_shape):
        with torch.no_grad():
            # Initialize Sobel Linear
            sample_input = torch.zeros(1, *input_shape).to(device)
            output = self.sobel_cnn(sample_input)
            flattened_size = output.view(-1).size(0)
            print(f"Flattened size: {flattened_size}")  # Debugging line
            self.sobel_linear = nn.Linear(flattened_size, 128).to(device)

            # Calculate the total feature size for fc1
            total_feature_size = 128 + 128 + 128 + 64  # xception + sobel + landmarks + YOLO
            self.fc1 = nn.Linear(total_feature_size, 128).to(device)

    def forward(self, image, sobel_image, yolo_features, face_landmarks):
        # Process features
        yolo_features = yolo_features.float()  # Fix for dtype mismatch
        image_features = self.xception(image)  # Output: [batch_size, 128]
        sobel_features = self.sobel_cnn(sobel_image)  # Output: [batch_size, C, H, W]
        print(f"Sobel features shape before flattening: {sobel_features.shape}")  # Debugging line
        sobel_features = self.sobel_linear(sobel_features.view(sobel_features.size(0), -1))
        print(f"Sobel features shape after flattening: {sobel_features.shape}")  # Debugging line
        yolo_features = torch.relu(self.fc_yolo(yolo_features))
        landmark_features = torch.relu(self.fc_landmarks(face_landmarks))

        # Combine features
        combined = torch.cat((image_features, sobel_features, yolo_features, landmark_features), dim=1)

        # Fully connected layers
        x = torch.relu(self.fc1(combined))
        x = self.fc2(x)
        return x
    
    
# Module 2 definition (DNN)
class DNN(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim, dropout_prob=0.2):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.fc3 = nn.Linear(hidden_dim_2, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Function to process images
def process_images(folder_path, transform, limit=None, labels=True):
    images = []
    filenames = []
    y = []

    if labels:
        label_dirs = ["0_real", "1_fake"]
    else:
        label_dirs = [""]

    for label_dir in label_dirs:
        if labels:
            label_path = os.path.join(folder_path, label_dir)
            label = 0 if label_dir == "0_real" else 1
        else:
            label_path = folder_path
            label = None

        for i, fname in enumerate(tqdm(os.listdir(label_path), desc=f"Processing {label_dir}")):
            if limit and len(images) >= limit:
                break
            img_path = os.path.join(label_path, fname)
            image = cv2.imread(img_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = transform(Image.fromarray(image))
                images.append(image)
                filenames.append(fname)
                y.append(label)

    # Convert to tensors
    X = torch.stack(images)
    if labels:
        y = torch.tensor(y, dtype=torch.long)
    return X, filenames, y


# Process CSV features for Module 2
def process_csv(path, limit=None, labels=True):
    df = pd.read_csv(path)
    features = df['features'].apply(lambda x: list(map(float, x.strip('[]').split(','))))
    filenames = df['image_name'].tolist()  # Ensure CSV has 'image_name' column
    X = torch.tensor(features.tolist(), dtype=torch.float32)
    if labels:
        y = torch.tensor(df['label'].values, dtype=torch.long)
    else:
        y = [None]*len(filenames)

    if limit:
        X = X[:limit]
        filenames = filenames[:limit]
        y = y[:limit]

    return X, filenames, y


# Set up Mediapipe for facial landmarks extraction
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Load Xception model
ssl._create_default_https_context = ssl._create_unverified_context
xception_model = pretrainedmodels.__dict__['xception'](pretrained='imagenet').to(device)
xception_model.last_linear = torch.nn.Linear(xception_model.last_linear.in_features, 128).to(device)  
# Load YOLOv8 model
yolo_model = YOLO("checkpoints/yolov8n.pt").to(device)  # Ensure YOLO runs on GPU if available)  # Choose the YOLOv8 model variant based on resources
# Define COCO classes we are interested in (people, vehicles, animals, household items, etc.)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

def generate_sobel_edges(image, transform):
    """
    Generates Sobel edges for a given image.
    """
    gray_image = cv2.cvtColor(image.permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    sobel_combined = cv2.convertScaleAbs(sobel_combined)
    sobel_combined = cv2.merge([sobel_combined, sobel_combined, sobel_combined])
    return transform(Image.fromarray(sobel_combined))

def extract_yolo_features_and_landmarks(image):
    """
    Extracts YOLO object detection features and face landmarks from the given image.
    """
    results = yolo_model(image.permute(1, 2, 0).cpu().numpy())
    detected_objects = []
    landmarks = np.zeros((936,), dtype=np.float32)

    for result in results[0].boxes:
        class_id = int(result.cls[0])
        class_name = yolo_model.names[class_id]

        if class_name == "person":
            crop = image.permute(1, 2, 0).cpu().numpy()[
                int(result.xyxy[0][1]):int(result.xyxy[0][3]),
                int(result.xyxy[0][0]):int(result.xyxy[0][2]),
            ]
            crop = (crop * 255).astype(np.uint8) if crop.max() <= 1.0 else crop.astype(np.uint8)
            face_result = face_mesh.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            if face_result.multi_face_landmarks:
                landmarks = np.array(
                    [[p.x, p.y] for p in face_result.multi_face_landmarks[0].landmark]
                ).flatten()

        detected_objects.append(class_id)

    yolo_features = torch.tensor([1 if i in detected_objects else 0 for i in range(len(COCO_CLASSES))])
    return yolo_features, torch.tensor(landmarks)

def prepare_ensemble_data(csv_path, folder_path, transform, module1, module2, batch_size=4, labels=True):
    """
    Prepares the ensemble dataset by processing data in batches.
    """
    images, img_filenames, labels1 = process_images(folder_path, transform, labels=labels)
    csv_features, csv_filenames, labels2 = process_csv(csv_path, labels=labels)

    img_base_names = [os.path.splitext(fname)[0] for fname in img_filenames]
    csv_base_names = [os.path.splitext(fname)[0] for fname in csv_filenames]

    mapping = {f"{base}_{labels1[i].item() if labels1[i] is not None else 'None'}": (i, None) for i, base in enumerate(img_base_names)}
    for i, base in enumerate(csv_base_names):
        key = f"{base}_{labels2[i].item() if labels2[i] is not None else 'None'}"
        if key in mapping:
            mapping[key] = (mapping[key][0], i)

    img_indices, csv_indices = [], []
    for key, (img_idx, csv_idx) in mapping.items():
        if csv_idx is not None:
            img_indices.append(img_idx)
            csv_indices.append(csv_idx)
    
    images = images[img_indices]
    if isinstance(labels1, list):
        labels1 = [labels1[i] for i in img_indices]
    else:
        labels1 = labels1[img_indices]
    csv_features = csv_features[csv_indices]

    combined_outputs_module1, combined_outputs_module2 = [], []
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size].to(device)
        sobel_images, yolo_features, face_landmarks = [], [], []

        for img in batch_images:
            sobel_images.append(generate_sobel_edges(img, transform))
            yolo, landmarks = extract_yolo_features_and_landmarks(img)
            yolo_features.append(yolo)
            face_landmarks.append(landmarks)

        sobel_images = torch.stack(sobel_images).to(device)
        yolo_features = torch.stack(yolo_features).to(device)
        face_landmarks = torch.stack(face_landmarks).to(device)
        print(batch_images.shape, sobel_images.shape, yolo_features.shape, face_landmarks.shape)

        # Get outputs from module1
        module1.eval()
        with torch.no_grad():
            module1_output = module1(batch_images, sobel_images, yolo_features, face_landmarks)
            combined_outputs_module1.append(module1_output.cpu())

        # Get outputs from module2
        module2.eval()
        with torch.no_grad():
            module2_output = module2(csv_features[i:i + batch_size].to(device))
            combined_outputs_module2.append(module2_output.cpu())

    module1_outputs = torch.cat(combined_outputs_module1, dim=0)
    module2_outputs = torch.cat(combined_outputs_module2, dim=0)

    # Ensure labels match the outputs
    min_size = min(len(module1_outputs), len(labels1), len(module2_outputs))
    module1_outputs = module1_outputs[:min_size]
    module2_outputs = module2_outputs[:min_size]
    labels1 = labels1[:min_size]

    if labels:
        return TensorDataset(module1_outputs, module2_outputs, labels1)
    else:
        return TensorDataset(module1_outputs, module2_outputs)

# # Initialize models
# module1 = DeepfakeClassifier().to(device)
# module1.initialize_sobel_linear(input_shape=(3, 299, 299))
# module1.load_state_dict(torch.load("checkpoints/best_model_module1_WildRF.pth", map_location=device))

# module2 = DNN(input_dim=768, hidden_dim_1=128, hidden_dim_2=256, output_dim=2).to(device)
# module2.load_state_dict(torch.load("checkpoints/best_model_module2_WildRF.pth", map_location=device))

# # Freeze weights of module1 and module2
# for param in module1.parameters():
#     param.requires_grad = False

# for param in module2.parameters():
#     param.requires_grad = False
    
# train_dataset = prepare_ensemble_data(train_csv, train_folder, transform, module1, module2)
# val_dataset = prepare_ensemble_data(val_csv, val_folder, transform, module1, module2)


# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

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
    



# # Initialize the EnsembleModel
# module1_output_dim = 2  # Output size of module1
# module2_output_dim = 2  # Output size of module2
# ensemble_model = EnsembleModel(module1_output_dim, module2_output_dim, output_dim=2).to(device)


# # Define loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(ensemble_model.parameters(), lr=0.001, weight_decay=1e-5)

# # Training and Validation loop
# num_epochs = 100
# patience = 3  # Number of epochs to wait for improvement
# best_val_loss = float('inf')
# epochs_no_improve = 0  # Counter for epochs with no improvement

# train_losses, val_losses = [], []
# train_accuracies, val_accuracies = [], []

# for epoch in range(num_epochs):
#     # Training
#     ensemble_model.train()
#     running_train_loss = 0.0
#     correct_train = 0
#     total_train = 0

#     for module1_inputs, module2_inputs, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
#         module1_inputs, module2_inputs, y = module1_inputs.to(device), module2_inputs.to(device), y.to(device)

#         optimizer.zero_grad()

#         # Forward pass through ensemble model
#         outputs = ensemble_model(module1_inputs, module2_inputs)

#         # Compute loss
#         loss = criterion(outputs, y)

#         # Backward pass and optimization
#         loss.backward()
#         optimizer.step()

#         # Update training metrics
#         running_train_loss += loss.item()
#         _, preds = torch.max(outputs, 1)
#         correct_train += (preds == y).sum().item()
#         total_train += y.size(0)

#     train_loss = running_train_loss / len(train_loader)
#     train_accuracy = correct_train / total_train * 100
#     train_losses.append(train_loss)
#     train_accuracies.append(train_accuracy)

#     # Validation
#     ensemble_model.eval()
#     running_val_loss = 0.0
#     correct_val = 0
#     total_val = 0

#     with torch.no_grad():
#         for module1_inputs, module2_inputs, y in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
#             module1_inputs, module2_inputs, y = module1_inputs.to(device), module2_inputs.to(device), y.to(device)

#             # Forward pass through ensemble model
#             outputs = ensemble_model(module1_inputs, module2_inputs)

#             # Compute loss
#             loss = criterion(outputs, y)

#             # Update validation metrics
#             running_val_loss += loss.item()
#             _, preds = torch.max(outputs, 1)
#             correct_val += (preds == y).sum().item()
#             total_val += y.size(0)

#     val_loss = running_val_loss / len(val_loader)
#     val_accuracy = correct_val / total_val * 100
#     val_losses.append(val_loss)
#     val_accuracies.append(val_accuracy)

#     print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
#           f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

#     # Check for improvement in validation loss
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         epochs_no_improve = 0  # Reset the counter
#         torch.save(ensemble_model.state_dict(), "checkpoints/best_ensemble_model_WildRF.pth")
#         print(f"Saved best model at epoch {epoch + 1}")
#     else:
#         epochs_no_improve += 1
#         print(f"No improvement for {epochs_no_improve} epoch(s)")

#     # Early stopping
#     if epochs_no_improve >= patience:
#         print("Early stopping triggered. Stopping training.")
#         break

# # Plot Loss and Accuracy Curves
# plt.figure(figsize=(12, 6))

# # Accuracy subplot
# plt.subplot(1, 2, 1)
# plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Training Accuracy", linestyle='--')
# plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label="Validation Accuracy", linestyle='--')
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.title("Training and Validation Accuracy over Epochs")
# plt.legend()
# plt.grid(True)

# # Loss subplot
# plt.subplot(1, 2, 2)
# plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss", linestyle='--')
# plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", linestyle='--')
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training and Validation Loss over Epochs")
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()
# plt.savefig("training_validation_curves_Ensemble_WildRF.png")
# print("Saved training and validation curves as 'training_validation_curves.png'")



# ### TEST SET ###

# # Load the best saved model
# ensemble_model.load_state_dict(torch.load("checkpoints/best_ensemble_model_WildRF.pth", map_location=device))
# ensemble_model.eval()  # Set the model to evaluation mode

# # Paths to test folders and CSVs
# test_platforms = {
#     "Facebook": {
#         "csv": os.path.join(test_dir, "facebook/features.csv"),
#         "folder": os.path.join(test_dir, "facebook")
#     },
#     "Reddit": {
#         "csv": os.path.join(test_dir, "reddit/features.csv"),
#         "folder": os.path.join(test_dir, "reddit")
#     },
#     "Twitter": {
#         "csv": os.path.join(test_dir, "twitter/features.csv"),
#         "folder": os.path.join(test_dir, "twitter")
#     }
# }

# # Iterate over each test platform
# for platform, paths in test_platforms.items():
#     print(f"\nTesting on {platform} dataset:")

#     # Prepare the test dataset and DataLoader
#     test_dataset = prepare_ensemble_data(
#         paths["csv"], paths["folder"], transform, module1, module2
#     )
#     test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

#     # Initialize metrics
#     correct_test = 0
#     total_test = 0
#     test_predictions = []
#     test_ground_truth = []

#     with torch.no_grad():
#         for module1_inputs, module2_inputs, y in tqdm(test_loader, desc=f"Testing {platform}"):
#             module1_inputs, module2_inputs, y = module1_inputs.to(device), module2_inputs.to(device), y.to(device)

#             # Forward pass through ensemble model
#             outputs = ensemble_model(module1_inputs, module2_inputs)

#             # Get predictions
#             _, preds = torch.max(outputs, 1)

#             # Update metrics
#             correct_test += (preds == y).sum().item()
#             total_test += y.size(0)

#             # Store predictions and ground truth for additional metrics
#             test_predictions.extend(preds.cpu().numpy())
#             test_ground_truth.extend(y.cpu().numpy())

#     # Compute test accuracy
#     test_accuracy = correct_test / total_test * 100
#     print(f"{platform} Test Accuracy: {test_accuracy:.2f}%")

#     # Additional Metrics (Optional)
#     from sklearn.metrics import classification_report

#     print(f"\nClassification Report for {platform}:")
#     print(classification_report(test_ground_truth, test_predictions, target_names=["Class 0", "Class 1"]))

#     # Save predictions to a CSV file (Optional)
#     import pandas as pd

#     predictions_df = pd.DataFrame({
#         "Ground Truth": test_ground_truth,
#         "Predictions": test_predictions
#     })
#     predictions_csv_path = f"{test_dir}/{platform}_test_predictions_WildRF.csv"
#     predictions_df.to_csv(predictions_csv_path, index=False)
#     print(f"Test predictions for {platform} saved to '{predictions_csv_path}'")