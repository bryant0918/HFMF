# HFMF: Hierarchical Fusion Meets Multi-Stream Models for Deepfake Detection
## # Accepted at WACV 2025 Workshop - AI4MFDD, Tucson, Arizona (US)
### Codebase is only adapted to WildRF version.

## Overview  
Our proposed model is a state-of-the-art deepfake detection architecture that integrates hierarchical feature fusion and multi-stream models. By combining Vision Transformers, ResNet, and specialized modules like Yolov8 and XceptionNet, it excels in detecting high-quality and realistic deepfakes generated by modern Generative AI techniques. The model emphasizes both spatial and semantic understanding while ensuring explainability through Grad-CAM visualizations.

## Key Features  
- **Hierarchical Feature Fusion**: Combines Vision Transformer (ViT) and ResNet50 outputs for enriched representations.  
- **Multi-Stream Architecture**: Incorporates facial features, context, and edge-aware details via Yolov8, Sobel filters, and XceptionNet.  
- **Explainability**: Employs Grad-CAM to produce class-discriminative heatmaps for output decisions.  
- **Calibration**: Uses Platt Scaling to refine output probabilities, reducing model overconfidence.  
- **Ensemble Decision Making**: Aggregates multi-stream predictions for robust and accurate results.  

## Architecture  
The model architecture comprises two main modules:  
1. **Module One**:  
   - Hierarchical fusion of ViT and ResNet50 for feature extraction.  
   - Calibration using Platt Scaling to ensure well-calibrated outputs.  
2. **Module Two**:  
   - Multi-stream feature extraction using Yolov8, Sobel filters, and XceptionNet.  
   - Grad-CAM for visualization and explainability.
  
    &nbsp; &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;<img src="https://github.com/user-attachments/assets/cabaa9ae-4f42-4c98-a575-d3294011ddd5" alt="Ensemble" width="500" />


## Results  

- **Calibration**: Achieved an 8% decrease in Expected Calibration Error (ECE) on the training set and a 13% decrease on the validation set.  
- **Explainability**: Grad-CAM heatmaps demonstrate accurate localization of manipulated regions.  
- **Ensemble Performance**: Outperforms baseline models in detecting deepfakes while maintaining computational efficiency.

  &nbsp; &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;<img src="https://github.com/user-attachments/assets/6d27282f-bc48-42b1-bb19-ebfd1ae14b29" alt="Comparison" width="600" />

## Data Preparation as follows:
```bash
!pip install gdown
from pathlib import PosixPath
#WildRF
!gdown --id 1A0xoL44Yg68ixd-FuIJn2VC4vdZ6M2gn -c
!unzip -q -n WildRF.zip

#CollabDiff
gdown --id 1GpGvkxQ7leXqCnfnEAsgY_DXFnJwIbO4 -c
!unzip -q -n CollabDiff.zip

```

## Comparison with SOTA on WildRF  

![Screenshot 2025-01-05 at 12 02 34 PM](https://github.com/user-attachments/assets/773ad989-28ae-4f65-9a47-84a65be1dc88)


## Ablation on WildRF  
![Screenshot 2025-01-05 at 12 02 56 PM](https://github.com/user-attachments/assets/3a7e2c53-1205-4afc-b0f2-01b6b4c5e1a0)


## Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/anantmehta33/Hierarchical-Multi-Stream-Fusion-for-Deepfake-Detection.git
   cd Hierarchical-Multi-Stream-Fusion-for-Deepfake-Detection
2. Running Instructions:

Follow the steps below to run the full pipeline and get the final output for HFMF.

## Step 1: Fetch Weights for ViT and ResNet

- Open `ViTb16_finetuned.ipynb` to get the fine-tuned weights for the ViT and ResNet models.
- Save the weights from this notebook and use them in `Module1_feature_extraction.ipynb` to obtain the final weights for Module 1.

## Step 2: Get Logits Using DNN_M1_WildRF.ipynb

- Use `DNN_M1_WildRF.ipynb` to get the logits. 
- The notebook `Module 1.ipynb` is a refined version of this process, so you may use it for further improvements.

## Step 3: Calibrate the Logits

- Once the logits are obtained, calibrate them using the `calibration.py` script to adjust the outputs for the final model.
![Screenshot 2025-01-05 at 12 02 47 PM](https://github.com/user-attachments/assets/930c4268-2ca4-4e19-a2ba-debcc5a63257)

## Step 4: Integrate with Module 2

- Use `Module2.ipynb` to integrate the calibrated logits with other models, including:
  - XceptionNet
  - Yolov8
  - Sobel filter
  - Grad Cam (Explainable AI)<br>
  <br>
![Screenshot 2025-01-05 at 12 03 15 PM](https://github.com/user-attachments/assets/f172aa36-fd26-43ca-b021-a81fddc4fbfd)


    
## Step 5: Make Ensemble
- Use `Ensemble.ipynb` to get the final ensemble of module 1 and 2.
This integration will generate the final output for the HFMF task.

## Here is the drive link for weights used and a small demo:
https://drive.google.com/drive/folders/1Ek7z7qaqwVf2aYMMRzi14-BWxSTeef7w?usp=sharing
