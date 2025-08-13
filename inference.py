#!/usr/bin/env python3
"""
Inference script for point cloud semantic segmentation
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os
import json

# Import the model class from simple_train.py
import sys
sys.path.append('/nevis/houston/home/sc5303/INSS_2025/point-cloud-cnn-segmentation')
from simple_train import Simple3DCNN, voxelize_point_cloud, PointCloudDataset

class InferenceDataset(Dataset):
    def __init__(self, data_file, label_file):
        print(f"Loading inference data from {data_file}")
        print(f"Loading inference labels from {label_file}")
        
        with h5py.File(data_file, 'r') as f:
            self.data = f['data'][:]
        
        with h5py.File(label_file, 'r') as f:
            self.labels = f['labels'][:]
        
        print(f"Loaded {len(self.data)} inference samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get point cloud data and reshape to (N, 4)
        point_cloud = self.data[idx]
        labels = self.labels[idx]
        
        # Reshape to (N, 4) for (x, y, z, energy)
        if len(point_cloud) % 4 != 0:
            point_cloud = point_cloud[:len(point_cloud)//4 * 4]
        
        point_cloud = point_cloud.reshape(-1, 4)
        
        return torch.FloatTensor(point_cloud), torch.LongTensor(labels)

def inference_collate_fn(batch):
    """Custom collate function for inference"""
    point_clouds, labels = zip(*batch)
    
    processed_batch = []
    original_labels = []
    
    for pc, label in zip(point_clouds, labels):
        # Voxelize each point cloud
        voxel_features, voxel_labels = voxelize_point_cloud(pc, label, grid_size=64, voxel_size=8)
        
        # Convert to tensor and rearrange dimensions for conv3d (C, H, W, D)
        voxel_tensor = torch.FloatTensor(voxel_features).permute(3, 0, 1, 2)
        label_tensor = torch.LongTensor(voxel_labels)
        
        processed_batch.append((voxel_tensor, label_tensor))
        original_labels.append(label)
    
    # Stack tensors
    voxel_batch = torch.stack([item[0] for item in processed_batch])
    label_batch = torch.stack([item[1] for item in processed_batch])
    
    return voxel_batch, label_batch, original_labels

def load_model(model_path, device):
    """Load the trained model"""
    print(f"Loading model from {model_path}")
    
    # Initialize model
    model = Simple3DCNN(num_classes=5)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Handle DataParallel wrapper
    if list(state_dict.keys())[0].startswith('module.'):
        # Remove 'module.' prefix from keys
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    return model

def run_inference(model, dataloader, device):
    """Run inference on the dataset"""
    print("Running inference...")
    
    all_predictions = []
    all_true_labels = []
    all_original_labels = []
    
    with torch.no_grad():
        for batch_idx, (voxels, labels, original_labels) in enumerate(dataloader):
            voxels = voxels.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(voxels)
            
            # Get predictions
            predictions = torch.argmax(outputs, dim=1)
            
            # Convert to numpy and flatten
            pred_np = predictions.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            # Flatten for each sample in the batch
            for i in range(pred_np.shape[0]):
                pred_flat = pred_np[i].flatten()
                label_flat = labels_np[i].flatten()
                
                all_predictions.extend(pred_flat)
                all_true_labels.extend(label_flat)
                all_original_labels.extend(original_labels[i].numpy())
            
            print(f"Processed batch {batch_idx + 1}/{len(dataloader)}")
    
    return np.array(all_predictions), np.array(all_true_labels), np.array(all_original_labels)

def compute_metrics(predictions, true_labels, save_dir="inference_results"):
    """Compute and save metrics"""
    os.makedirs(save_dir, exist_ok=True)
    
    # All classes are semantic classes (0-4), no background filtering needed
    print("\n=== SEMANTIC SEGMENTATION METRICS (ALL CLASSES) ===")
    
    # Overall accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    # Classification report for all classes
    class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
    class_report = classification_report(true_labels, predictions, 
                                       labels=[0, 1, 2, 3, 4], 
                                       target_names=class_names)
    print("\nClassification Report:")
    print(class_report)
    
    # Save classification report
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write("Classification Report (All Classes 0-4):\n")
        f.write(class_report)
    
    # Confusion matrix for all classes
    cm = confusion_matrix(true_labels, predictions, labels=[0, 1, 2, 3, 4])
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    
    # Confusion matrix with counts
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Counts)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Normalized confusion matrix
    plt.subplot(1, 2, 2)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'confusion_matrices.pdf'), bbox_inches='tight')
    print(f"\nConfusion matrices saved to {save_dir}/confusion_matrices.png and .pdf")
    
    # Save metrics as JSON
    metrics = {
        'overall_accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_normalized': cm_norm.tolist(),
        'total_voxels': len(true_labels),
        'class_distribution_true': {f'class_{i}': int(count) for i, count in enumerate(np.bincount(true_labels, minlength=5))},
        'class_distribution_pred': {f'class_{i}': int(count) for i, count in enumerate(np.bincount(predictions, minlength=5))}
    }
    
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {save_dir}/metrics.json")
    return cm

def main():
    # Configuration
    inference_data_file = "/nevis/houston/home/sc5303/INSS_2025/INSS2025/data/example_xyze_100.h5"
    inference_label_file = "/nevis/houston/home/sc5303/INSS_2025/INSS2025/data/example_label_100.h5"
    model_path = "/nevis/houston/home/sc5303/INSS_2025/point-cloud-cnn-segmentation/final_model.pth"
    
    batch_size = 4
    
    # Check if files exist
    for file_path in [inference_data_file, inference_label_file, model_path]:
        if not os.path.exists(file_path):
            print(f"Error: File not found - {file_path}")
            return
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create inference dataset and dataloader
    inference_dataset = InferenceDataset(inference_data_file, inference_label_file)
    inference_dataloader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False,
                                    collate_fn=inference_collate_fn, num_workers=2)
    
    # Load model
    model = load_model(model_path, device)
    
    # Run inference
    predictions, true_labels, original_labels = run_inference(model, inference_dataloader, device)
    
    # Compute metrics and confusion matrix
    cm = compute_metrics(predictions, true_labels)
    
    print("\nInference completed successfully!")
    print(f"Results saved in 'inference_results/' directory")

if __name__ == "__main__":
    main()
