#!/usr/bin/env python3
"""
Anomaly detection inference using trained autoencoder
Detects Class 2 points based on high reconstruction error
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import os
import json

# Import the autoencoder model
from autoencoder_train import Autoencoder3D, voxelize_for_autoencoder

class AnomalyDetectionDataset(Dataset):
    """Dataset for anomaly detection inference (with original labels for evaluation)"""
    def __init__(self, data_file, label_file):
        print(f"Loading inference data from {data_file}")
        print(f"Loading inference labels from {label_file}")
        
        with h5py.File(data_file, 'r') as f:
            self.data = f['data'][:]
        
        with h5py.File(label_file, 'r') as f:
            self.labels = f['labels'][:]
        
        print(f"Loaded {len(self.data)} inference samples")
        
        # Analyze dataset for Class 2 distribution
        total_points = 0
        class2_points = 0
        samples_with_class2 = 0
        
        for labels in self.labels:
            total_points += len(labels)
            class2_count = np.sum(labels == 2)
            class2_points += class2_count
            if class2_count > 0:
                samples_with_class2 += 1
        
        print(f"📊 Inference dataset statistics:")
        print(f"   Total samples: {len(self.data)}")
        print(f"   Total points: {total_points:,}")
        print(f"   Class 2 points: {class2_points:,} ({100*class2_points/total_points:.4f}%)")
        print(f"   Samples with Class 2: {samples_with_class2}/{len(self.data)} ({100*samples_with_class2/len(self.data):.1f}%)")

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

def anomaly_collate_fn(batch):
    """Custom collate function for anomaly detection"""
    point_clouds, labels = zip(*batch)
    
    processed_batch = []
    original_labels = []
    
    for pc, label in zip(point_clouds, labels):
        # Voxelize point cloud
        voxel_features = voxelize_for_autoencoder(pc, grid_size=64, voxel_size=8)
        
        # Convert to tensor and rearrange dimensions for conv3d (C, H, W, D)
        voxel_tensor = torch.FloatTensor(voxel_features).permute(3, 0, 1, 2)
        
        processed_batch.append(voxel_tensor)
        original_labels.append(label)
    
    # Stack tensors
    voxel_batch = torch.stack(processed_batch)
    
    return voxel_batch, original_labels

def load_trained_autoencoder(model_path, device):
    """Load the trained autoencoder model"""
    print(f"Loading autoencoder from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get latent dimension
    latent_dim = checkpoint.get('latent_dim', 256)
    
    # Initialize model
    model = Autoencoder3D(latent_dim=latent_dim)
    
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
    
    # Print model info
    if 'best_val_loss' in checkpoint:
        print(f"📊 Best validation loss: {checkpoint['best_val_loss']:.6f}")
    if 'epoch' in checkpoint:
        print(f"📅 Trained for: {checkpoint['epoch']} epochs")
    
    print("✅ Autoencoder loaded successfully")
    return model

def compute_reconstruction_errors(model, dataloader, device):
    """Compute reconstruction errors for anomaly detection"""
    print("🔍 Computing reconstruction errors for anomaly detection...")
    
    sample_errors = []  # Per-sample reconstruction errors
    sample_labels = []  # Whether each sample contains Class 2
    point_errors = []   # Per-voxel reconstruction errors
    point_labels = []   # Per-voxel ground truth labels
    
    with torch.no_grad():
        for batch_idx, (voxels, original_labels) in enumerate(dataloader):
            voxels = voxels.to(device)
            
            # Forward pass through autoencoder
            reconstructed, latent = model(voxels)
            
            # Compute reconstruction error (MSE per sample)
            mse_per_sample = torch.mean((voxels - reconstructed) ** 2, dim=(1, 2, 3, 4))
            
            # Store sample-level errors and labels
            for i, (sample_error, original_label) in enumerate(zip(mse_per_sample, original_labels)):
                sample_errors.append(sample_error.cpu().item())
                # Sample is positive if it contains any Class 2 points
                has_class2 = np.any(original_label == 2)
                sample_labels.append(int(has_class2))
                
                # Also store per-voxel information for detailed analysis
                voxel_errors = ((voxels[i] - reconstructed[i]) ** 2).mean(dim=0).cpu().numpy()
                point_errors.extend(voxel_errors.flatten())
                
                # Create voxel-level labels (simplified mapping from original labels)
                voxel_labels = np.zeros_like(voxel_errors)
                # This is approximate since we can't perfectly map back to voxels
                # For evaluation, we'll focus on sample-level detection
                point_labels.extend(voxel_labels.flatten())
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx + 1}/{len(dataloader)}")
    
    return np.array(sample_errors), np.array(sample_labels), np.array(point_errors), np.array(point_labels)

def evaluate_anomaly_detection(sample_errors, sample_labels, save_dir="anomaly_results"):
    """Evaluate anomaly detection performance"""
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n🎯 === AUTOENCODER ANOMALY DETECTION RESULTS ===")
    
    # Basic statistics
    normal_errors = sample_errors[sample_labels == 0]
    anomaly_errors = sample_errors[sample_labels == 1]
    
    print(f"📊 Reconstruction Error Statistics:")
    print(f"   Normal samples: {len(normal_errors)} (mean error: {np.mean(normal_errors):.6f} ± {np.std(normal_errors):.6f})")
    print(f"   Anomaly samples: {len(anomaly_errors)} (mean error: {np.mean(anomaly_errors):.6f} ± {np.std(anomaly_errors):.6f})")
    print(f"   Error ratio (anomaly/normal): {np.mean(anomaly_errors)/np.mean(normal_errors):.2f}x")
    
    # ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(sample_labels, sample_errors)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall Curve (better for imbalanced data)
    precision, recall, pr_thresholds = precision_recall_curve(sample_labels, sample_errors)
    pr_auc = auc(recall, precision)
    
    print(f"🎯 Performance Metrics:")
    print(f"   ROC AUC: {roc_auc:.4f}")
    print(f"   PR AUC: {pr_auc:.4f}")
    
    # Find optimal thresholds
    # Threshold that maximizes Youden's J statistic (TPR - FPR)
    youden_j = tpr - fpr
    optimal_roc_idx = np.argmax(youden_j)
    optimal_roc_threshold = roc_thresholds[optimal_roc_idx]
    
    # Threshold that maximizes F1 score
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_f1_idx = np.argmax(f1_scores[:-1])  # Exclude last point
    optimal_f1_threshold = pr_thresholds[optimal_f1_idx]
    
    print(f"🔍 Optimal Thresholds:")
    print(f"   ROC-based threshold: {optimal_roc_threshold:.6f} (TPR: {tpr[optimal_roc_idx]:.3f}, FPR: {fpr[optimal_roc_idx]:.3f})")
    print(f"   F1-based threshold: {optimal_f1_threshold:.6f} (F1: {f1_scores[optimal_f1_idx]:.3f})")
    
    # Evaluate at different thresholds
    thresholds_to_test = [
        ("90th percentile", np.percentile(sample_errors, 90)),
        ("95th percentile", np.percentile(sample_errors, 95)),
        ("99th percentile", np.percentile(sample_errors, 99)),
        ("ROC optimal", optimal_roc_threshold),
        ("F1 optimal", optimal_f1_threshold)
    ]
    
    print(f"\n📋 Performance at Different Thresholds:")
    for name, threshold in thresholds_to_test:
        predictions = (sample_errors > threshold).astype(int)
        tp = np.sum((predictions == 1) & (sample_labels == 1))
        fp = np.sum((predictions == 1) & (sample_labels == 0))
        tn = np.sum((predictions == 0) & (sample_labels == 0))
        fn = np.sum((predictions == 0) & (sample_labels == 1))
        
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
        
        print(f"   {name:15s}: Threshold={threshold:.6f}, P={precision_val:.3f}, R={recall_val:.3f}, F1={f1:.3f}")
    
    # Create comprehensive plots
    plt.figure(figsize=(20, 12))
    
    # 1. Reconstruction error distributions
    plt.subplot(2, 4, 1)
    plt.hist(normal_errors, bins=50, alpha=0.7, label=f'Normal (n={len(normal_errors)})', color='blue', density=True)
    plt.hist(anomaly_errors, bins=50, alpha=0.7, label=f'Anomaly (n={len(anomaly_errors)})', color='red', density=True)
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.title('Reconstruction Error Distribution')
    plt.legend()
    plt.yscale('log')
    
    # 2. ROC Curve
    plt.subplot(2, 4, 2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    # 3. Precision-Recall Curve
    plt.subplot(2, 4, 3)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    
    # 4. Box plot of errors
    plt.subplot(2, 4, 4)
    box_data = [normal_errors, anomaly_errors]
    box_labels = ['Normal', 'Anomaly']
    plt.boxplot(box_data, labels=box_labels)
    plt.ylabel('Reconstruction Error')
    plt.title('Error Distribution Box Plot')
    plt.yscale('log')
    
    # 5. Error vs Sample Index
    plt.subplot(2, 4, 5)
    colors = ['blue' if label == 0 else 'red' for label in sample_labels]
    plt.scatter(range(len(sample_errors)), sample_errors, c=colors, alpha=0.6, s=10)
    plt.xlabel('Sample Index')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Error by Sample')
    plt.yscale('log')
    
    # 6. Threshold analysis
    plt.subplot(2, 4, 6)
    threshold_range = np.linspace(np.min(sample_errors), np.max(sample_errors), 100)
    precision_curve = []
    recall_curve = []
    f1_curve = []
    
    for thresh in threshold_range:
        pred = (sample_errors > thresh).astype(int)
        tp = np.sum((pred == 1) & (sample_labels == 1))
        fp = np.sum((pred == 1) & (sample_labels == 0))
        fn = np.sum((pred == 0) & (sample_labels == 1))
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        precision_curve.append(p)
        recall_curve.append(r)
        f1_curve.append(f1)
    
    plt.plot(threshold_range, precision_curve, label='Precision', color='blue')
    plt.plot(threshold_range, recall_curve, label='Recall', color='green')
    plt.plot(threshold_range, f1_curve, label='F1 Score', color='red')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs Threshold')
    plt.legend()
    
    # 7. Confusion matrix at optimal F1 threshold
    plt.subplot(2, 4, 7)
    pred_optimal = (sample_errors > optimal_f1_threshold).astype(int)
    cm = confusion_matrix(sample_labels, pred_optimal)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'], 
                yticklabels=['Normal', 'Anomaly'])
    plt.title(f'Confusion Matrix\n(Threshold: {optimal_f1_threshold:.6f})')
    
    # 8. Error correlation
    plt.subplot(2, 4, 8)
    if len(anomaly_errors) > 1 and len(normal_errors) > 1:
        plt.scatter(normal_errors[:-1], normal_errors[1:], alpha=0.5, label='Normal', color='blue', s=10)
        if len(anomaly_errors) > 1:
            plt.scatter(anomaly_errors[:-1], anomaly_errors[1:], alpha=0.5, label='Anomaly', color='red', s=10)
    plt.xlabel('Error(i)')
    plt.ylabel('Error(i+1)')
    plt.title('Error Autocorrelation')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'anomaly_detection_analysis.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'anomaly_detection_analysis.pdf'), bbox_inches='tight')
    
    # Save detailed results
    results = {
        'total_samples': len(sample_errors),
        'normal_samples': len(normal_errors),
        'anomaly_samples': len(anomaly_errors),
        'normal_error_mean': float(np.mean(normal_errors)),
        'normal_error_std': float(np.std(normal_errors)),
        'anomaly_error_mean': float(np.mean(anomaly_errors)),
        'anomaly_error_std': float(np.std(anomaly_errors)),
        'error_separation_ratio': float(np.mean(anomaly_errors)/np.mean(normal_errors)),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'optimal_roc_threshold': float(optimal_roc_threshold),
        'optimal_f1_threshold': float(optimal_f1_threshold),
        'optimal_f1_score': float(f1_scores[optimal_f1_idx]),
        'thresholds': {name: {'threshold': float(thresh), 'precision': float(precision_val), 'recall': float(recall_val), 'f1': float(f1)}
                      for (name, thresh), precision_val, recall_val, f1 in 
                      [(thresholds_to_test[i], *[tp/(tp+fp) if tp+fp>0 else 0, tp/(tp+fn) if tp+fn>0 else 0][0:2], 
                        2*(tp/(tp+fp) if tp+fp>0 else 0)*(tp/(tp+fn) if tp+fn>0 else 0)/((tp/(tp+fp) if tp+fp>0 else 0)+(tp/(tp+fn) if tp+fn>0 else 0)) if (tp/(tp+fp) if tp+fp>0 else 0)+(tp/(tp+fn) if tp+fn>0 else 0)>0 else 0)
                       for i, (tp, fp, tn, fn) in enumerate([
                           tuple(np.sum([(sample_errors > thresh) & (sample_labels == 1), 
                                        (sample_errors > thresh) & (sample_labels == 0),
                                        (sample_errors <= thresh) & (sample_labels == 0),
                                        (sample_errors <= thresh) & (sample_labels == 1)], axis=1))
                           for _, thresh in thresholds_to_test])]}
    }
    
    with open(os.path.join(save_dir, 'anomaly_detection_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Analysis plots saved to {save_dir}/anomaly_detection_analysis.*")
    print(f"📋 Results saved to {save_dir}/anomaly_detection_results.json")
    
    return optimal_f1_threshold, results

def main():
    # Configuration
    inference_data_file = "/nevis/houston/home/sc5303/INSS_2025/INSS2025/data/example_xyze_100.h5"
    inference_label_file = "/nevis/houston/home/sc5303/INSS_2025/INSS2025/data/example_label_100.h5"
    
    # Try to use the best autoencoder model
    model_candidates = [
        "/nevis/houston/home/sc5303/INSS_2025/point-cloud-cnn-segmentation/best_autoencoder_model.pth",
        "/nevis/houston/home/sc5303/INSS_2025/point-cloud-cnn-segmentation/final_autoencoder_model.pth"
    ]
    
    model_path = None
    for candidate in model_candidates:
        if os.path.exists(candidate):
            model_path = candidate
            print(f"🎯 Using autoencoder model: {candidate}")
            break
    
    if model_path is None:
        print("❌ Error: No autoencoder model file found!")
        print("Please run autoencoder_train.py first to train the autoencoder.")
        return
    
    batch_size = 4
    
    # Check if files exist
    for file_path in [inference_data_file, inference_label_file]:
        if not os.path.exists(file_path):
            print(f"❌ Error: File not found - {file_path}")
            return
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    # Create inference dataset and dataloader
    inference_dataset = AnomalyDetectionDataset(inference_data_file, inference_label_file)
    inference_dataloader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False,
                                    collate_fn=anomaly_collate_fn, num_workers=2)
    
    # Load autoencoder model
    model = load_trained_autoencoder(model_path, device)
    
    # Compute reconstruction errors
    sample_errors, sample_labels, point_errors, point_labels = compute_reconstruction_errors(
        model, inference_dataloader, device)
    
    # Evaluate anomaly detection performance
    optimal_threshold, results = evaluate_anomaly_detection(sample_errors, sample_labels)
    
    print(f"\n🎉 Autoencoder anomaly detection completed successfully!")
    print(f"📁 Results saved in 'anomaly_results/' directory")
    print(f"🔍 Optimal threshold for Class 2 detection: {optimal_threshold:.6f}")
    print(f"📊 ROC AUC: {results['roc_auc']:.4f}, PR AUC: {results['pr_auc']:.4f}")

if __name__ == "__main__":
    print("🔧 Autoencoder-based Class 2 Anomaly Detection")
    main()
