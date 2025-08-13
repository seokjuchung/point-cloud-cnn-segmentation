#!/usr/bin/env python3
"""
Hybrid two-stage inference for Class 2 vs Class 3 distinction:
Stage 1: Use autoencoder to identify anomalous samples (high reconstruction error)
Stage 2: Use binary classifier to distinguish Class 2 vs Class 3 in anomalous samples
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import os
import json

# Import models from training script
from hybrid_train import HybridAnomalyDataset, HybridBinaryClassifier, voxelize_hybrid
from autoencoder_train import Autoencoder3D

class HybridInferenceDataset(Dataset):
    """Dataset for hybrid inference - keeps all original data"""
    def __init__(self, data_file, label_file):
        print(f"Loading inference data from {data_file}")
        print(f"Loading inference labels from {label_file}")
        
        with h5py.File(data_file, 'r') as f:
            self.data = f['data'][:]
        
        with h5py.File(label_file, 'r') as f:
            self.labels = f['labels'][:]
        
        print(f"Loaded {len(self.data)} inference samples")
        self._analyze_dataset()

    def _analyze_dataset(self):
        """Analyze the complete inference dataset"""
        total_points = 0
        class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        samples_with_rare = 0
        
        for labels in self.labels:
            total_points += len(labels)
            for class_id in range(5):
                class_counts[class_id] += np.sum(labels == class_id)
            
            # Check if sample contains rare classes
            if np.any((labels == 2) | (labels == 3)):
                samples_with_rare += 1
        
        print(f"📊 Complete inference dataset:")
        print(f"   Total samples: {len(self.data)}")
        print(f"   Total points: {total_points:,}")
        for class_id in range(5):
            pct = 100 * class_counts[class_id] / total_points if total_points > 0 else 0
            print(f"   Class {class_id}: {class_counts[class_id]:,} ({pct:.3f}%)")
        print(f"   Samples with Class 2 or 3: {samples_with_rare}/{len(self.data)} ({100*samples_with_rare/len(self.data):.1f}%)")

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

def hybrid_inference_collate_fn(batch):
    """Collate function for inference"""
    point_clouds, labels = zip(*batch)
    
    processed_batch = []
    original_labels = []
    
    for pc, label in zip(point_clouds, labels):
        # Voxelize for autoencoder (Stage 1)
        voxel_features = voxelize_hybrid(pc, for_autoencoder=True)
        voxel_tensor = torch.FloatTensor(voxel_features).permute(3, 0, 1, 2)
        
        processed_batch.append(voxel_tensor)
        original_labels.append(label)
    
    # Stack tensors
    voxel_batch = torch.stack(processed_batch)
    
    return voxel_batch, original_labels

def load_hybrid_models(device):
    """Load both stage 1 (autoencoder) and stage 2 (binary) models"""
    models = {}
    
    # Load Stage 1 autoencoder
    stage1_paths = [
        'hybrid_stage1_autoencoder.pth',
        'best_autoencoder_model.pth'
    ]
    
    stage1_model = None
    for path in stage1_paths:
        if os.path.exists(path):
            print(f"Loading Stage 1 autoencoder from {path}")
            checkpoint = torch.load(path, map_location=device)
            latent_dim = checkpoint.get('latent_dim', 256)
            
            stage1_model = Autoencoder3D(latent_dim=latent_dim)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Handle DataParallel wrapper
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            
            stage1_model.load_state_dict(state_dict)
            stage1_model = stage1_model.to(device)
            stage1_model.eval()
            
            if 'best_val_loss' in checkpoint:
                print(f"  📊 Stage 1 validation loss: {checkpoint['best_val_loss']:.6f}")
            
            break
    
    if stage1_model is None:
        print("❌ Stage 1 autoencoder model not found!")
        return None
    
    # Load Stage 2 binary classifier
    stage2_paths = [
        'hybrid_stage2_binary.pth'
    ]
    
    stage2_model = None
    for path in stage2_paths:
        if os.path.exists(path):
            print(f"Loading Stage 2 binary classifier from {path}")
            checkpoint = torch.load(path, map_location=device)
            
            stage2_model = HybridBinaryClassifier()
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Handle DataParallel wrapper
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            
            stage2_model.load_state_dict(state_dict)
            stage2_model = stage2_model.to(device)
            stage2_model.eval()
            
            if 'best_val_loss' in checkpoint:
                print(f"  📊 Stage 2 validation loss: {checkpoint['best_val_loss']:.4f}")
            
            break
    
    if stage2_model is None:
        print("❌ Stage 2 binary classifier model not found!")
        return None
    
    models['stage1'] = stage1_model
    models['stage2'] = stage2_model
    
    print("✅ Both hybrid models loaded successfully")
    return models

def run_hybrid_inference(models, dataloader, device, stage1_threshold_percentile=95):
    """Run two-stage hybrid inference"""
    print(f"🔍 Running hybrid two-stage inference...")
    print(f"📊 Stage 1 threshold: {stage1_threshold_percentile}th percentile of reconstruction error")
    
    stage1_model = models['stage1']
    stage2_model = models['stage2']
    
    # Stage 1: Compute reconstruction errors for all samples
    print("\n🏗️ Stage 1: Computing reconstruction errors...")
    all_reconstruction_errors = []
    all_sample_labels = []  # Ground truth for each sample (has rare class or not)
    all_original_labels = []
    
    with torch.no_grad():
        for batch_idx, (voxels, original_labels) in enumerate(dataloader):
            voxels = voxels.to(device)
            
            # Forward pass through autoencoder
            reconstructed, latent = stage1_model(voxels)
            
            # Compute reconstruction error per sample
            mse_per_sample = torch.mean((voxels - reconstructed) ** 2, dim=(1, 2, 3, 4))
            
            for i, (sample_error, original_label) in enumerate(zip(mse_per_sample, original_labels)):
                all_reconstruction_errors.append(sample_error.cpu().item())
                
                # Sample label: 1 if contains Class 2 or 3, 0 otherwise
                has_rare_class = np.any((original_label == 2) | (original_label == 3))
                all_sample_labels.append(int(has_rare_class))
                all_original_labels.append(original_label.numpy())
            
            if batch_idx % 10 == 0:
                print(f"  Processed batch {batch_idx + 1}/{len(dataloader)}")
    
    all_reconstruction_errors = np.array(all_reconstruction_errors)
    all_sample_labels = np.array(all_sample_labels)
    
    # Determine Stage 1 threshold
    stage1_threshold = np.percentile(all_reconstruction_errors, stage1_threshold_percentile)
    print(f"📊 Stage 1 reconstruction error threshold: {stage1_threshold:.6f}")
    
    # Identify anomalous samples (high reconstruction error)
    anomalous_indices = np.where(all_reconstruction_errors > stage1_threshold)[0]
    print(f"🎯 Stage 1 identified {len(anomalous_indices)} anomalous samples out of {len(all_reconstruction_errors)}")
    
    # Stage 1 performance evaluation
    stage1_predictions = (all_reconstruction_errors > stage1_threshold).astype(int)
    stage1_accuracy = np.mean(stage1_predictions == all_sample_labels)
    print(f"📈 Stage 1 accuracy (rare class detection): {stage1_accuracy:.3f}")
    
    if len(anomalous_indices) == 0:
        print("❌ No anomalous samples found in Stage 1. Cannot proceed to Stage 2.")
        return None
    
    # Stage 2: Binary classification on anomalous samples
    print(f"\n🎯 Stage 2: Binary classification on {len(anomalous_indices)} anomalous samples...")
    
    # Create dataset for Stage 2 (only anomalous samples with rare classes)
    stage2_predictions = {}  # Will store predictions for anomalous samples
    stage2_ground_truth = {}  # Ground truth for anomalous samples
    
    # Process anomalous samples for Stage 2
    anomalous_data = []
    anomalous_labels = []
    valid_anomalous_indices = []
    
    for idx in anomalous_indices:
        original_label = all_original_labels[idx]
        
        # Extract rare class points from this sample
        rare_mask = (original_label == 2) | (original_label == 3)
        
        if np.any(rare_mask):
            # Get point cloud for this sample
            point_cloud = dataloader.dataset.data[idx]
            
            if len(point_cloud) % 4 != 0:
                point_cloud = point_cloud[:len(point_cloud)//4 * 4]
            point_cloud = point_cloud.reshape(-1, 4)
            
            # Extract rare class points
            rare_pc = point_cloud[rare_mask]
            rare_labels = original_label[rare_mask]
            
            # Convert to binary: Class 2 = 0, Class 3 = 1
            binary_labels = (rare_labels == 3).astype(np.int64)
            
            anomalous_data.append(rare_pc)
            anomalous_labels.append(binary_labels)
            valid_anomalous_indices.append(idx)
    
    if len(anomalous_data) == 0:
        print("❌ No valid rare class samples found in anomalous data.")
        return None
    
    print(f"🔍 Stage 2 processing {len(anomalous_data)} samples with rare classes...")
    
    # Process Stage 2 data in batches
    stage2_batch_size = 4
    stage2_all_predictions = []
    stage2_all_true_labels = []
    
    with torch.no_grad():
        for i in range(0, len(anomalous_data), stage2_batch_size):
            batch_data = anomalous_data[i:i + stage2_batch_size]
            batch_labels = anomalous_labels[i:i + stage2_batch_size]
            
            batch_voxels = []
            batch_voxel_labels = []
            
            for pc, labels in zip(batch_data, batch_labels):
                # Voxelize for binary classification
                voxel_features, voxel_labels = voxelize_hybrid(
                    torch.FloatTensor(pc), 
                    torch.LongTensor(labels), 
                    for_autoencoder=False
                )
                
                voxel_tensor = torch.FloatTensor(voxel_features).permute(3, 0, 1, 2)
                batch_voxels.append(voxel_tensor)
                batch_voxel_labels.append(torch.LongTensor(voxel_labels))
            
            # Stack batch
            voxel_batch = torch.stack(batch_voxels).to(device)
            label_batch = torch.stack(batch_voxel_labels).to(device)
            
            # Stage 2 forward pass
            outputs = stage2_model(voxel_batch)
            
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            
            # Collect results
            for j in range(len(batch_data)):
                pred_flat = predicted[j].cpu().numpy().flatten()
                label_flat = label_batch[j].cpu().numpy().flatten()
                
                stage2_all_predictions.extend(pred_flat)
                stage2_all_true_labels.extend(label_flat)
    
    stage2_all_predictions = np.array(stage2_all_predictions)
    stage2_all_true_labels = np.array(stage2_all_true_labels)
    
    # Stage 2 performance evaluation
    stage2_accuracy = np.mean(stage2_all_predictions == stage2_all_true_labels)
    print(f"📈 Stage 2 accuracy (Class 2 vs 3): {stage2_accuracy:.3f}")
    
    # Combined results
    results = {
        'stage1': {
            'threshold': stage1_threshold,
            'reconstruction_errors': all_reconstruction_errors,
            'predictions': stage1_predictions,
            'true_labels': all_sample_labels,
            'accuracy': stage1_accuracy,
            'anomalous_samples': len(anomalous_indices),
            'total_samples': len(all_reconstruction_errors)
        },
        'stage2': {
            'predictions': stage2_all_predictions,
            'true_labels': stage2_all_true_labels,
            'accuracy': stage2_accuracy,
            'processed_samples': len(anomalous_data)
        },
        'original_labels': all_original_labels
    }
    
    return results

def evaluate_hybrid_performance(results, save_dir="hybrid_results"):
    """Evaluate and visualize hybrid approach performance"""
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n🏆 === HYBRID TWO-STAGE PERFORMANCE ANALYSIS ===")
    
    stage1_results = results['stage1']
    stage2_results = results['stage2']
    
    # Stage 1 Analysis
    print(f"\n🏗️ STAGE 1 (Anomaly Detection) Results:")
    print(f"   Total samples: {stage1_results['total_samples']}")
    print(f"   Anomalous samples identified: {stage1_results['anomalous_samples']}")
    print(f"   Detection rate: {100*stage1_results['anomalous_samples']/stage1_results['total_samples']:.1f}%")
    print(f"   Accuracy: {stage1_results['accuracy']:.3f}")
    
    # Stage 1 detailed metrics
    stage1_cm = confusion_matrix(stage1_results['true_labels'], stage1_results['predictions'])
    stage1_report = classification_report(stage1_results['true_labels'], stage1_results['predictions'],
                                        target_names=['Normal', 'Rare Classes'], zero_division=0)
    print(f"   Confusion Matrix:\n{stage1_cm}")
    print(f"   Classification Report:\n{stage1_report}")
    
    # Stage 2 Analysis
    print(f"\n🎯 STAGE 2 (Binary Classification) Results:")
    print(f"   Processed samples: {stage2_results['processed_samples']}")
    print(f"   Accuracy (Class 2 vs 3): {stage2_results['accuracy']:.3f}")
    
    # Stage 2 detailed metrics
    if len(stage2_results['true_labels']) > 0:
        stage2_cm = confusion_matrix(stage2_results['true_labels'], stage2_results['predictions'])
        stage2_report = classification_report(stage2_results['true_labels'], stage2_results['predictions'],
                                            target_names=['Class 2', 'Class 3'], zero_division=0)
        print(f"   Confusion Matrix:\n{stage2_cm}")
        print(f"   Classification Report:\n{stage2_report}")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Stage 1: Reconstruction error distribution
    normal_errors = stage1_results['reconstruction_errors'][stage1_results['true_labels'] == 0]
    anomaly_errors = stage1_results['reconstruction_errors'][stage1_results['true_labels'] == 1]
    
    axes[0, 0].hist(normal_errors, bins=30, alpha=0.7, label='Normal', color='blue', density=True)
    axes[0, 0].hist(anomaly_errors, bins=30, alpha=0.7, label='Rare Classes', color='red', density=True)
    axes[0, 0].axvline(stage1_results['threshold'], color='green', linestyle='--', 
                      label=f'Threshold: {stage1_results["threshold"]:.6f}')
    axes[0, 0].set_xlabel('Reconstruction Error')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Stage 1: Reconstruction Error Distribution')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # 2. Stage 1: ROC Curve
    fpr, tpr, _ = roc_curve(stage1_results['true_labels'], stage1_results['reconstruction_errors'])
    roc_auc = auc(fpr, tpr)
    
    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('Stage 1: ROC Curve')
    axes[0, 1].legend(loc="lower right")
    
    # 3. Stage 1: Confusion Matrix
    sns.heatmap(stage1_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Rare Classes'], 
                yticklabels=['Normal', 'Rare Classes'], ax=axes[0, 2])
    axes[0, 2].set_title('Stage 1: Confusion Matrix')
    
    # 4. Stage 2: Confusion Matrix (if data available)
    if len(stage2_results['true_labels']) > 0 and len(np.unique(stage2_results['true_labels'])) > 1:
        sns.heatmap(stage2_cm, annot=True, fmt='d', cmap='Greens',
                   xticklabels=['Class 2', 'Class 3'],
                   yticklabels=['Class 2', 'Class 3'], ax=axes[1, 0])
        axes[1, 0].set_title('Stage 2: Confusion Matrix (Class 2 vs 3)')
    else:
        axes[1, 0].text(0.5, 0.5, 'Insufficient Stage 2 data\nfor confusion matrix', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Stage 2: Limited Data')
    
    # 5. Overall pipeline flow
    stages = ['Input\nSamples', 'Stage 1\nAutoencoder', 'Anomalous\nSamples', 'Stage 2\nBinary', 'Final\nPredictions']
    counts = [
        stage1_results['total_samples'],
        stage1_results['total_samples'],
        stage1_results['anomalous_samples'],
        stage2_results['processed_samples'],
        stage2_results['processed_samples']
    ]
    
    axes[1, 1].bar(stages, counts, color=['lightblue', 'orange', 'red', 'green', 'purple'], alpha=0.7)
    axes[1, 1].set_ylabel('Number of Samples')
    axes[1, 1].set_title('Hybrid Pipeline Flow')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    for i, (stage, count) in enumerate(zip(stages, counts)):
        axes[1, 1].text(i, count + max(counts)*0.01, str(count), ha='center', va='bottom')
    
    # 6. Stage accuracies comparison
    stage_names = ['Stage 1\n(Anomaly Detection)', 'Stage 2\n(Binary Classification)', 'Combined\nPipeline']
    stage_accuracies = [
        stage1_results['accuracy'],
        stage2_results['accuracy'],
        stage1_results['accuracy'] * stage2_results['accuracy']  # Approximate combined accuracy
    ]
    
    bars = axes[1, 2].bar(stage_names, stage_accuracies, 
                         color=['orange', 'green', 'blue'], alpha=0.7)
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].set_title('Stage Performance Comparison')
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    for bar, acc in zip(bars, stage_accuracies):
        height = bar.get_height()
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'hybrid_analysis.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'hybrid_analysis.pdf'), bbox_inches='tight')
    
    # Save detailed results
    detailed_results = {
        'approach': 'hybrid_two_stage',
        'stage1': {
            'type': 'autoencoder_anomaly_detection',
            'threshold': float(stage1_results['threshold']),
            'accuracy': float(stage1_results['accuracy']),
            'total_samples': int(stage1_results['total_samples']),
            'anomalous_samples': int(stage1_results['anomalous_samples']),
            'detection_rate': float(stage1_results['anomalous_samples'] / stage1_results['total_samples']),
            'roc_auc': float(roc_auc)
        },
        'stage2': {
            'type': 'binary_classification_class2_vs_class3',
            'accuracy': float(stage2_results['accuracy']),
            'processed_samples': int(stage2_results['processed_samples'])
        },
        'combined_performance': {
            'approximate_accuracy': float(stage1_results['accuracy'] * stage2_results['accuracy']),
            'pipeline_efficiency': f"{stage1_results['anomalous_samples']}/{stage1_results['total_samples']} → {stage2_results['processed_samples']}"
        }
    }
    
    with open(os.path.join(save_dir, 'hybrid_results.json'), 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n📊 Analysis plots saved to {save_dir}/hybrid_analysis.*")
    print(f"📋 Results saved to {save_dir}/hybrid_results.json")
    
    return detailed_results

def main():
    # Configuration
    inference_data_file = "/nevis/houston/home/sc5303/INSS_2025/INSS2025/data/example_xyze_100.h5"
    inference_label_file = "/nevis/houston/home/sc5303/INSS_2025/INSS2025/data/example_label_100.h5"
    
    batch_size = 4
    stage1_threshold_percentile = 95  # Can be tuned
    
    # Check if files exist
    for file_path in [inference_data_file, inference_label_file]:
        if not os.path.exists(file_path):
            print(f"❌ Error: File not found - {file_path}")
            return
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    # Load hybrid models
    models = load_hybrid_models(device)
    if models is None:
        print("❌ Could not load hybrid models. Please train them first:")
        print("   python hybrid_train.py")
        return
    
    # Create inference dataset and dataloader
    inference_dataset = HybridInferenceDataset(inference_data_file, inference_label_file)
    inference_dataloader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False,
                                    collate_fn=hybrid_inference_collate_fn, num_workers=2)
    
    # Run hybrid inference
    results = run_hybrid_inference(models, inference_dataloader, device, stage1_threshold_percentile)
    
    if results is None:
        print("❌ Hybrid inference failed")
        return
    
    # Evaluate performance
    detailed_results = evaluate_hybrid_performance(results)
    
    print(f"\n🎉 Hybrid two-stage inference completed successfully!")
    print(f"📁 Results saved in 'hybrid_results/' directory")
    print(f"🔍 Stage 1 identified {results['stage1']['anomalous_samples']} anomalous samples")
    print(f"🎯 Stage 2 processed {results['stage2']['processed_samples']} samples for Class 2 vs 3")
    print(f"📊 Overall pipeline accuracy: {detailed_results['combined_performance']['approximate_accuracy']:.3f}")

if __name__ == "__main__":
    print("🔄 Hybrid Two-Stage Class 2 vs Class 3 Detection")
    main()
