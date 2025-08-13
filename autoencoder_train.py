#!/usr/bin/env python3
"""
Autoencoder-based anomaly detection for Class 2 point detection
Train on non-Class-2 data, detect Class 2 as anomalies with high reconstruction error
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import os
import json

class NormalPointCloudDataset(Dataset):
    """Dataset containing only non-Class-2 points for autoencoder training"""
    def __init__(self, data_file, label_file, filter_class2=True):
        print(f"Loading data from {data_file}")
        print(f"Loading labels from {label_file}")
        
        with h5py.File(data_file, 'r') as f:
            self.data = f['data'][:]
        
        with h5py.File(label_file, 'r') as f:
            self.labels = f['labels'][:]
        
        if filter_class2:
            print("🔧 Filtering out Class 2 points for autoencoder training...")
            self.data, self.labels = self._filter_class2_points()
        
        print(f"Loaded {len(self.data)} samples")
        
        # Analyze filtered dataset
        total_points = 0
        class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        
        for labels in self.labels:
            total_points += len(labels)
            for class_id in range(5):
                class_counts[class_id] += np.sum(labels == class_id)
        
        print(f"📊 Filtered dataset statistics:")
        print(f"   Total points: {total_points:,}")
        for class_id in range(5):
            pct = 100 * class_counts[class_id] / total_points if total_points > 0 else 0
            print(f"   Class {class_id}: {class_counts[class_id]:,} ({pct:.3f}%)")

    def _filter_class2_points(self):
        """Remove Class 2 points from the dataset"""
        filtered_data = []
        filtered_labels = []
        
        total_original_points = 0
        total_filtered_points = 0
        removed_class2_points = 0
        
        for i in range(len(self.data)):
            # Get point cloud and labels
            point_cloud = self.data[i]
            labels = self.labels[i]
            
            # Reshape point cloud
            if len(point_cloud) % 4 != 0:
                point_cloud = point_cloud[:len(point_cloud)//4 * 4]
            point_cloud = point_cloud.reshape(-1, 4)
            
            total_original_points += len(labels)
            
            # Create mask to keep non-Class-2 points
            non_class2_mask = labels != 2
            
            # Filter points and labels
            filtered_pc = point_cloud[non_class2_mask]
            filtered_lab = labels[non_class2_mask]
            
            total_filtered_points += len(filtered_lab)
            removed_class2_points += np.sum(labels == 2)
            
            # Only keep samples that have points after filtering
            if len(filtered_pc) > 0:
                # Flatten back for storage
                filtered_data.append(filtered_pc.flatten())
                filtered_labels.append(filtered_lab)
        
        print(f"📋 Filtering results:")
        print(f"   Original points: {total_original_points:,}")
        print(f"   Filtered points: {total_filtered_points:,}")
        print(f"   Removed Class 2 points: {removed_class2_points:,}")
        print(f"   Retention rate: {100*total_filtered_points/total_original_points:.2f}%")
        
        return np.array(filtered_data, dtype=object), np.array(filtered_labels, dtype=object)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get point cloud data and reshape to (N, 4)
        point_cloud = self.data[idx]
        
        # Handle variable length arrays
        if isinstance(point_cloud, np.ndarray) and point_cloud.dtype == object:
            point_cloud = point_cloud.item()
        
        # Reshape to (N, 4) for (x, y, z, energy)
        if len(point_cloud) % 4 != 0:
            point_cloud = point_cloud[:len(point_cloud)//4 * 4]
        
        point_cloud = point_cloud.reshape(-1, 4)
        
        return torch.FloatTensor(point_cloud)

def voxelize_for_autoencoder(point_cloud, grid_size=64, voxel_size=8):
    """Convert point cloud to voxel grid for autoencoder"""
    # Initialize voxel grid with features only (no labels needed)
    voxel_features = np.zeros((grid_size, grid_size, grid_size, 4), dtype=np.float32)
    
    # Convert to numpy if tensor
    if hasattr(point_cloud, 'numpy'):
        point_cloud_np = point_cloud.numpy()
    else:
        point_cloud_np = point_cloud
    
    # Convert coordinates to voxel indices
    voxel_coords = (point_cloud_np[:, :3] // voxel_size).astype(int)
    
    # Clip coordinates to grid bounds
    voxel_coords = np.clip(voxel_coords, 0, grid_size - 1)
    
    # Fill voxel grid - use maximum energy for overlapping points
    for i, (x, y, z) in enumerate(voxel_coords):
        if voxel_features[x, y, z, 3] < point_cloud_np[i, 3]:
            voxel_features[x, y, z] = point_cloud_np[i].copy()
    
    return voxel_features

class Autoencoder3D(nn.Module):
    """3D CNN Autoencoder for point cloud reconstruction"""
    def __init__(self, latent_dim=512):
        super(Autoencoder3D, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: (4, 64, 64, 64)
            nn.Conv3d(4, 32, kernel_size=4, stride=2, padding=1),  # -> (32, 32, 32, 32)
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1), # -> (64, 16, 16, 16)
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1), # -> (128, 8, 8, 8)
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1), # -> (256, 4, 4, 4)
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )
        
        # Flatten and bottleneck
        self.flatten = nn.Flatten()
        self.encoder_fc = nn.Linear(256 * 4 * 4 * 4, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, 256 * 4 * 4 * 4)
        self.unflatten = nn.Unflatten(1, (256, 4, 4, 4))
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1), # -> (128, 8, 8, 8)
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 16, 16, 16)
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),   # -> (32, 32, 32, 32)
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose3d(32, 4, kernel_size=4, stride=2, padding=1),    # -> (4, 64, 64, 64)
            nn.Tanh()  # Assuming normalized input
        )
    
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        flattened = self.flatten(encoded)
        latent = self.encoder_fc(flattened)
        
        # Decode
        decoded_flat = self.decoder_fc(latent)
        unflattened = self.unflatten(decoded_flat)
        reconstructed = self.decoder(unflattened)
        
        return reconstructed, latent

def autoencoder_collate_fn(batch):
    """Custom collate function for autoencoder training"""
    processed_batch = []
    
    for point_cloud in batch:
        # Voxelize point cloud
        voxel_features = voxelize_for_autoencoder(point_cloud, grid_size=64, voxel_size=8)
        
        # Convert to tensor and rearrange dimensions for conv3d (C, H, W, D)
        voxel_tensor = torch.FloatTensor(voxel_features).permute(3, 0, 1, 2)
        
        processed_batch.append(voxel_tensor)
    
    # Stack tensors
    voxel_batch = torch.stack(processed_batch)
    
    return voxel_batch

def train_autoencoder():
    """Train autoencoder on non-Class-2 data"""
    # Configuration
    data_file = "/nevis/houston/home/sc5303/INSS_2025/INSS2025/data/train_xyze_1e4.h5"
    label_file = "/nevis/houston/home/sc5303/INSS_2025/INSS2025/data/train_label_1e4.h5"
    
    batch_size = 8  # Smaller batch size for autoencoder
    learning_rate = 0.001
    num_epochs = 50
    latent_dim = 256
    early_stopping_patience = 15
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    if torch.cuda.device_count() > 1:
        print(f"🔥 Using {torch.cuda.device_count()} GPUs")
    
    # Create dataset (filtered to remove Class 2)
    full_dataset = NormalPointCloudDataset(data_file, label_file, filter_class2=True)
    
    # Train/validation split (9:1)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    print(f"📚 Dataset split: {train_size} train, {val_size} validation")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=autoencoder_collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=autoencoder_collate_fn, num_workers=4)
    
    # Initialize model
    model = Autoencoder3D(latent_dim=latent_dim)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()  # Reconstruction loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    
    # Training tracking
    train_losses = []
    val_losses = []
    
    # Early stopping variables
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    print(f"\n🏗️ Starting autoencoder training (Class 2 filtered out)...")
    print(f"📋 Config: batch_size={batch_size}, lr={learning_rate}, epochs={num_epochs}, latent_dim={latent_dim}")
    print(f"⏰ Early stopping patience: {early_stopping_patience} epochs")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, voxels in enumerate(train_loader):
            voxels = voxels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass - autoencoder tries to reconstruct input
            reconstructed, latent = model(voxels)
            loss = criterion(reconstructed, voxels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.6f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for voxels in val_loader:
                voxels = voxels.to(device)
                reconstructed, latent = model(voxels)
                loss = criterion(reconstructed, voxels)
                val_loss += loss.item()
        
        # Calculate epoch metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss: {avg_val_loss:.6f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'latent_dim': latent_dim,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, 'best_autoencoder_model.pth')
            print(f"  ✅ New best autoencoder saved! (val_loss: {best_val_loss:.6f})")
        else:
            epochs_without_improvement += 1
            print(f"  ⏰ No improvement for {epochs_without_improvement}/{early_stopping_patience} epochs")
        
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\n🛑 Early stopping triggered after {epoch + 1} epochs")
            break
        
        print("-" * 60)
    
    # Save final model
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_val_loss': avg_val_loss,
        'latent_dim': latent_dim,
        'train_losses': train_losses,
        'val_losses': val_losses
    }, 'final_autoencoder_model.pth')
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss (MSE)')
    plt.title('Autoencoder Training Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses[-20:], label='Train Loss (Last 20)', color='blue')
    plt.plot(val_losses[-20:], label='Validation Loss (Last 20)', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss (MSE)')
    plt.title('Autoencoder Training Loss (Last 20 Epochs)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('autoencoder_training_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig('autoencoder_training_curves.pdf', bbox_inches='tight')
    
    print(f"\n🎉 Autoencoder training completed!")
    print(f"📊 Best validation loss: {best_val_loss:.6f}")
    print(f"💾 Models saved: best_autoencoder_model.pth, final_autoencoder_model.pth")
    print(f"📈 Training curves saved: autoencoder_training_curves.*")

if __name__ == "__main__":
    print("🔧 Training Autoencoder for Class 2 Anomaly Detection")
    train_autoencoder()
