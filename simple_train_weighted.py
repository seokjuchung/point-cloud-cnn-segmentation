#!/usr/bin/env python3
"""
Simple CNN-based semantic segmentation for point clouds with WEIGHTED LOSS
Using voxelization approach and class-balanced training
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import time
import os
import matplotlib.pyplot as plt
import json

class PointCloudDataset(Dataset):
    def __init__(self, data_file, label_file):
        print(f"Loading data from {data_file}")
        print(f"Loading labels from {label_file}")
        
        with h5py.File(data_file, 'r') as f:
            self.data = f['data'][:]
        
        with h5py.File(label_file, 'r') as f:
            self.labels = f['labels'][:]
        
        print(f"Loaded {len(self.data)} samples")

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

def voxelize_point_cloud(point_cloud, labels, grid_size=64, voxel_size=8):
    """Convert point cloud to voxel grid"""
    # Initialize voxel grids
    voxel_features = np.zeros((grid_size, grid_size, grid_size, 4), dtype=np.float32)
    voxel_labels = np.zeros((grid_size, grid_size, grid_size), dtype=np.int64)
    
    # Convert to numpy arrays if they're tensors
    if hasattr(point_cloud, 'numpy'):
        point_cloud_np = point_cloud.numpy()
    else:
        point_cloud_np = point_cloud
    
    if hasattr(labels, 'numpy'):
        labels_np = labels.numpy()
    else:
        labels_np = labels
    
    # Convert coordinates to voxel indices
    voxel_coords = (point_cloud_np[:, :3] // voxel_size).astype(int)
    
    # Clip coordinates to grid bounds
    voxel_coords = np.clip(voxel_coords, 0, grid_size - 1)
    
    # Fill voxel grid
    for i, (x, y, z) in enumerate(voxel_coords):
        if i < len(labels_np):
            # Keep the maximum energy if multiple points in same voxel
            if voxel_features[x, y, z, 3] < point_cloud_np[i, 3]:
                voxel_features[x, y, z] = point_cloud_np[i].copy()
                voxel_labels[x, y, z] = labels_np[i]
    
    return voxel_features, voxel_labels

class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(Simple3DCNN, self).__init__()
        
        # Input: (batch, 4, 64, 64, 64)
        self.encoder = nn.Sequential(
            # First conv block
            nn.Conv3d(4, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),  # (32, 32, 32, 32)
            
            # Second conv block  
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),  # (64, 16, 16, 16)
            
            # Third conv block
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2),  # (128, 8, 8, 8)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Upsample 1
            nn.ConvTranspose3d(128, 64, 2, stride=2),  # (64, 16, 16, 16)
            nn.BatchNorm3d(64),
            nn.ReLU(),
            
            # Upsample 2
            nn.ConvTranspose3d(64, 32, 2, stride=2),   # (32, 32, 32, 32)
            nn.BatchNorm3d(32),
            nn.ReLU(),
            
            # Upsample 3
            nn.ConvTranspose3d(32, 16, 2, stride=2),   # (16, 64, 64, 64)
            nn.BatchNorm3d(16),
            nn.ReLU(),
            
            # Final conv
            nn.Conv3d(16, num_classes, 1)  # (num_classes, 64, 64, 64)
        )
        
    def forward(self, x):
        # Encoder
        encoded = self.encoder(x)
        
        # Decoder
        decoded = self.decoder(encoded)
        
        return decoded

def collate_fn(batch):
    """Custom collate function for variable-length point clouds"""
    point_clouds, labels = zip(*batch)
    
    processed_batch = []
    for pc, label in zip(point_clouds, labels):
        # Voxelize each point cloud
        voxel_features, voxel_labels = voxelize_point_cloud(pc, label, grid_size=64, voxel_size=8)
        
        # Convert to tensor and rearrange dimensions for conv3d (C, H, W, D)
        voxel_tensor = torch.FloatTensor(voxel_features).permute(3, 0, 1, 2)
        label_tensor = torch.LongTensor(voxel_labels)
        
        processed_batch.append((voxel_tensor, label_tensor))
    
    # Stack tensors
    voxel_batch = torch.stack([item[0] for item in processed_batch])
    label_batch = torch.stack([item[1] for item in processed_batch])
    
    return voxel_batch, label_batch

def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for voxels, labels in dataloader:
            voxels = voxels.to(device)
            labels = labels.to(device)
            
            outputs = model(voxels)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def save_loss_curves(train_losses, val_losses, save_dir="results"):
    """Save loss curves as plot and JSON"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss Curves (Weighted Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curves_weighted.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'loss_curves_weighted.pdf'), bbox_inches='tight')
    print(f"Loss curves saved to {save_dir}/loss_curves_weighted.png and .pdf")
    
    # Save loss data as JSON
    loss_data = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epochs': list(epochs)
    }
    with open(os.path.join(save_dir, 'loss_data_weighted.json'), 'w') as f:
        json.dump(loss_data, f, indent=2)
    
    print(f"Loss data saved to {save_dir}/loss_data_weighted.json")

def train_model():
    # Configuration
    data_path = "data"
    data_file = os.path.join(data_path, "train_xyze_1e4.h5")
    label_file = os.path.join(data_path, "train_label_1e4.h5")
    
    batch_size = 128  # Slightly reduce batch size for more stable training with weighted loss
    num_epochs = 128  # Increase epochs for better convergence with class imbalance
    learning_rate = 0.0005  # Lower learning rate for more stable training
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = PointCloudDataset(data_file, label_file)
    
    # Split dataset into train/val (9:1 ratio)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], 
                                             generator=torch.Generator().manual_seed(42))
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                collate_fn=collate_fn, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=2)
    
    # Create model
    model = Simple3DCNN(num_classes=5)
    model = model.to(device)
    
    # Use DataParallel for multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Class weights based on inference analysis
    # Class 0: 99.78%, Class 1: 0.06%, Class 2: 0.002%, Class 3: 0.002%, Class 4: 0.15%
    # Using inverse frequency weighting
    class_weights = torch.tensor([
        1.0,        # Class 0 (most frequent, baseline weight)
        1663.0,     # Class 1 (99.78/0.06 ≈ 1663)
        49890.0,    # Class 2 (99.78/0.002 ≈ 49890) 
        49890.0,    # Class 3 (99.78/0.002 ≈ 49890)
        665.0       # Class 4 (99.78/0.15 ≈ 665)
    ]).to(device)
    
    print(f"Using class-balanced weighted loss:")
    print(f"  Class weights: {class_weights}")
    
    # Loss and optimizer with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Early stopping and best model tracking
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    early_stopping_patience = 16
    
    # Lists to store loss values
    train_losses = []
    val_losses = []
    
    print("Starting class-balanced training...")
    print(f"Note: Higher loss values expected due to weighted loss for minority classes")
    print(f"Early stopping: Will stop if validation loss doesn't improve for {early_stopping_patience} epochs")
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        num_train_batches = 0
        
        for batch_idx, (voxels, labels) in enumerate(train_dataloader):
            voxels = voxels.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(voxels)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            num_train_batches += 1
            
            if batch_idx % 20 == 0:  # Print less frequently due to higher loss values
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.4f}')
        
        # Calculate average training loss
        avg_train_loss = epoch_train_loss / num_train_batches
        
        # Validation phase
        avg_val_loss = evaluate_model(model, val_dataloader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store losses
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Check for best model and early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            
            # Save best model
            best_model_path = 'best_model_weighted.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'class_weights': class_weights,
                'best_val_loss': best_val_loss
            }, best_model_path)
            print(f'✅ New best model saved! Val Loss: {avg_val_loss:.4f} -> {best_model_path}')
        else:
            epochs_without_improvement += 1
            print(f'❌ No improvement for {epochs_without_improvement}/{early_stopping_patience} epochs')
        
        print(f'Epoch {epoch+1}/{num_epochs} completed.')
        print(f'  Training Loss: {avg_train_loss:.4f}')
        print(f'  Validation Loss: {avg_val_loss:.4f}')
        print(f'  Best Val Loss: {best_val_loss:.4f}')
        print(f'  Learning Rate: {current_lr:.6f}')
        print()
        
        # Early stopping check
        if epochs_without_improvement >= early_stopping_patience:
            print(f"🛑 Early stopping triggered after {early_stopping_patience} epochs without improvement")
            print(f"Best validation loss: {best_val_loss:.4f} at epoch {epoch + 1 - epochs_without_improvement}")
            break
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'model_weighted_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'class_weights': class_weights,
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
    
    # Save final model and loss curves
    final_model_path = 'final_model_weighted.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'class_weights': class_weights,
        'best_val_loss': best_val_loss,
        'epochs_trained': len(train_losses),
        'early_stopped': epochs_without_improvement >= early_stopping_patience,
        'config': {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'weighted_loss': True,
            'early_stopping_patience': early_stopping_patience
        }
    }, final_model_path)
    
    save_loss_curves(train_losses, val_losses)
    
    print("🎉 Class-balanced training completed!")
    print(f"📊 Epochs trained: {len(train_losses)}")
    print(f"📈 Final Training Loss: {train_losses[-1]:.4f}")
    print(f"📉 Final Validation Loss: {val_losses[-1]:.4f}")
    print(f"🏆 Best Validation Loss: {best_val_loss:.4f}")
    
    if epochs_without_improvement >= early_stopping_patience:
        print(f"⏰ Training stopped early due to no improvement for {early_stopping_patience} epochs")
    else:
        print(f"✅ Training completed all {num_epochs} epochs")
    
    print(f"💾 Final model saved as: '{final_model_path}'")
    print(f"🏅 Best model saved as: 'best_model_weighted.pth'")
    print("🔍 Next step: Run inference with the best model!")

if __name__ == "__main__":
    print("Starting Point Cloud Semantic Segmentation with Weighted Loss")
    train_model()
