#!/usr/bin/env python3
"""
Simple CNN-based semantic segmentation for point clouds
Using voxelization approach
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import os

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
    
    # Convert coordinates to voxel indices
    voxel_coords = (point_cloud[:, :3].numpy() // voxel_size).astype(int)
    
    # Clip coordinates to grid bounds
    voxel_coords = np.clip(voxel_coords, 0, grid_size - 1)
    
    # Fill voxel grid
    for i, (x, y, z) in enumerate(voxel_coords):
        if i < len(labels):
            # Keep the maximum energy if multiple points in same voxel
            if voxel_features[x, y, z, 3] < point_cloud[i, 3]:
                voxel_features[x, y, z] = point_cloud[i]
                voxel_labels[x, y, z] = labels[i]
    
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

def train_model():
    # Configuration
    data_path = "data"
    data_file = os.path.join(data_path, "train_xyze_1e4.h5")
    label_file = os.path.join(data_path, "train_label_1e4.h5")
    
    batch_size = 64  # Start small due to memory constraints with 3D data
    num_epochs = 10
    learning_rate = 0.001
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    dataset = PointCloudDataset(data_file, label_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          collate_fn=collate_fn, num_workers=2)
    
    # Create model
    model = Simple3DCNN(num_classes=5)
    model = model.to(device)
    
    # Use DataParallel for multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore background
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (voxels, labels) in enumerate(dataloader):
            voxels = voxels.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(voxels)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
        
        avg_loss = epoch_loss / num_batches
        print(f'Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
            print(f'Model saved at epoch {epoch+1}')

if __name__ == "__main__":
    print("Starting Point Cloud Semantic Segmentation Training")
    train_model()
    print("Training completed!")
