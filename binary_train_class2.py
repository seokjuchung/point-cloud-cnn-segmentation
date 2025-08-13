#!/usr/bin/env python3
"""
Binary classification to detect Class 2 points only
Converts multi-class problem to binary: Class 2 vs Not Class 2
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_recall_curve, roc_curve, auc, classification_report

class BinaryPointCloudDataset(Dataset):
    def __init__(self, data_file, label_file):
        print(f"Loading data from {data_file}")
        print(f"Loading labels from {label_file}")
        
        with h5py.File(data_file, 'r') as f:
            self.data = f['data'][:]
        
        with h5py.File(label_file, 'r') as f:
            self.labels = f['labels'][:]
        
        print(f"Loaded {len(self.data)} samples")
        
        # Analyze class distribution
        total_points = 0
        class2_points = 0
        
        for labels in self.labels:
            total_points += len(labels)
            class2_points += np.sum(labels == 2)
        
        print(f"📊 Dataset statistics:")
        print(f"   Total points: {total_points:,}")
        print(f"   Class 2 points: {class2_points:,} ({100*class2_points/total_points:.3f}%)")
        print(f"   Other points: {total_points-class2_points:,} ({100*(total_points-class2_points)/total_points:.3f}%)")
        print(f"   Class imbalance ratio: {(total_points-class2_points)/class2_points:.1f}:1")

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
        
        # Convert to binary labels: 1 for Class 2, 0 for all others
        binary_labels = (labels == 2).astype(np.int64)
        
        return torch.FloatTensor(point_cloud), torch.LongTensor(binary_labels)

def voxelize_binary_point_cloud(point_cloud, binary_labels, grid_size=64, voxel_size=8):
    """Convert point cloud to voxel grid for binary classification"""
    # Initialize voxel grids
    voxel_features = np.zeros((grid_size, grid_size, grid_size, 4), dtype=np.float32)
    voxel_labels = np.zeros((grid_size, grid_size, grid_size), dtype=np.int64)
    
    # Convert to numpy arrays if they're tensors
    if hasattr(point_cloud, 'numpy'):
        point_cloud_np = point_cloud.numpy()
    else:
        point_cloud_np = point_cloud
    
    if hasattr(binary_labels, 'numpy'):
        labels_np = binary_labels.numpy()
    else:
        labels_np = binary_labels
    
    # Convert coordinates to voxel indices
    voxel_coords = (point_cloud_np[:, :3] // voxel_size).astype(int)
    
    # Clip coordinates to grid bounds
    voxel_coords = np.clip(voxel_coords, 0, grid_size - 1)
    
    # Fill voxel grid - prioritize Class 2 points
    for i, (x, y, z) in enumerate(voxel_coords):
        if i < len(labels_np):
            # If current point is Class 2, always use it
            if labels_np[i] == 1:  # Binary label 1 = original Class 2
                voxel_features[x, y, z] = point_cloud_np[i].copy()
                voxel_labels[x, y, z] = labels_np[i]
            # If voxel is empty or current has higher energy, use it
            elif voxel_labels[x, y, z] == 0 or voxel_features[x, y, z, 3] < point_cloud_np[i, 3]:
                voxel_features[x, y, z] = point_cloud_np[i].copy()
                voxel_labels[x, y, z] = labels_np[i]
    
    return voxel_features, voxel_labels

class BinarySimple3DCNN(nn.Module):
    def __init__(self):
        super(BinarySimple3DCNN, self).__init__()
        
        # Input: (batch, 4, 64, 64, 64)
        self.encoder = nn.Sequential(
            # First conv block
            nn.Conv3d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 32x32x32
            
            # Second conv block
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 16x16x16
            
            # Third conv block
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 8x8x8
        )
        
        self.decoder = nn.Sequential(
            # Upsample 8x8x8 -> 16x16x16
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            # Upsample 16x16x16 -> 32x32x32
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            # Upsample 32x32x32 -> 64x64x64
            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            
            # Final classification layer - Binary output
            nn.Conv3d(16, 2, kernel_size=1),  # 2 classes: Not Class 2, Class 2
        )
    
    def forward(self, x):
        # Encoder
        encoded = self.encoder(x)
        
        # Decoder
        decoded = self.decoder(encoded)
        
        return decoded

def binary_collate_fn(batch):
    """Custom collate function for binary classification"""
    point_clouds, binary_labels = zip(*batch)
    
    processed_batch = []
    
    for pc, label in zip(point_clouds, binary_labels):
        # Voxelize each point cloud
        voxel_features, voxel_labels = voxelize_binary_point_cloud(pc, label, grid_size=64, voxel_size=8)
        
        # Convert to tensor and rearrange dimensions for conv3d (C, H, W, D)
        voxel_tensor = torch.FloatTensor(voxel_features).permute(3, 0, 1, 2)
        label_tensor = torch.LongTensor(voxel_labels)
        
        processed_batch.append((voxel_tensor, label_tensor))
    
    # Stack tensors
    voxel_batch = torch.stack([item[0] for item in processed_batch])
    label_batch = torch.stack([item[1] for item in processed_batch])
    
    return voxel_batch, label_batch

def compute_binary_class_weights(dataset):
    """Compute class weights for binary classification"""
    total_class2 = 0
    total_other = 0
    
    print("🔍 Computing class weights for binary classification...")
    
    for i in range(len(dataset)):
        _, binary_labels = dataset[i]
        total_class2 += torch.sum(binary_labels == 1).item()
        total_other += torch.sum(binary_labels == 0).item()
    
    total_samples = total_class2 + total_other
    
    # Compute inverse frequency weights
    weight_other = total_samples / (2 * total_other)
    weight_class2 = total_samples / (2 * total_class2)
    
    class_weights = torch.FloatTensor([weight_other, weight_class2])
    
    print(f"📊 Binary class distribution:")
    print(f"   Not Class 2 (0): {total_other:,} samples")
    print(f"   Class 2 (1): {total_class2:,} samples")
    print(f"   Imbalance ratio: {total_other/total_class2:.1f}:1")
    print(f"   Class weights: [Not Class 2: {weight_other:.3f}, Class 2: {weight_class2:.3f}]")
    
    return class_weights

def train_binary_model():
    # Configuration
    data_file = "/nevis/houston/home/sc5303/INSS_2025/INSS2025/data/train_xyze_1e4.h5"
    label_file = "/nevis/houston/home/sc5303/INSS_2025/INSS2025/data/train_label_1e4.h5"
    
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 30
    early_stopping_patience = 10
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    if torch.cuda.device_count() > 1:
        print(f"🔥 Using {torch.cuda.device_count()} GPUs")
    
    # Create dataset
    full_dataset = BinaryPointCloudDataset(data_file, label_file)
    
    # Train/validation split (9:1)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    print(f"📚 Dataset split: {train_size} train, {val_size} validation")
    
    # Compute class weights from training set only
    class_weights = compute_binary_class_weights(train_dataset)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=binary_collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=binary_collate_fn, num_workers=4)
    
    # Initialize model
    model = BinarySimple3DCNN()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # Loss and optimizer with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    
    # Training tracking
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Early stopping variables
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    print(f"\n🎯 Starting binary training for Class 2 detection...")
    print(f"📋 Config: batch_size={batch_size}, lr={learning_rate}, epochs={num_epochs}")
    print(f"⏰ Early stopping patience: {early_stopping_patience} epochs")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (voxels, labels) in enumerate(train_loader):
            voxels = voxels.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(voxels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            train_total += labels.numel()
            train_correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for voxels, labels in val_loader:
                voxels = voxels.to(device)
                labels = labels.to(device)
                
                outputs = model(voxels)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                val_total += labels.numel()
                val_correct += (predicted == labels).sum().item()
        
        # Calculate epoch metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} ({100*train_acc:.2f}%)")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f} ({100*val_acc:.2f}%)")
        
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
                'class_weights': class_weights,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies
            }, 'best_binary_model_class2.pth')
            print(f"  ✅ New best model saved! (val_loss: {best_val_loss:.4f})")
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
        'class_weights': class_weights,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }, 'final_binary_model_class2.pth')
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Binary Training Loss (Class 2 Detection)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Train Accuracy', color='blue')
    plt.plot(val_accuracies, label='Validation Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Binary Training Accuracy (Class 2 Detection)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot([100*acc for acc in train_accuracies], label='Train Accuracy (%)', color='blue')
    plt.plot([100*acc for acc in val_accuracies], label='Validation Accuracy (%)', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Binary Training Accuracy % (Class 2 Detection)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('binary_class2_training_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig('binary_class2_training_curves.pdf', bbox_inches='tight')
    
    print(f"\n🎉 Binary training completed!")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")
    print(f"💾 Models saved: best_binary_model_class2.pth, final_binary_model_class2.pth")
    print(f"📈 Training curves saved: binary_class2_training_curves.*")

if __name__ == "__main__":
    print("🔍 Binary Point Cloud CNN for Class 2 Detection")
    train_binary_model()
