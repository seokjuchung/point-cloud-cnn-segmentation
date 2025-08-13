#!/usr/bin/env python3
"""
Hybrid two-stage approach for Class 2 vs Class 3 distinction:
Stage 1: Autoencoder to filter out common classes (0, 1, 4) 
Stage 2: Binary classifier to distinguish Class 2 vs Class 3 in anomalous data
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report

class HybridAnomalyDataset(Dataset):
    """Dataset for hybrid approach: remove common classes, focus on Class 2 vs 3"""
    def __init__(self, data_file, label_file, stage='autoencoder'):
        print(f"Loading data from {data_file}")
        print(f"Loading labels from {label_file}")
        
        with h5py.File(data_file, 'r') as f:
            self.data = f['data'][:]
        
        with h5py.File(label_file, 'r') as f:
            self.labels = f['labels'][:]
        
        self.stage = stage
        
        if stage == 'autoencoder':
            print("🔧 Stage 1: Preparing data for autoencoder (filtering out Class 2 & 3)...")
            self.data, self.labels = self._filter_rare_classes()
        elif stage == 'binary':
            print("🔧 Stage 2: Preparing data for binary classification (Class 2 vs 3 only)...")
            self.data, self.labels = self._extract_rare_classes()
        
        print(f"Loaded {len(self.data)} samples for {stage} stage")
        self._analyze_dataset()

    def _filter_rare_classes(self):
        """Remove Class 2 and 3 points for autoencoder training (keep 0, 1, 4)"""
        filtered_data = []
        filtered_labels = []
        
        total_original_points = 0
        total_filtered_points = 0
        removed_class2_points = 0
        removed_class3_points = 0
        
        for i in range(len(self.data)):
            # Get point cloud and labels
            point_cloud = self.data[i]
            labels = self.labels[i]
            
            # Reshape point cloud
            if len(point_cloud) % 4 != 0:
                point_cloud = point_cloud[:len(point_cloud)//4 * 4]
            point_cloud = point_cloud.reshape(-1, 4)
            
            total_original_points += len(labels)
            
            # Create mask to keep common classes (0, 1, 4)
            common_mask = (labels == 0) | (labels == 1) | (labels == 4)
            
            # Filter points and labels
            filtered_pc = point_cloud[common_mask]
            filtered_lab = labels[common_mask]
            
            total_filtered_points += len(filtered_lab)
            removed_class2_points += np.sum(labels == 2)
            removed_class3_points += np.sum(labels == 3)
            
            # Only keep samples that have points after filtering
            if len(filtered_pc) > 0:
                filtered_data.append(filtered_pc.flatten())
                filtered_labels.append(filtered_lab)
        
        print(f"📋 Autoencoder filtering results:")
        print(f"   Original points: {total_original_points:,}")
        print(f"   Kept points (Classes 0,1,4): {total_filtered_points:,}")
        print(f"   Removed Class 2: {removed_class2_points:,}")
        print(f"   Removed Class 3: {removed_class3_points:,}")
        print(f"   Retention rate: {100*total_filtered_points/total_original_points:.2f}%")
        
        return np.array(filtered_data, dtype=object), np.array(filtered_labels, dtype=object)

    def _extract_rare_classes(self):
        """Extract only Class 2 and 3 points for binary classification"""
        rare_data = []
        rare_labels = []
        
        total_original_points = 0
        total_class2_points = 0
        total_class3_points = 0
        samples_with_rare_classes = 0
        
        for i in range(len(self.data)):
            # Get point cloud and labels
            point_cloud = self.data[i]
            labels = self.labels[i]
            
            # Reshape point cloud
            if len(point_cloud) % 4 != 0:
                point_cloud = point_cloud[:len(point_cloud)//4 * 4]
            point_cloud = point_cloud.reshape(-1, 4)
            
            total_original_points += len(labels)
            
            # Create mask for rare classes (2, 3)
            rare_mask = (labels == 2) | (labels == 3)
            
            if np.any(rare_mask):
                samples_with_rare_classes += 1
                
                # Filter points and labels
                rare_pc = point_cloud[rare_mask]
                rare_lab = labels[rare_mask]
                
                # Convert to binary labels: Class 2 = 0, Class 3 = 1
                binary_lab = (rare_lab == 3).astype(np.int64)
                
                total_class2_points += np.sum(rare_lab == 2)
                total_class3_points += np.sum(rare_lab == 3)
                
                rare_data.append(rare_pc.flatten())
                rare_labels.append(binary_lab)
        
        print(f"📋 Binary extraction results:")
        print(f"   Samples with rare classes: {samples_with_rare_classes}")
        print(f"   Class 2 points: {total_class2_points:,} (binary label 0)")
        print(f"   Class 3 points: {total_class3_points:,} (binary label 1)")
        print(f"   Class balance ratio: {total_class2_points}:{total_class3_points}")
        
        return np.array(rare_data, dtype=object), np.array(rare_labels, dtype=object)

    def _analyze_dataset(self):
        """Analyze the processed dataset"""
        if self.stage == 'autoencoder':
            total_points = 0
            class_counts = {0: 0, 1: 0, 4: 0}  # Only common classes remain
            
            for labels in self.labels:
                if isinstance(labels, np.ndarray) and labels.dtype == object:
                    labels = labels.item()
                total_points += len(labels)
                for class_id in [0, 1, 4]:
                    class_counts[class_id] += np.sum(labels == class_id)
            
            print(f"📊 Autoencoder dataset (common classes only):")
            print(f"   Total points: {total_points:,}")
            for class_id in [0, 1, 4]:
                pct = 100 * class_counts[class_id] / total_points if total_points > 0 else 0
                print(f"   Class {class_id}: {class_counts[class_id]:,} ({pct:.1f}%)")
        
        elif self.stage == 'binary':
            total_points = 0
            class2_count = 0  # binary label 0
            class3_count = 0  # binary label 1
            
            for labels in self.labels:
                if isinstance(labels, np.ndarray) and labels.dtype == object:
                    labels = labels.item()
                total_points += len(labels)
                class2_count += np.sum(labels == 0)
                class3_count += np.sum(labels == 1)
            
            print(f"📊 Binary dataset (rare classes only):")
            print(f"   Total points: {total_points:,}")
            print(f"   Class 2 (label 0): {class2_count:,} ({100*class2_count/total_points:.1f}%)")
            print(f"   Class 3 (label 1): {class3_count:,} ({100*class3_count/total_points:.1f}%)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get point cloud data
        point_cloud = self.data[idx]
        
        # Handle variable length arrays
        if isinstance(point_cloud, np.ndarray) and point_cloud.dtype == object:
            point_cloud = point_cloud.item()
        
        # Reshape to (N, 4) for (x, y, z, energy)
        if len(point_cloud) % 4 != 0:
            point_cloud = point_cloud[:len(point_cloud)//4 * 4]
        point_cloud = point_cloud.reshape(-1, 4)
        
        if self.stage == 'autoencoder':
            return torch.FloatTensor(point_cloud)
        else:  # binary stage
            labels = self.labels[idx]
            if isinstance(labels, np.ndarray) and labels.dtype == object:
                labels = labels.item()
            return torch.FloatTensor(point_cloud), torch.LongTensor(labels)

def voxelize_hybrid(point_cloud, labels=None, grid_size=64, voxel_size=8, for_autoencoder=True):
    """Voxelize for hybrid approach"""
    if for_autoencoder:
        # Same as autoencoder voxelization
        voxel_features = np.zeros((grid_size, grid_size, grid_size, 4), dtype=np.float32)
        
        # Convert to numpy if tensor
        if hasattr(point_cloud, 'numpy'):
            point_cloud_np = point_cloud.numpy()
        else:
            point_cloud_np = point_cloud
        
        # Convert coordinates to voxel indices
        voxel_coords = (point_cloud_np[:, :3] // voxel_size).astype(int)
        voxel_coords = np.clip(voxel_coords, 0, grid_size - 1)
        
        # Fill voxel grid
        for i, (x, y, z) in enumerate(voxel_coords):
            if voxel_features[x, y, z, 3] < point_cloud_np[i, 3]:
                voxel_features[x, y, z] = point_cloud_np[i].copy()
        
        return voxel_features
    else:
        # Binary classification voxelization
        voxel_features = np.zeros((grid_size, grid_size, grid_size, 4), dtype=np.float32)
        voxel_labels = np.zeros((grid_size, grid_size, grid_size), dtype=np.int64)
        
        # Convert to numpy arrays
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
        voxel_coords = np.clip(voxel_coords, 0, grid_size - 1)
        
        # Fill voxel grid
        for i, (x, y, z) in enumerate(voxel_coords):
            if i < len(labels_np):
                if voxel_features[x, y, z, 3] < point_cloud_np[i, 3]:
                    voxel_features[x, y, z] = point_cloud_np[i].copy()
                    voxel_labels[x, y, z] = labels_np[i]
        
        return voxel_features, voxel_labels

def hybrid_autoencoder_collate_fn(batch):
    """Collate function for autoencoder stage"""
    processed_batch = []
    
    for point_cloud in batch:
        # Voxelize point cloud
        voxel_features = voxelize_hybrid(point_cloud, for_autoencoder=True)
        voxel_tensor = torch.FloatTensor(voxel_features).permute(3, 0, 1, 2)
        processed_batch.append(voxel_tensor)
    
    return torch.stack(processed_batch)

def hybrid_binary_collate_fn(batch):
    """Collate function for binary stage"""
    point_clouds, labels = zip(*batch)
    processed_batch = []
    
    for pc, label in zip(point_clouds, labels):
        # Voxelize each point cloud
        voxel_features, voxel_labels = voxelize_hybrid(pc, label, for_autoencoder=False)
        voxel_tensor = torch.FloatTensor(voxel_features).permute(3, 0, 1, 2)
        label_tensor = torch.LongTensor(voxel_labels)
        processed_batch.append((voxel_tensor, label_tensor))
    
    # Stack tensors
    voxel_batch = torch.stack([item[0] for item in processed_batch])
    label_batch = torch.stack([item[1] for item in processed_batch])
    
    return voxel_batch, label_batch

# Import existing architectures
from autoencoder_train import Autoencoder3D

class HybridBinaryClassifier(nn.Module):
    """Binary classifier for Class 2 vs Class 3 distinction"""
    def __init__(self):
        super(HybridBinaryClassifier, self).__init__()
        
        # Similar architecture to the multi-class CNN but binary output
        self.encoder = nn.Sequential(
            # Input: (4, 64, 64, 64)
            nn.Conv3d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 32x32x32
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 16x16x16
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 8x8x8
        )
        
        self.decoder = nn.Sequential(
            # Upsample back to original size
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            
            # Binary classification: Class 2 vs Class 3
            nn.Conv3d(16, 2, kernel_size=1),
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_hybrid_stage1_autoencoder():
    """Stage 1: Train autoencoder on common classes (0, 1, 4)"""
    print("🏗️ STAGE 1: Training Autoencoder on Common Classes")
    print("=" * 60)
    
    # Configuration
    data_file = "/nevis/houston/home/sc5303/INSS_2025/INSS2025/data/train_xyze_1e4.h5"
    label_file = "/nevis/houston/home/sc5303/INSS_2025/INSS2025/data/train_label_1e4.h5"
    
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 256
    latent_dim = 256
    early_stopping_patience = 32
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    # Create dataset (common classes only)
    full_dataset = HybridAnomalyDataset(data_file, label_file, stage='autoencoder')
    
    # Train/validation split
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    print(f"📚 Dataset split: {train_size} train, {val_size} validation")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=hybrid_autoencoder_collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=hybrid_autoencoder_collate_fn, num_workers=4)
    
    # Initialize model
    model = Autoencoder3D(latent_dim=latent_dim)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    
    # Training tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    print(f"\n🏗️ Starting Stage 1 training...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, voxels in enumerate(train_loader):
            voxels = voxels.to(device)
            
            optimizer.zero_grad()
            reconstructed, latent = model(voxels)
            loss = criterion(reconstructed, voxels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 25 == 0:
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
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
                'latent_dim': latent_dim,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, 'hybrid_stage1_autoencoder.pth')
            print(f"  ✅ Stage 1 best model saved! (val_loss: {best_val_loss:.6f})")
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\n🛑 Stage 1 early stopping after {epoch + 1} epochs")
            break
    
    print(f"\n🎉 Stage 1 (Autoencoder) training completed!")
    print(f"📊 Best validation loss: {best_val_loss:.6f}")
    
    return model

def train_hybrid_stage2_binary():
    """Stage 2: Train binary classifier on Class 2 vs Class 3"""
    print("\n🎯 STAGE 2: Training Binary Classifier (Class 2 vs 3)")
    print("=" * 60)
    
    # Configuration
    data_file = "/nevis/houston/home/sc5303/INSS_2025/INSS2025/data/train_xyze_1e4.h5"
    label_file = "/nevis/houston/home/sc5303/INSS_2025/INSS2025/data/train_label_1e4.h5"
    
    batch_size = 128  # Can be larger since we have much less data
    learning_rate = 0.001
    num_epochs = 256
    early_stopping_patience = 16
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    # Create dataset (rare classes only: 2 vs 3)
    full_dataset = HybridAnomalyDataset(data_file, label_file, stage='binary')
    
    if len(full_dataset) == 0:
        print("❌ No samples with Class 2 or 3 found in training data!")
        return None
    
    # Train/validation split
    train_size = max(1, int(0.8 * len(full_dataset)))  # Use 80/20 split due to small data
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    print(f"📚 Dataset split: {train_size} train, {val_size} validation")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=hybrid_binary_collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=hybrid_binary_collate_fn, num_workers=2)
    
    # Initialize model
    model = HybridBinaryClassifier()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # Loss and optimizer (no weights needed since Class 2 and 3 have similar frequency)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8, factor=0.5, verbose=True)
    
    # Training tracking
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    print(f"\n🎯 Starting Stage 2 training...")
    
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
            
            if batch_idx % 5 == 0:
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
        train_acc = train_correct / train_total if train_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.3f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.3f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies
            }, 'hybrid_stage2_binary.pth')
            print(f"  ✅ Stage 2 best model saved! (val_loss: {best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\n🛑 Stage 2 early stopping after {epoch + 1} epochs")
            break
    
    print(f"\n🎉 Stage 2 (Binary Classifier) training completed!")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")
    
    return model

def main():
    print("🔄 HYBRID TWO-STAGE APPROACH FOR CLASS 2 vs 3 DISTINCTION")
    print("=" * 70)
    print("Stage 1: Autoencoder filters out common classes (0, 1, 4)")
    print("Stage 2: Binary classifier distinguishes Class 2 vs Class 3")
    print()
    
    # Stage 1: Train autoencoder
    autoencoder_model = train_hybrid_stage1_autoencoder()
    
    # Stage 2: Train binary classifier
    binary_model = train_hybrid_stage2_binary()
    
    print(f"\n🏆 HYBRID TRAINING COMPLETED!")
    print(f"💾 Models saved:")
    print(f"   - hybrid_stage1_autoencoder.pth (filters common classes)")
    print(f"   - hybrid_stage2_binary.pth (distinguishes Class 2 vs 3)")
    print(f"")
    print(f"📋 Next steps:")
    print(f"   1. Create hybrid inference script")
    print(f"   2. Run inference: Stage 1 → Stage 2")
    print(f"   3. Compare with single-stage approaches")

if __name__ == "__main__":
    main()
