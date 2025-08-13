import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np

class Trainer:
    def __init__(self, model, data_loader, config):
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.device = config.device
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Use weighted cross entropy for class imbalance
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore empty voxels
        
        # Mixed precision training for faster training on multiple GPUs
        self.scaler = GradScaler()
        
    def voxelize_point_cloud(self, point_cloud, labels, grid_size=96, voxel_size=8):
        """Convert point cloud to voxel grid"""
        # Initialize voxel grid with zeros for features and -1 for labels (ignore index)
        voxel_grid = np.zeros((grid_size, grid_size, grid_size, 4), dtype=np.float32)
        label_grid = np.full((grid_size, grid_size, grid_size), -1, dtype=np.int64)
        
        # Convert coordinates to voxel indices
        voxel_coords = (point_cloud[:, :3] / voxel_size).astype(int)
        
        # Clip coordinates to grid bounds
        voxel_coords = np.clip(voxel_coords, 0, grid_size - 1)
        
        # Fill voxel grid (using maximum energy if multiple points in same voxel)
        for i, (x, y, z) in enumerate(voxel_coords):
            if i < len(labels):
                # Only update if this voxel has higher energy or is empty
                if voxel_grid[x, y, z, 3] < point_cloud[i, 3]:
                    voxel_grid[x, y, z] = point_cloud[i]
                    label_grid[x, y, z] = labels[i]
        
        return voxel_grid, label_grid
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        dataloader = self.data_loader.get_dataloader()
        
        for batch_idx, (point_clouds, labels) in enumerate(dataloader):
            # Process each point cloud in the batch
            batch_voxels = []
            batch_labels = []
            
            for pc, label in zip(point_clouds, labels):
                # Voxelize point cloud and labels
                voxel_grid, label_grid = self.voxelize_point_cloud(
                    pc.numpy(), 
                    label.numpy(),
                    self.config.grid_size, 
                    self.config.voxel_size
                )
                
                # Convert to tensor and add to batch
                voxel_tensor = torch.FloatTensor(voxel_grid).permute(3, 0, 1, 2)  # (C, H, W, D)
                label_tensor = torch.LongTensor(label_grid)
                
                batch_voxels.append(voxel_tensor)
                batch_labels.append(label_tensor)
            
            # Stack batch tensors
            batch_voxels = torch.stack(batch_voxels).to(self.device)
            batch_labels = torch.stack(batch_labels).to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast():
                outputs = self.model(batch_voxels)
                # Reshape for loss calculation
                outputs = outputs.view(-1, self.config.num_classes)
                targets = batch_labels.view(-1)
                loss = self.criterion(outputs, targets)
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        return total_loss / num_batches
    
    def train(self):
        print(f"Starting training on {self.device}")
        print(f"Using {torch.cuda.device_count()} GPUs" if self.config.use_multi_gpu else f"Using single GPU: {self.device}")
        
        for epoch in range(self.config.num_epochs):
            avg_loss = self.train_epoch(epoch)
            print(f'Epoch {epoch+1}/{self.config.num_epochs}, Average Loss: {avg_loss:.4f}')
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)
    
    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pth')
        print(f'Checkpoint saved for epoch {epoch+1}')