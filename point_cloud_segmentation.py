import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
from tqdm import tqdm
import time
from collections import Counter
from sklearn.metrics import f1_score, classification_report

# Set device to use all available GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"Number of GPUs available: {torch.cuda.device_count()}")

class PointCloudDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data_file = h5py.File(data_path, 'r')
        self.label_file = h5py.File(label_path, 'r')
        self.num_events = len(self.data_file['data'])
        print(f"Dataset loaded with {self.num_events} events")
        
    def __len__(self):
        return self.num_events
    
    def __getitem__(self, idx):
        # Load data and labels for event idx
        # Data is stored as variable length arrays, need to reshape to (N, 4)
        points_flat = self.data_file['data'][idx]
        points = torch.tensor(points_flat.reshape(-1, 4), dtype=torch.float32)  # Shape: (N, 4) - x,y,z,e
        labels = torch.tensor(self.label_file['labels'][idx], dtype=torch.long)    # Shape: (N,)
        return points, labels
    
    def __del__(self):
        if hasattr(self, 'data_file'):
            self.data_file.close()
        if hasattr(self, 'label_file'):
            self.label_file.close()

def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    points_list, labels_list = zip(*batch)
    
    # Get batch size and max number of points
    batch_size = len(points_list)
    max_points = max([p.shape[0] for p in points_list])
    
    # Create padded tensors
    points_padded = torch.zeros(batch_size, max_points, 4)
    labels_padded = torch.full((batch_size, max_points), -1, dtype=torch.long)  # Use -1 for padding
    masks = torch.zeros(batch_size, max_points, dtype=torch.bool)
    
    for i, (points, labels) in enumerate(zip(points_list, labels_list)):
        num_points = points.shape[0]
        points_padded[i, :num_points] = points
        labels_padded[i, :num_points] = labels
        masks[i, :num_points] = True
    
    return points_padded, labels_padded, masks

class PointNetSegmentation(nn.Module):
    def __init__(self, num_classes, input_dim=4):
        super(PointNetSegmentation, self).__init__()
        
        # Point-wise MLPs for feature extraction
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        
        # Global feature extraction
        self.global_feat = nn.Conv1d(1024, 1024, 1)
        
        # Segmentation head
        self.seg_conv1 = nn.Conv1d(1088, 512, 1)  # 1024 global + 64 local
        self.seg_conv2 = nn.Conv1d(512, 256, 1)
        self.seg_conv3 = nn.Conv1d(256, 128, 1)
        self.seg_conv4 = nn.Conv1d(128, num_classes, 1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn_global = nn.BatchNorm1d(1024)
        self.bn_seg1 = nn.BatchNorm1d(512)
        self.bn_seg2 = nn.BatchNorm1d(256)
        self.bn_seg3 = nn.BatchNorm1d(128)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x shape: (batch_size, max_points, 4)
        batch_size, max_points, _ = x.shape
        
        # Transpose for conv1d: (batch_size, 4, max_points)
        x = x.transpose(1, 2)
        
        # Point-wise feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        point_feat = F.relu(self.bn2(self.conv2(x)))  # Save for skip connection
        x = F.relu(self.bn3(self.conv3(point_feat)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Global feature extraction
        global_feat = F.relu(self.bn_global(self.global_feat(x)))
        global_feat = torch.max(global_feat, 2, keepdim=True)[0]  # Global max pooling
        
        # Expand global features to match point features
        global_feat_expanded = global_feat.repeat(1, 1, max_points)
        
        # Concatenate global and local features
        concat_feat = torch.cat([point_feat, global_feat_expanded], dim=1)
        
        # Segmentation head
        x = F.relu(self.bn_seg1(self.seg_conv1(concat_feat)))
        x = self.dropout(x)
        x = F.relu(self.bn_seg2(self.seg_conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_seg3(self.seg_conv3(x)))
        x = self.seg_conv4(x)
        
        # Transpose back: (batch_size, max_points, num_classes)
        x = x.transpose(1, 2)
        
        return x

def train_model():
    # Data paths
    data_path = '/nevis/houston/home/sc5303/INSS_2025/point-cloud-cnn-segmentation/data/train_xyze_1e4.h5'
    label_path = '/nevis/houston/home/sc5303/INSS_2025/point-cloud-cnn-segmentation/data/train_label_1e4.h5'
    # data_path = '/nevis/houston/home/sc5303/INSS_2025/INSS2025/data/example_xyze_100.h5'
    # label_path = '/nevis/houston/home/sc5303/INSS_2025/INSS2025/data/example_label_100.h5'

    # Load dataset
    print("Loading dataset...")
    dataset = PointCloudDataset(data_path, label_path)
    
    # Check number of classes by examining more samples for better statistics
    print("Analyzing dataset to determine number of classes and class distribution...")
    all_labels = []
    for i in range(min(1000, len(dataset))):  # Check first 1000 samples for better statistics
        _, labels = dataset[i]
        all_labels.extend(labels.numpy())
    
    num_classes = len(set(all_labels))
    print(f"Number of classes detected: {num_classes}")
    print(f"Classes: {sorted(set(all_labels))}")
    
    # Calculate class distribution and weights
    class_counts = Counter(all_labels)
    total_samples = len(all_labels)
    
    print("Class distribution:")
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = count / total_samples * 100
        print(f"  Class {class_id}: {count} samples ({percentage:.2f}%)")
    
    # Calculate class weights - inverse frequency with extra emphasis on class 2
    class_weights = []
    max_count = max(class_counts.values())
    for class_id in range(num_classes):
        if class_id in class_counts:
            # Use inverse frequency weighting
            weight = max_count / class_counts[class_id]
            # Give extra weight to class 2 to improve its F1 score
            if class_id == 2:
                weight *= 2.0  # Double the weight for class 2
            class_weights.append(weight)
        else:
            class_weights.append(1.0)
    
    # Normalize weights so they sum to num_classes
    weight_sum = sum(class_weights)
    class_weights = [w * num_classes / weight_sum for w in class_weights]
    
    print("Class weights (normalized):")
    for i, weight in enumerate(class_weights):
        print(f"  Class {i}: {weight:.3f}")
    
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    # Split dataset into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Data loaders
    batch_size = 64  # Adjust based on GPU memory
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=collate_fn, num_workers=32)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           collate_fn=collate_fn, num_workers=32)
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Model
    model = PointNetSegmentation(num_classes=num_classes)
    
    # Use multiple GPUs if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights_tensor)  # Use class weights
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training loop
    num_epochs = 128
    best_val_loss = float('inf')
    best_f1_class2 = 0.0  # Track best F1 score for class 2
    patience = 16
    patience_counter = 0
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (points, labels, masks) in enumerate(train_pbar):
            points = points.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(points)  # Shape: (batch_size, max_points, num_classes)
            
            # Reshape for loss calculation
            outputs_flat = outputs.contiguous().view(-1, num_classes)
            labels_flat = labels.contiguous().view(-1)
            
            # Calculate loss only on non-padded points
            loss = criterion(outputs_flat, labels_flat)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            
            # Accuracy calculation (only for non-padded points)
            _, predicted = torch.max(outputs, 2)
            mask_flat = masks.contiguous().view(-1)
            correct = ((predicted.contiguous().view(-1) == labels_flat) & mask_flat).sum().item()
            total = mask_flat.sum().item()
            train_correct += correct
            train_total += total
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total if total > 0 else 0:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total if train_total > 0 else 0
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for points, labels, masks in val_pbar:
                points = points.to(device)
                labels = labels.to(device)
                masks = masks.to(device)
                
                outputs = model(points)
                
                # Reshape for loss calculation
                outputs_flat = outputs.contiguous().view(-1, num_classes)
                labels_flat = labels.contiguous().view(-1)
                
                loss = criterion(outputs_flat, labels_flat)
                val_loss += loss.item()
                
                # Accuracy calculation
                _, predicted = torch.max(outputs, 2)
                mask_flat = masks.contiguous().view(-1)
                correct = ((predicted.contiguous().view(-1) == labels_flat) & mask_flat).sum().item()
                total = mask_flat.sum().item()
                val_correct += correct
                val_total += total
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total if total > 0 else 0:.2f}%'
                })
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total if val_total > 0 else 0
        
        # Calculate F1 scores for validation set
        model.eval()
        all_predictions = []
        all_true_labels = []
        
        with torch.no_grad():
            for points, labels, masks in val_loader:
                points = points.to(device)
                labels = labels.to(device)
                masks = masks.to(device)
                
                outputs = model(points)
                _, predicted = torch.max(outputs, 2)
                
                # Flatten and apply masks
                pred_flat = predicted.contiguous().view(-1)
                labels_flat = labels.contiguous().view(-1)
                mask_flat = masks.contiguous().view(-1)
                
                # Only keep non-padded predictions
                valid_pred = pred_flat[mask_flat]
                valid_labels = labels_flat[mask_flat]
                
                all_predictions.extend(valid_pred.cpu().numpy())
                all_true_labels.extend(valid_labels.cpu().numpy())
        
        # Calculate F1 scores
        f1_macro = f1_score(all_true_labels, all_predictions, average='macro')
        f1_weighted = f1_score(all_true_labels, all_predictions, average='weighted')
        f1_per_class = f1_score(all_true_labels, all_predictions, average=None)
        
        # Focus on class 2 F1 score
        f1_class2 = f1_per_class[2] if len(f1_per_class) > 2 else 0.0
        
        # Update learning rate
        scheduler.step()
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'F1 Macro: {f1_macro:.4f}, F1 Weighted: {f1_weighted:.4f}')
        print(f'F1 Class 2: {f1_class2:.4f} (target class)')
        print(f'F1 per class: {[f"{f:.3f}" for f in f1_per_class]}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model based on class 2 F1 score (with fallback to val_loss)
        model_improved = False
        if f1_class2 > best_f1_class2:
            best_f1_class2 = f1_class2
            best_val_loss = val_loss  # Update best val loss too
            model_improved = True
            save_reason = f'F1 Class 2: {f1_class2:.4f}'
        elif f1_class2 == best_f1_class2 and val_loss < best_val_loss:
            best_val_loss = val_loss
            model_improved = True
            save_reason = f'Val Loss: {val_loss:.4f} (same F1 Class 2: {f1_class2:.4f})'
        
        if model_improved:
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'f1_class2': f1_class2,
                'f1_per_class': f1_per_class.tolist(),
                'num_classes': num_classes
            }, 'best_model.pth')
            print(f'New best model saved with {save_reason}')
        else:
            patience_counter += 1
            print(f'No improvement for {patience_counter}/{patience} epochs')
        
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping triggered after {patience} epochs without improvement')
            break
        
        print('-' * 80)
    
    print("Training completed!")
    return model, num_classes

def inference_example(model_path='best_model.pth'):
    """Example of how to use the trained model for inference"""
    # Load the trained model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    num_classes = checkpoint['num_classes']
    
    model = PointNetSegmentation(num_classes=num_classes)
    
    # Load state dict - handle DataParallel case
    state_dict = checkpoint['model_state_dict']
    
    # Check if the saved model has 'module.' prefix
    has_module_prefix = any(key.startswith('module.') for key in state_dict.keys())
    
    if torch.cuda.device_count() > 1:
        # We want to use DataParallel for inference
        if not has_module_prefix:
            # Add 'module.' prefix to state_dict keys
            new_state_dict = {}
            for key, value in state_dict.items():
                new_state_dict['module.' + key] = value
            state_dict = new_state_dict
        model = nn.DataParallel(model)
    else:
        # Single GPU - remove 'module.' prefix if present
        if has_module_prefix:
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('module.', '')
                new_state_dict[new_key] = value
            state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # Load test data (using same dataset for example)
    data_path = '/nevis/houston/home/sc5303/INSS_2025/point-cloud-cnn-segmentation/data/train_xyze_1e4.h5'
    label_path = '/nevis/houston/home/sc5303/INSS_2025/point-cloud-cnn-segmentation/data/train_label_1e4.h5'
    
    dataset = PointCloudDataset(data_path, label_path)
    
    # Test on a single event
    event_idx = 0
    points, true_labels = dataset[event_idx]
    print(f"\nTesting on event {event_idx}")
    print(f"Number of points: {points.shape[0]}")
    print(f"Input shape: {points.shape}")
    
    # Prepare input (add batch dimension)
    points_batch = points.unsqueeze(0).to(device)  # Shape: (1, N, 4)
    
    with torch.no_grad():
        outputs = model(points_batch)  # Shape: (1, N, num_classes)
        predictions = torch.argmax(outputs, dim=2).squeeze(0)  # Shape: (N,)
    
    predictions = predictions.cpu().numpy()
    true_labels = true_labels.numpy()
    
    print(f"Predicted classes: {predictions}")
    print(f"True classes: {true_labels}")
    print(f"Accuracy: {(predictions == true_labels).mean() * 100:.2f}%")
    
    return predictions

if __name__ == "__main__":
    print("Point Cloud Semantic Segmentation Training")
    print("=" * 50)
    
    # Train the model
    model, num_classes = train_model()
    
    print("\nTraining finished! Running inference example...")
    
    # Run inference example
    try:
        predictions = inference_example()
        print(f"\nInference successful! Predictions shape: {predictions.shape}")
    except Exception as e:
        print(f"Inference failed: {e}")
    
    print("\nTo use the model for new data:")
    print("1. Load the saved model from 'best_model.pth'")
    print("2. Prepare your point cloud data as tensor with shape (N, 4)")
    print("3. Add batch dimension: points.unsqueeze(0)")
    print("4. Run model(points) to get predictions")
    print("5. Use torch.argmax(outputs, dim=2) to get class predictions")
