import torch
import torch.nn as nn
import h5py
import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset import Dataset
from src.data.data_loader import DataLoader
from src.models.cnn_model import CNNModel
from src.training.trainer import Trainer
from src.training.config import Config

def main():
    print("Starting point cloud semantic segmentation training...")
    
    # Load the configuration
    print("Loading configuration...")
    config = Config()
    
    # Set up distributed training if multiple GPUs
    if config.use_multi_gpu:
        torch.backends.cudnn.benchmark = True
    
    # Initialize the dataset and data loader
    print("Initializing dataset...")
    dataset = Dataset(config.data_path)
    print(f"Data path: {config.data_path}")
    
    train_dataset = dataset.get_train_dataset()
    print(f"Dataset size: {len(train_dataset)}")
    
    # Test one sample
    print("Testing one sample...")
    sample_pc, sample_labels = train_dataset[0]
    print(f"Sample point cloud shape: {sample_pc.shape}")
    print(f"Sample labels shape: {sample_labels.shape}")
    
    data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    # Initialize the model
    print("Initializing model...")
    model = CNNModel(num_classes=config.num_classes)
    
    # Move model to GPU and wrap with DataParallel for multi-GPU
    if torch.cuda.is_available():
        model = model.to(config.device)
        if config.use_multi_gpu:
            model = nn.DataParallel(model)
    
    # Initialize the trainer
    print("Initializing trainer...")
    trainer = Trainer(model, data_loader, config)
    
    # Start training
    print("Starting training...")
    trainer.train()

if __name__ == "__main__":
    main()
