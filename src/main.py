import torch
import torch.nn as nn
import h5py
import numpy as np
from src.data.dataset import Dataset
from src.data.data_loader import DataLoader
from src.models.cnn_model import CNNModel
from src.training.trainer import Trainer
from src.training.config import Config

def main():
    # Load the configuration
    config = Config()
    
    # Set up distributed training if multiple GPUs
    if config.use_multi_gpu:
        torch.backends.cudnn.benchmark = True
    
    # Initialize the dataset and data loader
    dataset = Dataset(config.data_path)
    train_dataset = dataset.get_train_dataset()
    data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    # Initialize the model
    model = CNNModel(num_classes=config.num_classes)
    
    # Move model to GPU and wrap with DataParallel for multi-GPU
    if torch.cuda.is_available():
        model = model.to(config.device)
        if config.use_multi_gpu:
            model = nn.DataParallel(model)
    
    # Initialize the trainer
    trainer = Trainer(model, data_loader, config)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()