import os
import torch

class Config:
    def __init__(self):
        # Data configuration
        # Get the absolute path to the data directory
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_path = os.path.join(current_dir, "data")
        self.num_classes = 5  # Labels 0-4
        
        # Training configuration
        self.batch_size = 8  # Adjust based on GPU memory
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.weight_decay = 1e-4
        
        # Multi-GPU configuration
        self.use_multi_gpu = torch.cuda.device_count() > 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gpus = torch.cuda.device_count()
        
        # Model configuration
        self.voxel_size = 8  # Voxelization size
        self.grid_size = 96  # 768/8 = 96
        
        print(f"Using device: {self.device}")
        print(f"Number of GPUs available: {self.num_gpus}")
        if self.use_multi_gpu:
            print(f"Multi-GPU training enabled with {self.num_gpus} GPUs")