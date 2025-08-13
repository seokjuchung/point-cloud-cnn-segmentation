import sys
import os
sys.path.append('/nevis/houston/home/sc5303/INSS_2025/point-cloud-cnn-segmentation')

try:
    from src.training.config import Config
    print("Config imported successfully")
    config = Config()
    print("Config created successfully")
    print(f"Data path: {config.data_path}")
except Exception as e:
    print(f"Error importing config: {e}")
    import traceback
    traceback.print_exc()

try:
    from src.data.dataset import Dataset
    print("Dataset imported successfully")
except Exception as e:
    print(f"Error importing dataset: {e}")
    import traceback
    traceback.print_exc()
