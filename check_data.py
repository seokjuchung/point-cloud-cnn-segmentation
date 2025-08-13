import h5py
import numpy as np

# Check the data structure
data_file = "data/train_xyze_1e4.h5"
label_file = "data/train_label_1e4.h5"

print("Checking data file structure...")
with h5py.File(data_file, 'r') as f:
    print("Keys in data file:", list(f.keys()))
    for key in f.keys():
        print(f"Shape of {key}:", f[key].shape)
        print(f"Data type of {key}:", f[key].dtype)
        if len(f[key].shape) > 0 and f[key].shape[0] > 0:
            print(f"Sample data from {key}:", f[key][0][:5] if len(f[key][0]) > 5 else f[key][0])

print("\nChecking label file structure...")
with h5py.File(label_file, 'r') as f:
    print("Keys in label file:", list(f.keys()))
    for key in f.keys():
        print(f"Shape of {key}:", f[key].shape)
        print(f"Data type of {key}:", f[key].dtype)
        if len(f[key].shape) > 0 and f[key].shape[0] > 0:
            print(f"Sample data from {key}:", f[key][0][:5] if len(f[key][0]) > 5 else f[key][0])
            print(f"Unique labels in {key}:", np.unique(f[key][:]))
