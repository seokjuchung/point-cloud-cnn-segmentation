# Point Cloud CNN Segmentation

This project implements a Convolutional Neural Network (CNN) for semantic segmentation on point clouds. The model is trained using point cloud energy data and corresponding labels stored in HDF5 files.

## Project Structure

```
point-cloud-cnn-segmentation
├── src
│   ├── data
│   │   ├── dataset.py       # Handles loading and preprocessing of point cloud data
│   │   └── data_loader.py   # Manages batching and shuffling of the dataset
│   ├── models
│   │   ├── cnn_model.py     # Defines the CNN architecture for semantic segmentation
│   │   └── losses.py        # Contains loss functions for training
│   ├── training
│   │   ├── trainer.py       # Orchestrates the training process
│   │   └── config.py        # Configuration settings for training
│   └── utils
│       ├── visualization.py  # Functions for visualizing point clouds and results
│       └── metrics.py       # Functions to compute evaluation metrics
├── data
│   ├── train_xyze_1e4.h5    # Training dataset with point cloud energy data
│   └── train_label_1e4.h5    # Training labels corresponding to the point cloud data
├── notebooks
│   └── exploration.ipynb     # Jupyter notebook for exploratory data analysis
├── configs
│   └── train_config.yaml     # Configuration settings for training in YAML format
├── requirements.txt          # Python dependencies required for the project
└── README.md                 # Documentation for the project
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd point-cloud-cnn-segmentation
pip install -r requirements.txt
```

## Usage

1. Prepare your data in the `data` directory, ensuring that the point cloud energy data and labels are in HDF5 format.
2. Modify the training configuration in `configs/train_config.yaml` as needed.
3. Run the main script to start training:

```bash
python src/main.py
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

Written by Claude Sonnet 4 using Copilot