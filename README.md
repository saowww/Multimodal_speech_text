# Multimodal Emotion Recognition

## Algorithmic Scheme of Multimodal Emotion Recognition

![Multimodal Emotion Recognition Architecture](img/figure_multimodel_mamba.png)

## Project Structure

```
Multimodal_speech_text/
├── data/                    # Data processing modules
│   ├── download.py         # Dataset download
│   ├── audio_extractor.py  # Audio feature extraction
│   ├── text_extractor.py   # Text feature extraction
│   └── split.py            # Data splitting
├── model/                   # Model architectures
│   ├── transformer_model.py # Transformer-based model
│   └── mamba_model.py       # Mamba-based model
├── utils/                   # Utility functions
│   ├── data_loader.py      # Data loading
│   ├── train.py            # Training functions with wandb integration
│   ├── plot.py             # Plotting functions
│   └── utils.py            # Utility functions
├── main.py                  # Main training script
└── README.md               # Project documentation
```

## Features

- Multimodal emotion recognition using speech and text features
- Two model architectures: Transformer and Mamba
- Wandb integration for experiment tracking
- Comprehensive visualization and evaluation

## Usage

```bash
python main.py
```

## Requirements

- PyTorch
- transformers
- wandb
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- tqdm
