# MNIST 99.4% Accuracy Model

A PyTorch implementation achieving 99.4% test accuracy on MNIST using less than 20,000 parameters.

## Architecture Highlights

- **Parameters**: Less than 20,000
- **Best Accuracy**: 99.4%
- **Architecture Type**: Custom CNN with residual connections
- **Training Time**: ~20 epochs

### Key Features
- Efficient channel progression (1->18->32->18->32)
- Dilated convolutions for larger receptive field
- Residual connections for better gradient flow
- Batch normalization and minimal dropout (0.008)
- Global Average Pooling instead of large FC layers

## Model Architecture

```
Input Block (1->18 channels)
└── Conv2d(1, 18, 3x3) + BN + ReLU + Dropout(0.008)

CONV BLOCK 1 (18->32 channels)
└── Conv2d(18, 32, 3x3) + BN + ReLU + Dropout(0.008)

Transition Block 1 with Skip Connection
└── Conv2d(32, 18, 1x1) + BN + ReLU + MaxPool2d(2,2)

CONV BLOCK 2 with Dilation (18->32 channels)
└── Conv2d(18, 32, 3x3, dilation=2) + BN + ReLU + Dropout(0.008)

Transition Block 2
└── Conv2d(32, 18, 1x1) + BN + ReLU + MaxPool2d(2,2)

CONV BLOCK 3 (18->32 channels)
└── Conv2d(18, 32, 3x3) + BN + ReLU + Dropout(0.008)

Output Block
└── Conv2d(32, 10, 1x1) + BN + ReLU + GAP
```

## Requirements

```bash
pip install -r requirements.txt
```

## Training

```bash
python model/model.py --epochs 20 --target-accuracy 99.4
```

## Model Features

1. **Efficient Feature Extraction**:
   - Progressive channel growth
   - Dilated convolutions
   - Skip connections

2. **Regularization**:
   - Minimal dropout (0.008)
   - Batch normalization
   - Data augmentation

3. **Training Optimizations**:
   - OneCycleLR scheduler
   - Adam optimizer
   - Cosine annealing

## Results

- Training accuracy: >96%
- Test accuracy: 99.4%
- Parameters: <20,000
- Epochs to target: ~20

## Training Configuration

```python
# Optimizer
optimizer = Adam(lr=0.01, weight_decay=1e-4)

# Learning Rate Schedule
scheduler = OneCycleLR(
    max_lr=0.01,
    epochs=20,
    steps_per_epoch=len(train_loader),
    pct_start=0.2,
    div_factor=10.0,
    final_div_factor=100.0
)

# Data Augmentation
transforms.Compose([
    RandomRotation((-7.0, 7.0)),
    RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        shear=(-5, 5)
    ),
    ColorJitter(brightness=0.2, contrast=0.2),
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])
```

## Device Support
- CPU
- CUDA (NVIDIA GPU)
- MPS (Apple Silicon)

## License
MIT

## Acknowledgments
- PyTorch
- torchvision
- MNIST dataset
