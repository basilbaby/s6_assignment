# MNIST Accuracy Model

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
## Test accuracy log

```bash
(venv) luttappi: model git:(main) python model.py --epochs 20 --target-accuracy 99.4      
Using Apple Silicon GPU (MPS)
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
Failed to download (trying next):
HTTP Error 403: Forbidden

Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz to ../data/MNIST/raw/train-images-idx3-ubyte.gz
100%|█████████████████████████████████████████████████████████████████████| 9.91M/9.91M [00:00<00:00, 28.5MB/s]
Extracting ../data/MNIST/raw/train-images-idx3-ubyte.gz to ../data/MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
Failed to download (trying next):
HTTP Error 403: Forbidden

Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz to ../data/MNIST/raw/train-labels-idx1-ubyte.gz
100%|█████████████████████████████████████████████████████████████████████| 28.9k/28.9k [00:00<00:00, 1.72MB/s]
Extracting ../data/MNIST/raw/train-labels-idx1-ubyte.gz to ../data/MNIST/raw

Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz to ../data/MNIST/raw/t10k-images-idx3-ubyte.gz
100%|█████████████████████████████████████████████████████████████████████| 1.65M/1.65M [00:00<00:00, 13.5MB/s]
Extracting ../data/MNIST/raw/t10k-images-idx3-ubyte.gz to ../data/MNIST/raw

Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz to ../data/MNIST/raw/t10k-labels-idx1-ubyte.gz
100%|█████████████████████████████████████████████████████████████████████| 4.54k/4.54k [00:00<00:00, 1.43MB/s]
Extracting ../data/MNIST/raw/t10k-labels-idx1-ubyte.gz to ../data/MNIST/raw

Total trainable parameters: 17,506

Starting training...

Epoch 1
--------------------
Epoch=1 Loss=0.3010 Batch_id=1874 Accuracy=91.35%: 100%|███████████████████| 1875/1875 [00:33<00:00, 56.54it/s]

Test set: Average loss: 0.1331, Accuracy: 9821/10000 (98.21%)

New best accuracy: 98.21%

Epoch 2
--------------------
Epoch=2 Loss=0.1571 Batch_id=1874 Accuracy=96.25%: 100%|███████████████████| 1875/1875 [00:33<00:00, 55.22it/s]

Test set: Average loss: 0.0721, Accuracy: 9866/10000 (98.66%)

New best accuracy: 98.66%

Epoch 3
--------------------
Epoch=3 Loss=0.1134 Batch_id=1874 Accuracy=97.06%: 100%|███████████████████| 1875/1875 [00:33<00:00, 56.02it/s]

Test set: Average loss: 0.0663, Accuracy: 9862/10000 (98.62%)


Epoch 4
--------------------
Epoch=4 Loss=0.1069 Batch_id=1874 Accuracy=97.30%: 100%|███████████████████| 1875/1875 [00:34<00:00, 54.98it/s]

Test set: Average loss: 0.0375, Accuracy: 9908/10000 (99.08%)

New best accuracy: 99.08%

Epoch 5
--------------------
Epoch=5 Loss=0.1218 Batch_id=1874 Accuracy=97.64%: 100%|███████████████████| 1875/1875 [00:35<00:00, 53.00it/s]

Test set: Average loss: 0.0349, Accuracy: 9907/10000 (99.07%)


Epoch 6
--------------------
Epoch=6 Loss=0.1759 Batch_id=1874 Accuracy=97.77%: 100%|███████████████████| 1875/1875 [00:46<00:00, 40.53it/s]

Test set: Average loss: 0.0310, Accuracy: 9918/10000 (99.18%)

New best accuracy: 99.18%

Epoch 7
--------------------
Epoch=7 Loss=0.1277 Batch_id=1874 Accuracy=97.89%: 100%|███████████████████| 1875/1875 [01:13<00:00, 25.47it/s]

Test set: Average loss: 0.0329, Accuracy: 9912/10000 (99.12%)


Epoch 8
--------------------
Epoch=8 Loss=0.0231 Batch_id=1874 Accuracy=98.02%: 100%|███████████████████| 1875/1875 [00:47<00:00, 39.25it/s]

Test set: Average loss: 0.0323, Accuracy: 9911/10000 (99.11%)


Epoch 9
--------------------
Epoch=9 Loss=0.0259 Batch_id=1874 Accuracy=98.11%: 100%|███████████████████| 1875/1875 [00:38<00:00, 48.47it/s]

Test set: Average loss: 0.0256, Accuracy: 9930/10000 (99.30%)

New best accuracy: 99.30%

Epoch 10
--------------------
Epoch=10 Loss=0.0894 Batch_id=1874 Accuracy=98.15%: 100%|██████████████████| 1875/1875 [00:43<00:00, 43.32it/s]

Test set: Average loss: 0.0267, Accuracy: 9923/10000 (99.23%)


Epoch 11
--------------------
Epoch=11 Loss=0.0229 Batch_id=1874 Accuracy=98.32%: 100%|██████████████████| 1875/1875 [00:46<00:00, 40.35it/s]

Test set: Average loss: 0.0248, Accuracy: 9927/10000 (99.27%)


Epoch 12
--------------------
Epoch=12 Loss=0.0443 Batch_id=1874 Accuracy=98.34%: 100%|██████████████████| 1875/1875 [00:46<00:00, 40.66it/s]

Test set: Average loss: 0.0233, Accuracy: 9934/10000 (99.34%)

New best accuracy: 99.34%

Epoch 13
--------------------
Epoch=13 Loss=0.0336 Batch_id=1874 Accuracy=98.29%: 100%|██████████████████| 1875/1875 [00:49<00:00, 37.74it/s]

Test set: Average loss: 0.0241, Accuracy: 9927/10000 (99.27%)


Epoch 14
--------------------
Epoch=14 Loss=0.0204 Batch_id=1874 Accuracy=98.40%: 100%|██████████████████| 1875/1875 [00:44<00:00, 42.20it/s]

Test set: Average loss: 0.0240, Accuracy: 9931/10000 (99.31%)


Epoch 15
--------------------
Epoch=15 Loss=0.0285 Batch_id=1874 Accuracy=98.45%: 100%|██████████████████| 1875/1875 [00:44<00:00, 42.19it/s]

Test set: Average loss: 0.0238, Accuracy: 9931/10000 (99.31%)


Epoch 16
--------------------
Epoch=16 Loss=0.0942 Batch_id=1874 Accuracy=98.47%: 100%|██████████████████| 1875/1875 [00:44<00:00, 42.09it/s]

Test set: Average loss: 0.0224, Accuracy: 9928/10000 (99.28%)


Epoch 17
--------------------
Epoch=17 Loss=0.0147 Batch_id=1874 Accuracy=98.52%: 100%|██████████████████| 1875/1875 [00:44<00:00, 42.26it/s]

Test set: Average loss: 0.0204, Accuracy: 9944/10000 (99.44%)

New best accuracy: 99.44%

🎉 Target accuracy of 99.4% achieved in epoch 17!

Epoch 18
--------------------
Epoch=18 Loss=0.0283 Batch_id=1874 Accuracy=98.50%: 100%|██████████████████| 1875/1875 [00:45<00:00, 41.45it/s]

Test set: Average loss: 0.0224, Accuracy: 9933/10000 (99.33%)


Epoch 19
--------------------
Epoch=19 Loss=0.1346 Batch_id=1874 Accuracy=98.53%: 100%|██████████████████| 1875/1875 [00:44<00:00, 42.40it/s]

Test set: Average loss: 0.0221, Accuracy: 9939/10000 (99.39%)


Epoch 20
--------------------
Epoch=20 Loss=0.0286 Batch_id=1874 Accuracy=98.48%: 100%|██████████████████| 1875/1875 [00:48<00:00, 38.94it/s]

Test set: Average loss: 0.0197, Accuracy: 9938/10000 (99.38%)


Training completed!
Best test accuracy: 99.44%
✨ Successfully achieved target accuracy of 99.4%!
(venv) luttappi: model git:(main) 
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
