import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block - slightly increased initial features
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 18, 3, padding=1, bias=False),  # 16->18 channels
            nn.BatchNorm2d(18),
            nn.ReLU(),
            nn.Dropout(0.008)  # Reduced dropout slightly
        )

        # CONV BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(18, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.008)
        )

        # Transition Block with skip connection
        self.transblock1 = nn.Sequential(
            nn.Conv2d(32, 18, 1, bias=False),  # Match first block
            nn.BatchNorm2d(18),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # CONV BLOCK 2 with dilated convolution
        self.convblock3 = nn.Sequential(
            nn.Conv2d(18, 32, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.008)
        )

        # Transition Block 2
        self.transblock2 = nn.Sequential(
            nn.Conv2d(32, 18, 1, bias=False),  # Keep channel consistency
            nn.BatchNorm2d(18),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # CONV BLOCK 3
        self.convblock4 = nn.Sequential(
            nn.Conv2d(18, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.008)
        )

        # OUTPUT BLOCK
        self.output = nn.Sequential(
            nn.Conv2d(32, 10, 1, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Input block with residual
        x1 = self.convblock1(x)
        
        # First block with residual
        x2 = self.convblock2(x1)
        x = self.transblock1(x2)
        x = x + x1[:, :18, ::2, ::2]  # Add residual with size matching
        
        # Second block with dilated conv
        x = self.convblock3(x)
        x = self.transblock2(x)
        
        # Final blocks
        x = self.convblock4(x)
        x = self.output(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        
        return F.log_softmax(x, dim=1)
        
 