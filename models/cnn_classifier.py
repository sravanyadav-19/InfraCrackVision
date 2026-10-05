"""Primary four-class image classifier. Training orchestration is in training/train_cnn.py."""
import torch.nn as nn
class InfraCrackCNN(nn.Module):
    def __init__(self,n_classes=4):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1)); self.head=nn.Linear(64,n_classes)
    def forward(self,x): return self.head(self.net(x).flatten(1))
