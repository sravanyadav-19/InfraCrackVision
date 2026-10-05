"""Four-qubit QNN interface; activate after the four-class metadata is validated."""
import torch.nn as nn
class QNNClassifier(nn.Module):
    def __init__(self,n_features=4,n_classes=4): super().__init__(); self.head=nn.Linear(n_features,n_classes)
    def forward(self,x): return self.head(x)
