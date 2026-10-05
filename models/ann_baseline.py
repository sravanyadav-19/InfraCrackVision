import torch.nn as nn
class ANNClassifier(nn.Module):
    def __init__(self,n_features=9,n_classes=4): super().__init__(); self.net=nn.Sequential(nn.Linear(n_features,32),nn.ReLU(),nn.Dropout(.1),nn.Linear(32,n_classes))
    def forward(self,x): return self.net(x)
