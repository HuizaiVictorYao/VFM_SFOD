import torch.nn as nn
import torch
import torch.nn.init as init

class MLP(nn.Module):
    def __init__(self, feat_dim:int = 2048*3*3):
        super(MLP, self).__init__()
        self.feat_dim = feat_dim
        self.fc1 = nn.Linear(self.feat_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, self.feat_dim)
        self.bn2 = nn.BatchNorm1d(self.feat_dim)
        self.relu2 = nn.ReLU()

        self._initialize_weights()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x
    
    def _initialize_weights(self):
        init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        # if self.fc1.bias is not None:
        #     init.constant_(self.fc1.bias, 0)
        # if self.fc2.bias is not None:
        #     init.constant_(self.fc2.bias, 0)