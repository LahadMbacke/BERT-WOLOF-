import torch
from torch import nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self,d_model,hidden_dim,dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear((d_model),hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()


    def forward(self,x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x