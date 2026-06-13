import torch
import torch.nn as nn
import torch.nn.functional as F

class ffn(nn.Module):
    def __init__(self, hidden, ffn_dim, dropout=0.1):
        """标准Transformer FFN层  ffn_dim一般为hidden的四倍"""
        super().__init__()
        self.w1 = nn.Linear(hidden, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))
