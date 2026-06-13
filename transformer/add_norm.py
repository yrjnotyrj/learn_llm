import torch
import torch.nn as nn
import torch.nn.functional as F

class LN(nn.Module):
    def __init__(self, hidden,eps=1e-5):
        super().__init__()
        self.hidden = hidden
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden))
        self.beta = nn.Parameter(torch.zeros(hidden))

    def forward(self, x):
        # b, l, h = x.shape
        mu = x.mean(dim=-1, keepdims=True)
        sigma = (x - mu).pow(2).mean(dim=-1, keepdims=True)
        x_norm = (x-mu)/torch.sqrt((sigma+self.eps))
        out = self.gamma * x_norm + self.beta
        return out

class AddNorm(nn.Module):
    #原transformer 采取post-ln
    def __init__(self,dim,eps=1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(dim,eps=eps)

    def forward(self, x, sublayer):
        residual = x + sublayer(x)
        out = self.ln(residual)
        return out

class AddNorm1(nn.Module):
    #pre norm 主流版本
    def __init__(self,dim,eps=1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=eps)

    def forward(self,x,sublayer):
        x_norm = self.ln(x)
        subout = sublayer(x_norm)
        return subout + x




