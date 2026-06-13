import torch
import torch.nn as nn
import torch.nn.functional as F

# 多头潜在注意力
class MLA(nn.Module):
    def __init__(self,hidden,qd,kvd,heads,dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden//heads
        self.scale = self.head_dim ** 0.5

        self.dq = nn.Linear(hidden, qd)
        self.uq = nn.Linear(qd, hidden)

        self.dkv = nn.Linear(hidden, kvd)
        self.uk = nn.Linear(kvd, hidden)
        self.uv = nn.Linear(kvd, hidden)

        self.w_o = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        b, l, _ = x.shape

        #潜在向量
        dq = self.dq(x)
        dkv = self.dkv(x)

        #up
        q = self.uq(dq).view(b,l,self.heads,self.head_dim).transpose(1,2)
        k = self.uk(dkv).view(b,l,self.heads,self.head_dim).transpose(1,2)
        v = self.uv(dkv).view(b,l,self.heads,self.head_dim).transpose(1,2)

        score = torch.matmul(q, k.transpose(-1,-2)) / self.scale
        attn = F.softmax(score,dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1,2).contiguous().view(b,l,self.hidden)
        out = self.w_o(out)
        return out
