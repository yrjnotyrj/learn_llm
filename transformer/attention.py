'''
    MHA MQA GQA MLA
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class MHA(nn.Module):
    def __init__(self,hidden, heads, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // self.heads
        self.scale = self.head_dim ** 0.5

        self.w_q = nn.Linear(self.hidden, self.hidden)
        self.w_k = nn.Linear(self.hidden, self.hidden)
        self.w_v = nn.Linear(self.hidden, self.hidden)
        self.w_o = nn.Linear(self.hidden, self.hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, l, h = x.shape

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(b, l, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, l, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, l, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1,-2))/self.scale
        attention = F.softmax(scores,dim=-1)
        attention = self.dropout(attention)

        out = torch.matmul(attention,v)
        out = out.transpose(1,2).contiguous().view(b,l,h)
        out = self.w_o(out)
        return out

class MQA(nn.Module):
    def __init__(self,hidden,heads,dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.head_dim = self.hidden//self.heads
        self.scale = self.head_dim ** 0.5

        self.w_q = nn.Linear(self.hidden, self.hidden)
        self.w_k = nn.Linear(self.hidden, self.head_dim)
        self.w_v = nn.Linear(self.hidden, self.head_dim)
        self.w_o = nn.Linear(self.hidden, self.hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, l, h = x.shape

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(b, l, self.heads, self.head_dim).transpose(1, 2)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

        score = torch.matmul(q, k.transpose(-1,-2))/self.scale
        attn = F.softmax(score, dim = -1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b,l,h)
        out = self.w_o(out)
        return out

class GQA(nn.Module):
    def __init__(self,groups,heads,hidden,dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.groups = groups
        self.heads = heads
        self.head_dim = hidden // heads
        self.scale = self.head_dim ** 0.5
        self.a = self.heads // groups #一个k头对应a个q头

        self.w_q = nn.Linear(hidden, hidden)
        self.w_k = nn.Linear(self.hidden, self.head_dim * groups)
        self.w_v = nn.Linear(self.hidden, self.head_dim * groups)
        self.w_o = nn.Linear(self.hidden, self.head_dim * self.heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        b, l, _ = x.shape

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(b,l,self.heads,self.head_dim).transpose(1,2)
        k = k.view(b,l,self.groups,self.head_dim).transpose(1,2).repeat_interleave(self.a,dim=1)
        v = v.view(b,l,self.groups,self.head_dim).transpose(1,2).repeat_interleave(self.a,dim=1)

        score = torch.matmul(q,k.transpose(-1,-2))/self.scale
        attn = F.softmax(score, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn,v)
        out = out.transpose(1,2).contiguous().view(b,l,self.hidden)
        out = self.w_o(out)
        return out

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


