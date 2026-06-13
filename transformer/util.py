"""
记录一些常用的函数
"""
import torch
import torch.nn.functional as F

def relu(x):
    out = torch.clamp(x,min=0.0)
    return out

def softmax(x):
    # b, l, vocab_size   exp(x-mx)/sum(exp(x-mx))
    mx = x.max(dim=-1,keepdims=True)[0]
    x = x - mx
    x_exp = torch.exp(x)
    out = x_exp/x_exp.sum(dim=-1,keepdims=True)
    return out

def sigmoid(x):
    # 1/(1+exp(-x))
    x = torch.exp(-x) + 1
    out = 1 / x
    return out

def CrossEntropy(logits, label):
    # logits:b,l,vocab_size,  label: b,l
    # -y_*log(yi)
    s = F.softmax(logits, dim=-1)
    log_s = torch.log(s + 1e-10)
    b, l = label.shape
    total = 0
    # 并行计算
    # log_p = log_s.gather(dim=-1, index=label.unsqueeze(-1)).squeeze(-1)
    # loss = -log_p.mean()
    # return loss
    for i in range(b):
        for j in range(l):
            total -= log_s[i][j][label[i][j]]
    return total/(b*l)
