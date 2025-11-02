# python tokenizer/model/transformer_blocks.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import math
import copy

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x / torch.sqrt(ms + self.eps)
        output = x_normed * self.scale
        return output

x = torch.randn(2, 5, 4)
norm = RMSNorm(4)
y = norm(x)
print("Testing RMSNorm:")
print(f"{x}: x")
print("x stats:")
print(x.shape)        # should be (2,5,4)
print(x.mean(-1))     # varies, not necessarily 0 (RMSNorm doesn’t center)
print(x.std(-1))      # roughly 1 * scale

print(f"{y}: y")
print("y stats:")
print(y.shape)        # should be (2,5,4)
print(y.mean(-1))     # varies, not necessarily 0 (RMSNorm doesn’t center)
print(y.std(-1))      # roughly 1 * scale
