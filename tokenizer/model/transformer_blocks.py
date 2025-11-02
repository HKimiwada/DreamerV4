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
