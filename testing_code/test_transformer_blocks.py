# python testing_code/test_transformer_blocks.py
import torch
from tokenizer.model.transformer_blocks import RMSNorm

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
