# python testing_code/test_transformer_blocks.py
import torch
from tokenizer.model.transformer_blocks import RMSNorm
from tokenizer.model.transformer_blocks import BlockCausalTransformer

x = torch.randn(2, 5, 4)
norm = RMSNorm(4)
y = norm(x)
print("\nTesting RMSNorm:")
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

print("\nTesting BlockCausalTransformer:")
x = torch.randn(2, 64, 768)
block = BlockCausalTransformer(768, num_heads=8, causal_time=True)
y = block(x)
print(y.shape)   # torch.Size([2, 64, 768])
