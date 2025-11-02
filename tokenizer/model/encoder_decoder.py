"""
High-level overview:
    Given (patch_tokens, mask), how does the model encode visible patches, 
    compress them temporally, and then reconstruct the missing ones?
encoder_decoder.py:
    Defines how the tokenizer as a whole works (high-level pipeline). 
    Uses modules defined in transformer_blocks.py. 
Conceptual Overview:
    1. Input (masked patches)
    2. Stack of spatial + temporal blocks (encoder)
    3. tanh bottleneck
    4. Stack of spatial + temporal blocks (decoder)
    5. Linear projection 
    6. Output (reconstructed patches)

Imports — bring in BlockCausalTransformer, torch.nn as nn.
Define class CausalTokenizer(nn.Module)
In __init__:
    Create mask_token (nn.Parameter)
    Build encoder stack (ModuleList)
    Build decoder stack (ModuleList)
    Add latent projection layers (Linear + Tanh)
    Add output projection (Linear)
In forward():
    Replace masked tokens.
    Pass through encoder → latent → decoder.
    Return reconstructed patches (and optionally latents).
"""
import torch
import torch.nn as nn
from tokenizer.model.transformer_blocks import BlockCausalTransformer