# Code for Dynamics Model (the backbone of World Model)
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from world_model.wm.transformer_blocks_wm import BlockCausalTransformer

class WorldModel(nn.Module):
    """
    Predict the clean latent tokens for each frame, given:
        - Corrupted latents
        - Actions
        - Register tokens
        - Shortcut token (τ, d)
        - Causal history (previous timesteps)
    """
    def __init__(
        self,
        d_model: int,
        d_latent: int,
        num_layers: int,
        num_heads: int,
    ):
        super().__init__()
        self.pos_embed = nn.Parameter(...)    
        self.output_head = nn.Linear(D_model → D_latent)

        transformer_blocks = []
        for i in range(num_layers):
            causal_time = (i % 4 == 3)
            transformer_blocks.append(BlockCausalTransformer(embed_dim, num_heads, causal_time))
        self.transformer_blocks = nn.ModuleList(transformer_blocks)
        
    def forward(wm_input_tokens, T, N_total):
        x = wm_input_tokens + pos_embed     
        for block in self.transformer_blocks:
            x = block(x, T, N_total)
        pred_z_clean = self.output_head(x_latent_positions)
        return pred_z_clean
