# Data Loader to prepare data for input into WM.
# Input: latent tokens, numerically-converted actions (from wm_dataset)
# Output: (latent tokens, tokenized actions, register tokens, short cut token) for each timestep
import torch
from torch.nn as nn
from world_model.wm_preprocessing.wm_dataset import WorldModelDataset
from world_model.wm_preprocessing.action_tokenier import ActionTokenizer
 
class DataLoaderWM(nn.Module):
    """
    Generates sequence to feed into WM model.
    """
    def __init__(
        self,
        d_model: int,
        latent_dir: str = "data/latent_sequences",
        action_jsonl: str = "data/actions.jsonl",
        register_tokens: int = 8, # small learned vectors that help the transformer aggregate information. (Like a scratch pad of sorts)
        short_cut_token: int = 1, # contains: τ = noise level of that frame, d = step size (coarse/fine rollout) for short-cut forcing.
    )
        self.d_model = d_model
        self.latent_dir = latent_dir
        self.action_jsonl = action_jsonl
        self.register_tokens = register_tokens
        self.short_cut_token = short_cut_token

        self.register_embed = nn.Embedding(register_tokens, d_model)


    def forward(self):
        pass
