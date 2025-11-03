"""
Overview of Training script for tokenizer:
    1. Load video patch data from your preprocessed dataset (TokenizerDataset).
    2. Feed it into the tokenizer (encoder–decoder).
    3. Compute the reconstruction loss (MSE + 0.2×LPIPS) only on masked patches.
    4. Backpropagate, optimize, and log training metrics to wandb.

Goal:
A trained tokenizer whose encoder + tanh bottleneck produce stable, 
compact latents for the world model.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from tokenizer.tokenizer_dataset import TokenizerDataset
from tokenizer.model.encoder_decoder import CausalTokenizer  
from tokenizer.losses import CombinedLoss # MSE + Perceptual (LPIPS) loss                      

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

config = {
    # data
    "video_dir": "data/",       # path to preprocessed videos
    "batch_size": 4,
    "num_workers": 4,

    # model
    "input_dim": 768,
    "embed_dim": 768,
    "latent_dim": 256,
    "num_layers": 12,
    "num_heads": 8,

    # training
    "epochs": 100,
    "lr": 1e-4,
    "weight_decay": 0.05,
    "save_dir": "checkpoints/",
    "log_interval": 50,
}

train_dataset = TokenizerDataset(
    video_dir=config["video_dir"],
    clip_length=64,
    mode="random",             # random clips for diversity
    mask_prob_range=(0.0, 0.9),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["num_workers"],
)

model = CausalTokenizer(
    input_dim=config["input_dim"],
    embed_dim=config["embed_dim"],
    latent_dim=config["latent_dim"],
    num_heads=config["num_heads"],
    num_layers=config["num_layers"],
).to(device)

optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
loss_fn = CombinedLoss(alpha=0.2)
