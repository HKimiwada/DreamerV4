# Inference script for World Model (for WM trained via train_world_model_v4.py)
# CUDA_VISIBLE_DEVICES=3 python inference/world_model_inference.py
import torch
import torch.nn as nn
import numpy as np
import cv2
import json
import os
from pathlib import Path
from tqdm import tqdm

# Import your modules
# Ensure these paths are correct relative to where you run this script
from tokenizer.model.encoder_decoder import CausalTokenizer
from tokenizer.patchify_mask import Patchifier
from world_model.wm_preprocessing.wm_databuilder import DataBuilderWM
from world_model.wm.dynamics_model import WorldModel
# Re-used for easy action loading & encoding logic
from world_model.wm_preprocessing.wm_dataset import WorldModelDataset, encode_action_components

# --- Configuration ---
class InferenceConfig:
    # Paths
    tokenizer_ckpt = Path("checkpoints/tokenizer/complete_overfit_mse/v1_weights.pt")
    wm_ckpt_path = Path("checkpoints/world_model/train_world_model_v4/best_model.pt")
    output_dir = Path("inference/results/world_model")
    
    # Data Params
    action_path = "data/actions.jsonl"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model Architecture (Must match training)
    resize = (256, 448)
    patch_size = 16
    input_dim = 3 * 16 * 16
    embed_dim = 512   # Tokenizer dim
    latent_dim = 256  # World Model latent dim
    wm_layers = 12
    wm_heads = 8
    
    # Inference Params
    num_inference_steps = 30  # More steps = higher quality, slower
    clip_length = 60  # How many frames to generate
    guidance_scale = 1.0 # 1.0 = standard, >1.0 = force adherence to condition (if supported)

# --- Helper Functions ---

def load_models(cfg):
    print(f"Loading models on {cfg.device}...")
    
    # 1. Tokenizer
    tokenizer = CausalTokenizer(
        input_dim=cfg.input_dim,
        embed_dim=cfg.embed_dim,
        num_heads=16,
        num_layers=18,
        latent_dim=cfg.latent_dim,
        use_checkpoint=False,
    )
    tok_state = torch.load(cfg.tokenizer_ckpt, map_location="cpu", weights_only=False)
    if "model_state" in tok_state: tok_state = tok_state["model_state"]
    tok_state = {k.replace("module.", ""): v for k, v in tok_state.items()}
    tokenizer.load_state_dict(tok_state)
    tokenizer.to(cfg.device).eval()
    
    # 2. Data Builder & World Model
    data_builder = DataBuilderWM(d_model=512, d_latent=cfg.latent_dim)
    world_model = WorldModel(
        d_model=512,
        d_latent=cfg.latent_dim,
        num_layers=cfg.wm_layers,
        num_heads=cfg.wm_heads,
        n_latents=448,
        clip_length=600, # Max capacity
        use_checkpoint=False 
    )
    
    # Load WM Checkpoint
    if not cfg.wm_ckpt_path.exists():
        raise FileNotFoundError(f"World Model weights not found at {cfg.wm_ckpt_path}")
        
    wm_ckpt = torch.load(cfg.wm_ckpt_path, map_location="cpu", weights_only=False)
    world_model.load_state_dict(wm_ckpt['world_model'])
    data_builder.load_state_dict(wm_ckpt['data_builder'])
    
    world_model.to(cfg.device).eval()
    data_builder.to(cfg.device).eval()
    
    print("✓ All models loaded.")
    return tokenizer, world_model, data_builder

@torch.no_grad()
def decode_latents(tokenizer, latents, size=(256, 448)):
    """Decodes (1, T, N, D) latents into images (T, H, W, 3) numpy array"""
    B, T, N, D = latents.shape
    
    # Decode Logic (Same as training script)
    x = tokenizer.from_latent(latents)
    x = x.view(B, T * N, tokenizer.embed_dim)
    x = tokenizer._run_stack(x, tokenizer.decoder, T=T, N=N)
    x = x.view(B, T, N, tokenizer.embed_dim)
    patches = tokenizer.output_proj(x)
    
    patchifier = Patchifier(16)
    
    images_np = []
    for t in range(T):
        frame = patchifier.unpatchify(patches[0, t:t+1], size, 16)[0]
        frame = frame.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        images_np.append((frame * 255).astype(np.uint8))
        
    return np.array(images_np)

@torch.no_grad()
def generate_video_flow_matching(cfg, world_model, data_builder, tokenizer, actions):
    """
    Generates video using Euler ODE Solver for Flow Matching.
    
    Args:
        actions: Dictionary of action tensors, sliced to (1, T, ...)
    """
    B = 1 # Batch size fixed to 1 for inference
    T = cfg.clip_length
    device = cfg.device
    
    # 1. Initialize random noise (Gaussian)
    # Shape: (B, T, N_latents, D_latent)
    # N_latents = (H//P) * (W//P) = (256//16) * (448//16) = 16 * 28 = 448
    N = 448
    z = torch.randn(B, T, N, cfg.latent_dim, device=device)
    
    # 2. Setup Time Grid (0 to 1)
    # We integrate from t=0 (Noise) to t=1 (Data)
    timesteps = torch.linspace(0, 1, cfg.num_inference_steps + 1, device=device)
    dt = 1.0 / cfg.num_inference_steps
    
    print(f"Generating {T} frames over {cfg.num_inference_steps} steps...")
    
    # 3. ODE Solver Loop (Euler Method)
    for i in tqdm(range(cfg.num_inference_steps)):
        t_curr = timesteps[i]
        
        # Create condition tensors
        tau_map = torch.full((B, T), t_curr.item(), device=device)
        d_map   = torch.full((B,), 0.0, device=device) # d=0 usually means "no corruption" or specific noise level depending on training
        
        # Prepare Inputs via DataBuilder
        # Note: In inference, z_corrupted is our current 'z'. z_clean is unknown, so we pass dummy or z.
        # The model should rely on 'wm_input_tokens' which is built from z_corrupted.
        
        # We need to manually construct the input because data_builder(z, a) usually returns training dicts
        # Let's verify how data_builder constructs tokens:
        z_proj = data_builder.latent_project(z)
        
        # Ensure actions are sliced to match T
        curr_actions = {k: v[:, :T].to(device) for k, v in actions.items()}
        a_tokens = data_builder.action_tokenizer(curr_actions)
        
        # Add Register & Shortcut tokens (Time/Noise embeddings)
        Sr = data_builder.register_tokens
        reg_base = data_builder.register_embed(torch.arange(Sr, device=device))
        register_tokens = reg_base.view(1, 1, Sr, -1).expand(B, T, -1, -1)
        
        # Shortcut (Time embedding)
        d_expanded = d_map.view(B, 1).expand(B, T)
        feat = torch.stack([tau_map, d_expanded], dim=-1)
        shortcut_vec = data_builder.shortcut_mlp(feat) + data_builder.shortcut_slot.view(1, 1, -1)
        shortcut_tokens = shortcut_vec.unsqueeze(2)
        
        # Concatenate Input
        wm_tokens = torch.cat([z_proj, a_tokens, register_tokens, shortcut_tokens], dim=2)
        
        # Flatten for Transformer
        L_total = wm_tokens.shape[2]
        wm_input_tokens = wm_tokens.view(B, T * L_total, -1)
        
        wm_input_dict = {
            "wm_input_tokens": wm_input_tokens,
            "tau": tau_map
            # z_clean/z_corrupted not strictly needed for forward pass inside WorldModel 
            # unless loss is calculated, but let's be safe if architecture peeks at it
        }
        
        # Predict Vector Field (or Target)
        # Model returns pred_z (which is likely the predicted x_1 (clean data) in Rectified Flow)
        pred_z_clean = world_model(wm_input_dict, time_offset=0)
        
        # Vector Field Definition for Rectified Flow: v_t = z_1 - z_t
        # If model predicts z_1 (clean), then v = pred_z_clean - current_z
        v_pred = pred_z_clean - z
        
        # Euler Step: z_{t+1} = z_t + v_t * dt
        z = z + v_pred * dt
        
    # 4. Decode Final Latents
    print("Decoding latents to pixels...")
    video_frames = decode_latents(tokenizer, z)
    
    return video_frames

def save_video(frames, path, fps=30):
    H, W, C = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(path), fourcc, fps, (W, H))
    
    for frame in frames:
        # OpenCV uses BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"Video saved to {path}")

# --- Main ---
def main():
    cfg = InferenceConfig()
    cfg.output_dir.mkdir(exist_ok=True)
    
    # 1. Load Everything
    tokenizer, wm, db = load_models(cfg)
    
    # 2. Load Actions (Use the dataset class logic to handle jsonl)
    # We create a dummy dataset just to parse the actions easily
    print("Loading actions...")
    temp_dataset = WorldModelDataset(
        latent_dir="data/latent_sequences_long", # Path doesn't matter much for action loading
        action_jsonl=cfg.action_path,
        clip_length=cfg.clip_length,
        device="cpu"
    )
    
    # Get the first sequence of actions (since we overfitted on index 0)
    # dataset.actions is expected to be a dict of tensors or list of dicts. 
    # Based on training code, let's assume we want the actions corresponding to the first video.
    # We manually slice the raw actions from the dataset object
    
    # Retrieve first clip actions
    raw_actions = {}
    
    # If dataset.actions is a dict of full tensors (B, Total_T, ...)
    if isinstance(temp_dataset.actions, dict):
        for k, v in temp_dataset.actions.items():
            raw_actions[k] = v[:cfg.clip_length].unsqueeze(0) # Add Batch Dim -> (1, T, ...)
            
    # If it's a list (as handled in training script slicing)
    elif isinstance(temp_dataset.actions, list):
        print(f"Parsing {len(temp_dataset.actions)} action frames (using first {cfg.clip_length})...")
        
        # Take the first T frames
        clip_actions = temp_dataset.actions[:cfg.clip_length]
        
        # Initialize lists for all expected keys
        accumulated = {
            "mouse_cat": [], "scroll": [], "yaw_pitch": [], "hotbar": [], 
            "gui": [], "buttons": [], "keys": []
        }
        
        # Use helper to encode each frame
        for frame in clip_actions:
            enc = encode_action_components(frame)
            for k in accumulated:
                accumulated[k].append(enc[k])
        
        # Stack and add batch dimension
        for k, v in accumulated.items():
            # Stack time dimension -> (T, ...)
            # Add batch dimension -> (1, T, ...)
            if len(v) > 0:
                raw_actions[k] = torch.stack(v).unsqueeze(0)
    
    # fallback if action loading is complex or failed
    if not raw_actions:
        print("⚠️  Warning: Could not parse actions automatically. Using zero-actions.")
        T = cfg.clip_length
        raw_actions = {
            "mouse_cat": torch.zeros(1, T, dtype=torch.long),
            "scroll":    torch.zeros(1, T, dtype=torch.long),
            "yaw_pitch": torch.zeros(1, T, 2, dtype=torch.float32),
            "hotbar":    torch.zeros(1, T, dtype=torch.long),
            "gui":       torch.zeros(1, T, 2, dtype=torch.float32),
            "buttons":   torch.zeros(1, T, 3, dtype=torch.float32),
            "keys":      torch.zeros(1, T, 23, dtype=torch.float32),
        }

    # 3. Run Inference
    video_frames = generate_video_flow_matching(cfg, wm, db, tokenizer, raw_actions)
    
    # 4. Save
    save_path = cfg.output_dir / "inference_result.mp4"
    save_video(video_frames, save_path)

if __name__ == "__main__":
    main()