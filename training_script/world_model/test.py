# python training_script/world_model/test.py
import torch
import numpy as np
import wandb
from pathlib import Path
import torch.nn as nn

# ----------------------------------------------------------
# IMPORT YOUR COMPONENTS
# ----------------------------------------------------------
from world_model.wm_preprocessing.wm_dataset import WorldModelDataset
from world_model.wm_preprocessing.wm_databuilder import DataBuilderWM
from world_model.wm.dynamics_model import WorldModel
from tokenizer.model.encoder_decoder import CausalTokenizer
from tokenizer.patchify_mask import Patchifier


# ============================
# CONFIG — EDIT THESE AS NEEDED
# ============================
TOKENIZER_CKPT_PATH = "checkpoints/tokenizer/complete_overfit_mse/v1_weights.pt"
PATCH_SIZE = 16
RESIZE = (256, 448)   # same as tokenizer training
D_MODEL = 512
D_LATENT = 256
NUM_LAYERS = 16
NUM_HEADS = 8


# ----------------------------------------------------------
# LOAD TOKENIZER (ENCODER-DECODER) + DECODE FUNCTION
# ----------------------------------------------------------
def load_tokenizer(device="cuda"):
    model = CausalTokenizer(
        input_dim=3 * PATCH_SIZE * PATCH_SIZE,
        embed_dim=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=18,
        latent_dim=D_LATENT,
        use_checkpoint=False,
    )

    ckpt = torch.load(TOKENIZER_CKPT_PATH, map_location="cpu")
    state = {k.replace("module.", ""): v for k, v in ckpt["model_state"].items()}
    model.load_state_dict(state)
    model.to(device).eval()

    return model


@torch.no_grad()
def decode_latents(tokenizer, latents):
    """
    latents: (1, N_latents, D_latent)
    returns (1, 3, H, W)
    """
    # tokenizer expects: (B, T*N, D) flattened OR patch tokens.
    # But we need to "unproject" from latent → embed → decoder blocks.
    # CausalTokenizer has no decode_latents() API, so we manually rebuild input.

    # 1. Project latents back to embed_dim
    x = tokenizer.from_latent(latents)     # (1, N, embed_dim)

    # 2. Add pos embedding
    seq_len = x.shape[1]
    x = x + tokenizer.pos_embed[:, :seq_len, :]

    # 3. Flatten for decoder
    x = x.view(1, seq_len, D_MODEL)

    # 4. Run decoder
    x = tokenizer._run_stack(x, tokenizer.decoder, T=1, N=seq_len)

    # 5. Unproject to patch tokens
    patches = tokenizer.output_proj(x).view(1, seq_len, -1)

    # 6. Unpatchify → image
    patchifier = Patchifier(PATCH_SIZE)
    H, W = RESIZE
    frame = patchifier.unpatchify(patches, (H, W), PATCH_SIZE)[0]  # (3,H,W)

    return frame.unsqueeze(0)


# ----------------------------------------------------------
# VISUALIZATION
# ----------------------------------------------------------
@torch.no_grad()
def visualize_world_model(world_model, data_builder, sample, tokenizer, step, device, num_frames=4):

    world_model.eval()

    latents = sample["latents"]      # (T,N_latent,256)
    actions = sample["actions"]

    # Build WM input
    wm_input = data_builder(latents, actions)
    pred_z = world_model(wm_input)        # (T,N_latent,256)
    z_clean = wm_input["z_clean"]         # (T,N,256)

    # Select indices
    T = pred_z.shape[0]
    idxs = np.linspace(0, T - 1, num_frames, dtype=int)

    rows = []
    for t in idxs:
        pred = decode_latents(tokenizer, pred_z[t:t+1])
        gt   = decode_latents(tokenizer, z_clean[t:t+1])

        pred_np = (pred[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        gt_np   = (gt[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

        combined = np.concatenate([gt_np, pred_np], axis=1)   # GT | PRED
        rows.append(combined)

    final_img = np.concatenate(rows, axis=0)

    wandb.log({"wm_reconstruction": wandb.Image(final_img, caption=f"Step {step}")})

    world_model.train()


# ----------------------------------------------------------
# MAIN TEST SCRIPT
# ----------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Start wandb
    wandb.init(project="dreamer4-wm-visual-test")

    # Load Dataset
    dataset = WorldModelDataset(
        latent_dir="data/latent_sequences",
        action_jsonl="data/actions.jsonl",
        clip_length=8,
        device=device
    )
    sample = dataset[0]   # single clip for visualization

    # Components
    data_builder = DataBuilderWM(D_MODEL).to(device)
    world_model = WorldModel(D_MODEL, D_LATENT, NUM_LAYERS, NUM_HEADS).to(device)
    tokenizer = load_tokenizer(device)

    print("Testing visualization...")
    visualize_world_model(world_model, data_builder, sample, tokenizer, step=0, device=device)
    print("Visualization complete. Check wandb dashboard.")


if __name__ == "__main__":
    main()
