# Training script v0 (most simple with basic flow-matching)
import torch
from torch.utils.data import DataLoader
from tokenizer.model.encoder_decoder import CausalTokenizer
from world_model.wm_preprocessing.wm_dataset import WorldModelDataset
from world_model.wm_preprocessing.data_builder_wm import DataBuilderWM
from world_model.wm.world_model import WorldModel
from world_model.losses_wm import flow_loss_v1   
import wandb

class TokenizerConfig:
    ckpt_path = Path("checkpoints/overfit/latest_complete_overfit_mse/best_model.pt")

    # Model / dataset params (must match training)
    resize = (256, 448)
    patch_size = 16
    clip_length = 8
    input_dim = 3 * patch_size * patch_size
    embed_dim = 512
    latent_dim = 256
    num_heads = 16
    num_layers = 20
    device = "cuda" if torch.cuda.is_available() else "cpu"

def load_tokenizer(cfg):
    print(f"[Load] Loading checkpoint: {cfg.ckpt_path}")
    model = CausalTokenizer(
        input_dim=cfg.input_dim,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        latent_dim=cfg.latent_dim,
        use_checkpoint=False,
    )

    ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
    state = ckpt["model_state"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)

    model.to(cfg.device)
    model.eval()
    print("✓ Model loaded")
    return model

@torch.no_grad()
def visualize_world_model(
    world_model, 
    data_builder, 
    sample, 
    tokenizer, 
    step, 
    device, 
    num_frames_to_show=4
):
    """
    Visualize Dreamer-4 world model predictions:
      - predicted frames
      - ground truth frames 
    """
    world_model.eval()

    latents = sample["latents"]       # (T,N,D_latent)
    actions = sample["actions"]

    # Build WM inputs
    wm_input = data_builder(latents, actions)
    pred_z = world_model(wm_input)    # (T,N_latents,D_latent)
    z_clean = wm_input["z_clean"]     # (T,N,D_latent)

    # Select frames to visualize
    T = pred_z.shape[0]
    indices = torch.linspace(0, T-1, num_frames_to_show).long()

    rows = []
    for t in indices:
        # Decode predicted latent → recon image
        pred_img = tokenizer.decode(pred_z[t:t+1])
        gt_img   = tokenizer.decode(z_clean[t:t+1])

        pred_np = (pred_img[0].permute(1,2,0).cpu().numpy() * 255).astype("uint8")
        gt_np   = (gt_img[0].permute(1,2,0).cpu().numpy() * 255).astype("uint8")

        # Combine GT and Pred side by side
        combined = np.concatenate([gt_np, pred_np], axis=1)
        rows.append(combined)

    final_img = np.concatenate(rows, axis=0)

    wandb.log({"wm_reconstruction": wandb.Image(final_img, caption=f"step {step}")})

    world_model.train()

def train_overfit():
    cfg = TokenizerConfig()

    wandb.init(
        project="worldmodel-v0",
        config={
            "d_model": d_model,
            "d_latent": d_latent,
            "num_layers": 16,
            "num_heads": 8,
            "lr": 1e-3
        }
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Dataset
    dataset = WorldModelDataset(
        latent_dir="data/latent_sequences",
        action_jsonl="data/actions.jsonl",
        clip_length=8,
        device=device
    )

    # shuffle=True helps overfitting faster
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    d_model = 640
    d_latent = 256
    data_builder = DataBuilderWM(d_model).to(device)
    world_model = WorldModel(
        d_model=d_model,
        d_latent=d_latent,
        num_layers=16,
        num_heads=8
    ).to(device)

    optimizer = torch.optim.AdamW(world_model.parameters(), lr=1e-3)

    # Training Loop
    world_model.train()

    for step in range(5000):   
        for sample in loader:

            latents = sample["latents"]       
            actions = sample["actions"]

            # Build world-model input
            wm_input = data_builder(latents, actions)

            # Forward pass
            pred_z = world_model(wm_input)

            # Flow-matching loss (MSE)
            loss = flow_loss_v1(
                pred_z,       
                wm_input["z_clean"],
                wm_input["tau"]
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if step % 50 == 0:
            print(f"[step {step}] loss = {loss.item():.6f}")
        
        wandb.log({"flow_loss": loss.item(), "step": step})

    print("Completed Training")

if __name__ == "__main__":
    train_overfit()
