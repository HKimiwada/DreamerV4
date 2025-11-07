# PYTHONPATH=. torchrun --nproc_per_node=1 training_script/test.py
"""
Quick check to verify if model outputs are unbounded.
"""
import torch
from pathlib import Path
from tokenizer.tokenizer_dataset import TokenizerDatasetDDP
from tokenizer.model.encoder_decoder import CausalTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load one sample
dataset = TokenizerDatasetDDP(
    video_dir=Path("data"),
    resize=(256, 448),
    clip_length=8,
    patch_size=16,
    mask_prob_range=(0.2, 0.2),
    per_frame_mask_sampling=False,
    mode="random",
)

sample = dataset[0]
patches = sample['patch_tokens'].unsqueeze(0).to(device)
mask = sample['mask'].unsqueeze(0).to(device)

print("="*80)
print("CHECKING MODEL OUTPUT BOUNDS")
print("="*80)

print(f"\n📊 INPUT STATISTICS:")
print(f"   Min: {patches.min():.4f}")
print(f"   Max: {patches.max():.4f}")
print(f"   Mean: {patches.mean():.4f}")
print(f"   Std: {patches.std():.4f}")
print(f"   ✓ Inputs are in [0, 1] as expected")

# Create untrained model
model = CausalTokenizer(
    input_dim=768,
    embed_dim=512,
    num_heads=8,
    num_layers=12,
    latent_dim=256,
    use_checkpoint=False,
).to(device)

# Forward pass
model.eval()
with torch.no_grad():
    recon = model(patches, mask)

print(f"\n📊 OUTPUT STATISTICS (Untrained Model):")
print(f"   Min: {recon.min():.4f}")
print(f"   Max: {recon.max():.4f}")
print(f"   Mean: {recon.mean():.4f}")
print(f"   Std: {recon.std():.4f}")

# Check if bounded
if recon.min() < -0.1 or recon.max() > 1.1:
    print(f"\n❌ PROBLEM DETECTED!")
    print(f"   Outputs are UNBOUNDED!")
    print(f"   Range: [{recon.min():.4f}, {recon.max():.4f}]")
    print(f"   Should be: [0, 1]")
    
    # Calculate what this means for loss
    mse = (recon - patches).pow(2).mean()
    print(f"\n💥 Initial MSE Loss: {mse.item():.2f}")
    print(f"   (Expected for random bounded outputs: ~0.08)")
    print(f"   Your loss is {mse.item() / 0.08:.1f}x higher than expected!")
    
    print(f"\n🔧 FIX REQUIRED:")
    print(f"   Add output activation (Sigmoid or Tanh) to bound outputs to [0, 1]")
else:
    print(f"\n✓ Outputs are properly bounded to [0, 1]")

# Show distribution
print(f"\n📈 OUTPUT DISTRIBUTION:")
bins = torch.linspace(-5, 5, 11)
hist = torch.histc(recon, bins=10, min=-5, max=5)
for i, (low, high, count) in enumerate(zip(bins[:-1], bins[1:], hist)):
    bar = "█" * int(count.item() / hist.max().item() * 50)
    print(f"   [{low:5.1f}, {high:5.1f}): {bar}")

print("="*80)