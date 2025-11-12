"""
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python testing_script/debug_data_flow.py

Comprehensive data flow debugging script.
Traces every transformation from video file to loss computation.
"""

import torch
import numpy as np
from pathlib import Path
from tokenizer.tokenizer_dataset import TokenizerDatasetDDP
from tokenizer.model.encoder_decoder import CausalTokenizer2
from tokenizer.losses import CombinedLoss, MSELoss
from tokenizer.patchify_mask import Patchifier
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================
class DebugConfig:
    data_dir = Path("data")
    target_video = "cheeky-cornflower-setter-0a5ba522405b-20220422-133010.mp4"
    resize = (256, 448)
    patch_size = 16
    clip_length = 8
    embed_dim = 512
    latent_dim = 384
    num_heads = 8
    num_layers = 18

cfg = DebugConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Helper Functions
# ============================================================================

def print_tensor_stats(name, tensor, expected_range=None):
    """Print detailed statistics about a tensor."""
    print(f"\n{'='*80}")
    print(f"[{name}]")
    print(f"{'='*80}")
    print(f"  Shape:        {tensor.shape}")
    print(f"  Dtype:        {tensor.dtype}")
    print(f"  Device:       {tensor.device}")
    print(f"  Min:          {tensor.min().item():.6f}")
    print(f"  Max:          {tensor.max().item():.6f}")
    print(f"  Mean:         {tensor.mean().item():.6f}")
    print(f"  Std:          {tensor.std().item():.6f}")
    print(f"  Has NaN:      {torch.isnan(tensor).any().item()}")
    print(f"  Has Inf:      {torch.isinf(tensor).any().item()}")
    
    if expected_range:
        lo, hi = expected_range
        in_range = (tensor >= lo).all() and (tensor <= hi).all()
        status = "✓" if in_range else "❌"
        print(f"  Expected:     [{lo}, {hi}]")
        print(f"  In Range:     {status}")
        
        if not in_range:
            out_of_range_count = ((tensor < lo) | (tensor > hi)).sum().item()
            print(f"  Out of range: {out_of_range_count}/{tensor.numel()} elements")
    
    # Distribution info
    if tensor.numel() > 0:
        percentiles = torch.quantile(tensor.flatten().float(), 
                                     torch.tensor([0.01, 0.25, 0.5, 0.75, 0.99]).to(tensor.device))
        print(f"  Percentiles:")
        print(f"    1%:  {percentiles[0].item():.6f}")
        print(f"    25%: {percentiles[1].item():.6f}")
        print(f"    50%: {percentiles[2].item():.6f}")
        print(f"    75%: {percentiles[3].item():.6f}")
        print(f"    99%: {percentiles[4].item():.6f}")

def visualize_tensor_distribution(tensors_dict, save_path="debug_distributions.png"):
    """Visualize distributions of multiple tensors."""
    fig, axes = plt.subplots(2, len(tensors_dict), figsize=(5*len(tensors_dict), 10))
    
    for idx, (name, tensor) in enumerate(tensors_dict.items()):
        data = tensor.detach().cpu().flatten().numpy()
        
        # Histogram
        axes[0, idx].hist(data, bins=50, alpha=0.7, edgecolor='black')
        axes[0, idx].set_title(f"{name}\nHistogram")
        axes[0, idx].set_xlabel("Value")
        axes[0, idx].set_ylabel("Count")
        axes[0, idx].grid(True, alpha=0.3)
        
        # Box plot
        axes[1, idx].boxplot(data)
        axes[1, idx].set_title(f"{name}\nBox Plot")
        axes[1, idx].set_ylabel("Value")
        axes[1, idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n[Visualization] Saved to {save_path}")
    plt.close()

# ============================================================================
# STAGE 1: Dataset Loading
# ============================================================================

print("\n" + "="*80)
print("STAGE 1: DATASET LOADING")
print("="*80)

dataset = TokenizerDatasetDDP(
    video_dir=cfg.data_dir,
    resize=cfg.resize,
    clip_length=cfg.clip_length,
    patch_size=cfg.patch_size,
    mask_prob_range=(0.0, 0.0),
    per_frame_mask_sampling=True,
    mode="random",
)

# Find target video
target_idx = None
for idx, sample_info in enumerate(dataset.samples):
    if sample_info["video_path"].name == cfg.target_video:
        target_idx = idx
        break

if target_idx is None:
    raise RuntimeError(f"Video {cfg.target_video} not found!")

print(f"\n✓ Found target video at index {target_idx}")

# Load one sample
sample = dataset[target_idx]
patches_raw = sample["patch_tokens"]
mask = sample["mask"]
meta = sample["meta"]

print_tensor_stats("Raw Patches from Dataset", patches_raw, expected_range=(0, 1))
print_tensor_stats("Mask from Dataset", mask.float())

print(f"\nMetadata:")
for key, val in meta.items():
    if not isinstance(val, (list, tuple)):
        print(f"  {key}: {val}")

# ============================================================================
# STAGE 2: Verify Patchify/Unpatchify Invertibility
# ============================================================================

print("\n" + "="*80)
print("STAGE 2: PATCHIFY/UNPATCHIFY INVERTIBILITY TEST")
print("="*80)

patchifier = Patchifier(patch_size=cfg.patch_size)

# Test on first frame
test_patches = patches_raw[0:1]  # (1, N, D)
print_tensor_stats("Test Patches (1 frame)", test_patches, expected_range=(0, 1))

# Unpatchify
test_frame = patchifier.unpatchify(
    test_patches,
    frame_size=cfg.resize,
    patch_size=cfg.patch_size
)  # (1, C, H, W)

print_tensor_stats("Unpatchified Frame", test_frame, expected_range=(0, 1))

# Re-patchify
test_patches_roundtrip = patchifier(test_frame)  # (1, N, D)
print_tensor_stats("Re-patchified", test_patches_roundtrip, expected_range=(0, 1))

# Check invertibility
roundtrip_error = (test_patches - test_patches_roundtrip).abs()
print_tensor_stats("Roundtrip Error", roundtrip_error)

max_error = roundtrip_error.max().item()
if max_error < 1e-5:
    print("\n✓ PASS: Patchify/unpatchify is invertible")
else:
    print(f"\n❌ FAIL: Roundtrip error too large: {max_error:.10f}")
    print("  First 10 values of original:   ", test_patches[0, 0, :10].cpu().numpy())
    print("  First 10 values of roundtrip:  ", test_patches_roundtrip[0, 0, :10].cpu().numpy())

# ============================================================================
# STAGE 3: Model Input Preparation
# ============================================================================

print("\n" + "="*80)
print("STAGE 3: MODEL INPUT PREPARATION")
print("="*80)

# Add batch dimension
patches_input = patches_raw.unsqueeze(0).to(device)  # (1, T, N, D)
mask_input = mask.unsqueeze(0).to(device)  # (1, T, N)

print_tensor_stats("Model Input Patches", patches_input, expected_range=(0, 1))
print_tensor_stats("Model Input Mask", mask_input.float())

# ============================================================================
# STAGE 4: Model Forward Pass (Step-by-Step)
# ============================================================================

print("\n" + "="*80)
print("STAGE 4: MODEL FORWARD PASS (DETAILED)")
print("="*80)

# Initialize model
model = CausalTokenizer2(
    input_dim=3 * cfg.patch_size * cfg.patch_size,
    embed_dim=cfg.embed_dim,
    num_heads=cfg.num_heads,
    num_layers=cfg.num_layers,
    latent_dim=cfg.latent_dim,
    use_checkpoint=False,
).to(device)

model.eval()

with torch.no_grad():
    B, T, N, D_in = patches_input.shape
    
    # Step 1: Input projection
    print("\n--- Step 1: Input Projection ---")
    x = model.input_proj(patches_input)
    print_tensor_stats("After input_proj", x)
    
    # Step 2: Mask token replacement
    print("\n--- Step 2: Mask Token Replacement ---")
    print_tensor_stats("Mask Token (learnable)", model.mask_token)
    
    mask_exp = mask_input.unsqueeze(-1).expand(-1, -1, -1, model.embed_dim)
    x_masked = torch.where(mask_exp, model.mask_token.view(1, 1, 1, -1), x)
    print_tensor_stats("After masking", x_masked)
    
    num_masked = mask_input.sum().item()
    print(f"  Number of masked patches: {num_masked}/{T*N} ({num_masked/(T*N)*100:.1f}%)")
    
    # Step 3: Flatten
    print("\n--- Step 3: Flatten Spatial-Temporal ---")
    x_flat = x_masked.view(B, T * N, model.embed_dim)
    print_tensor_stats("After flattening", x_flat)
    
    # Step 4: Positional encoding
    print("\n--- Step 4: Positional Encoding ---")
    seq_len = T * N
    print_tensor_stats("Positional Embeddings (slice)", model.pos_embed[:, :seq_len, :])
    
    x_pos = x_flat + model.pos_embed[:, :seq_len, :]
    print_tensor_stats("After adding positional encoding", x_pos)
    
    # Step 5: Encoder
    print("\n--- Step 5: Encoder Stack ---")
    x_encoded = x_pos
    for i, layer in enumerate(model.encoder):
        x_before = x_encoded
        x_encoded = layer(x_encoded, T, N)
        
        if i % 4 == 0:  # Print every 4 layers
            print(f"\n  Layer {i}:")
            print_tensor_stats(f"    Output", x_encoded)
            delta = (x_encoded - x_before).abs().mean().item()
            print(f"    Change from input: {delta:.6f}")
    
    print("\n  Final encoder output:")
    print_tensor_stats("After encoder", x_encoded)
    
    # Step 6: Bottleneck
    print("\n--- Step 6: Bottleneck ---")
    x_bottleneck = x_encoded.view(B, T, N, model.embed_dim)
    
    x_latent = model.to_latent(x_bottleneck)
    print_tensor_stats("Latent representation", x_latent)
    
    x_from_latent = model.from_latent(x_latent)
    print_tensor_stats("After from_latent", x_from_latent)
    
    # Check information loss in bottleneck
    bottleneck_error = (x_bottleneck - x_from_latent).abs().mean().item()
    print(f"\n  Bottleneck reconstruction error: {bottleneck_error:.6f}")
    if bottleneck_error > 0.5:
        print("  ⚠️  WARNING: Large information loss in bottleneck!")
    
    # Step 7: Decoder
    print("\n--- Step 7: Decoder Stack ---")
    x_decoder = x_from_latent.view(B, T * N, model.embed_dim)
    
    for i, layer in enumerate(model.decoder):
        x_before = x_decoder
        x_decoder = layer(x_decoder, T, N)
        
        if i % 4 == 0:
            print(f"\n  Layer {i}:")
            print_tensor_stats(f"    Output", x_decoder)
            delta = (x_decoder - x_before).abs().mean().item()
            print(f"    Change from input: {delta:.6f}")
    
    print("\n  Final decoder output:")
    print_tensor_stats("After decoder", x_decoder)
    
    # Step 8: Output projection
    print("\n--- Step 8: Output Projection ---")
    x_output = x_decoder.view(B, T, N, model.embed_dim)
    reconstructed_tokens = model.output_proj(x_output)
    
    print_tensor_stats("After output_proj (BEFORE activation)", reconstructed_tokens)
    
    # Check if sigmoid exists
    if hasattr(model, 'output_proj') and isinstance(model.output_proj, torch.nn.Sequential):
        has_sigmoid = any(isinstance(m, torch.nn.Sigmoid) for m in model.output_proj)
        print(f"\n  Sigmoid in output_proj: {has_sigmoid}")
    
    # Apply sigmoid if not already in model
    if reconstructed_tokens.min() < -0.1 or reconstructed_tokens.max() > 1.1:
        print("\n  ⚠️  Output not in [0,1] range - applying sigmoid manually")
        reconstructed_tokens = torch.sigmoid(reconstructed_tokens)
        print_tensor_stats("After manual sigmoid", reconstructed_tokens, expected_range=(0, 1))
    
    # Full forward pass using model
    print("\n--- Full Forward Pass (using model()) ---")
    recon_full = model(patches_input, mask_input)
    print_tensor_stats("Model output (full forward)", recon_full, expected_range=(0, 1))

# ============================================================================
# STAGE 5: Loss Computation
# ============================================================================

print("\n" + "="*80)
print("STAGE 5: LOSS COMPUTATION")
print("="*80)

# MSE Loss
print("\n--- MSE Loss ---")
mse_criterion = MSELoss()
mse_loss = mse_criterion(recon_full, patches_input, None)
print(f"MSE Loss: {mse_loss.item():.6f}")

# Element-wise error
element_error = (recon_full - patches_input).pow(2)
print_tensor_stats("Element-wise squared error", element_error)

# Combined Loss (if using LPIPS)
print("\n--- Combined Loss (MSE + LPIPS) ---")
combined_criterion = CombinedLoss(
    lpips_weight=0.2,
    lpips_net='alex',
    patch_size=cfg.patch_size,
    frame_size=cfg.resize
).to(device)

print("\n  Unpatchifying for LPIPS...")

# Manual unpatchify (what visualization uses)
print("\n  Method 1: Manual unpatchify (visualization)")
frames_manual = []
for t in range(T):
    frame = patchifier.unpatchify(
        recon_full[0, t:t+1],
        frame_size=cfg.resize,
        patch_size=cfg.patch_size
    )[0]
    frames_manual.append(frame)
frames_manual = torch.stack(frames_manual).unsqueeze(0)  # (1, T, C, H, W)
print_tensor_stats("Manual unpatchify result", frames_manual, expected_range=(0, 1))

# Loss unpatchify (what LPIPS uses)
print("\n  Method 2: Loss unpatchify (LPIPS)")
frames_loss = combined_criterion.unpatchify_batch(recon_full)
print_tensor_stats("Loss unpatchify result", frames_loss, expected_range=(0, 1))

# Compare methods
diff = (frames_manual - frames_loss).abs()
print_tensor_stats("Difference between unpatchify methods", diff)

max_diff = diff.max().item()
if max_diff > 1e-5:
    print(f"\n❌ CRITICAL: Unpatchify methods differ by {max_diff:.10f}")
    print("   LPIPS is seeing different images than your visualizations!")
    
    # Show sample pixels
    print("\n  Sample comparison (frame 0, channel 0, top-left 5x5):")
    print("  Manual unpatchify:")
    print(frames_manual[0, 0, 0, :5, :5].cpu().numpy())
    print("\n  Loss unpatchify:")
    print(frames_loss[0, 0, 0, :5, :5].cpu().numpy())
else:
    print(f"\n✓ PASS: Unpatchify methods consistent (diff={max_diff:.10f})")

# Compute combined loss
print("\n  Computing combined loss...")
combined_loss, loss_dict = combined_criterion(recon_full, patches_input, None)

print(f"\n  Loss breakdown:")
print(f"    MSE:   {loss_dict['mse']:.6f}")
print(f"    LPIPS: {loss_dict['lpips']:.6f}")
print(f"    Total: {loss_dict['total']:.6f}")

if loss_dict['lpips'] > 0.5:
    print("\n  ⚠️  WARNING: LPIPS is very high (>0.5)")
    print("     This suggests perceptual quality is poor")

# LPIPS on identical images (should be ~0)
print("\n  LPIPS sanity check (self-comparison)...")
frames_target = []
for t in range(T):
    frame = patchifier.unpatchify(
        patches_input[0, t:t+1],
        frame_size=cfg.resize,
        patch_size=cfg.patch_size
    )[0]
    frames_target.append(frame)
frames_target = torch.stack(frames_target).unsqueeze(0)

# Normalize to [-1, 1] for LPIPS
frames_target_norm = (frames_target * 2 - 1).clamp(-1, 1).reshape(T, 3, *cfg.resize)
lpips_self = combined_criterion.lpips_fn(frames_target_norm, frames_target_norm).mean().item()

print(f"  LPIPS(target, target) = {lpips_self:.6f}")
if lpips_self > 0.01:
    print("  ❌ FAIL: LPIPS self-comparison should be ~0")
else:
    print("  ✓ PASS: LPIPS self-comparison near zero")

# ============================================================================
# STAGE 6: Gradient Flow Check
# ============================================================================

print("\n" + "="*80)
print("STAGE 6: GRADIENT FLOW CHECK")
print("="*80)

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Forward pass
patches_grad = patches_input.clone().detach()
recon_grad = model(patches_grad, mask_input)
loss_grad = mse_criterion(recon_grad, patches_grad, None)

print(f"\nLoss before backward: {loss_grad.item():.6f}")

# Backward pass
optimizer.zero_grad()
loss_grad.backward()

# Check gradients
print("\nGradient statistics:")
grad_stats = {}
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        grad_stats[name] = grad_norm
        
        if 'output_proj' in name or 'input_proj' in name or 'to_latent' in name:
            print(f"  {name:50s}: {grad_norm:.6f}")

# Check for dead gradients
dead_grads = [name for name, norm in grad_stats.items() if norm < 1e-7]
if dead_grads:
    print(f"\n❌ WARNING: {len(dead_grads)} parameters have near-zero gradients:")
    for name in dead_grads[:5]:
        print(f"    {name}")
else:
    print("\n✓ All parameters have non-zero gradients")

# Check gradient magnitudes
all_grads = list(grad_stats.values())
avg_grad = np.mean(all_grads)
max_grad = np.max(all_grads)
min_grad = np.min(all_grads)

print(f"\nGradient magnitude summary:")
print(f"  Min:  {min_grad:.6f}")
print(f"  Mean: {avg_grad:.6f}")
print(f"  Max:  {max_grad:.6f}")

if max_grad > 100:
    print("  ⚠️  WARNING: Very large gradients detected (possible explosion)")
if avg_grad < 1e-5:
    print("  ⚠️  WARNING: Very small gradients detected (possible vanishing)")

# ============================================================================
# STAGE 7: Visualization
# ============================================================================

print("\n" + "="*80)
print("STAGE 7: CREATING VISUALIZATIONS")
print("="*80)

# Collect all tensors for distribution plots
tensors_to_plot = {
    "Input": patches_input,
    "After Encoder": x_encoded,
    "Latent": x_latent,
    "After Decoder": x_decoder,
    "Output": recon_full,
}

visualize_tensor_distribution(tensors_to_plot, "debug_distributions.png")

# ============================================================================
# STAGE 8: Summary
# ============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

issues_found = []

# Check all conditions
if roundtrip_error.max().item() > 1e-5:
    issues_found.append("Patchify/unpatchify not invertible")

if recon_full.min() < 0 or recon_full.max() > 1:
    issues_found.append("Model output not in [0,1] range")

if max_diff > 1e-5:
    issues_found.append("Unpatchify methods inconsistent (LPIPS broken)")

if loss_dict['lpips'] > 0.5:
    issues_found.append("LPIPS very high (>0.5)")

if lpips_self > 0.01:
    issues_found.append("LPIPS self-comparison not near zero")

if dead_grads:
    issues_found.append(f"{len(dead_grads)} parameters with dead gradients")

if max_grad > 100:
    issues_found.append("Gradient explosion detected")

if avg_grad < 1e-5:
    issues_found.append("Gradient vanishing detected")

if bottleneck_error > 0.5:
    issues_found.append("Large information loss in bottleneck")

print(f"\nTotal issues found: {len(issues_found)}")

if issues_found:
    print("\n❌ CRITICAL ISSUES:")
    for i, issue in enumerate(issues_found, 1):
        print(f"  {i}. {issue}")
else:
    print("\n✓ All checks passed! Data flow appears correct.")

print("\n" + "="*80)
print("Debug complete. Check debug_distributions.png for visualizations.")
print("="*80)