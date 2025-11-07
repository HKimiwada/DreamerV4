# python tokenizer/losses.py
# tokenizer/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips

class MSELoss(nn.Module):
    """
    Pixel-wise Mean Squared Error loss, supports optional mask.
    """
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            recon: Tensor of shape (B, C, H, W) or (B, T, N, D) for patches
            target: Tensor of same shape
            mask: Optional tensor broadcastable to input shape
        Returns:
            scalar Tensor (loss)
        """
        if mask is not None:
            # ensure mask broadcast shape
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B,1,H,W) -> (B,1,H,W)
            # compute per-pixel squared error
            loss = (recon - target).pow(2)
            loss = loss * mask
            denom = mask.sum().clamp(min=1.0)
            return loss.sum() / denom
        else:
            return F.mse_loss(recon, target)


class CombinedLoss(nn.Module):
    """
    Combined MSE + LPIPS loss as used in Dreamer4.
    
    Loss = MSE + λ * LPIPS
    
    Supports both patch-based and image-based inputs.
    For patches, unpatchifies them internally before computing LPIPS.
    
    Args:
        lpips_weight: Weight for LPIPS loss (default 0.1 as in paper)
        lpips_net: Network for LPIPS ('alex', 'vgg', or 'squeeze')
        patch_size: Size of patches (required if using patch inputs)
        frame_size: (H, W) frame dimensions (required if using patch inputs)
    """
    def __init__(self, lpips_weight=0.1, lpips_net='alex', patch_size=None, frame_size=None):
        super().__init__()
        self.mse_loss = MSELoss()
        # LPIPS expects input in [-1, 1] range
        self.lpips_fn = lpips.LPIPS(net=lpips_net).eval()
        self.lpips_weight = lpips_weight
        
        # For unpatchifying
        self.patch_size = patch_size
        self.frame_size = frame_size
        if patch_size is not None:
            from tokenizer.patchify_mask import Patchifier
            self.patchifier = Patchifier(patch_size=patch_size)
        
        # Freeze LPIPS network
        for param in self.lpips_fn.parameters():
            param.requires_grad = False
    
    def unpatchify_batch(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Convert patches to images.
        Args:
            patches: (B, T, N, D) where D = C * patch_size * patch_size
        Returns:
            images: (B, T, C, H, W)
        """
        B, T, N, D = patches.shape
        
        # Reshape to (B*T, N, D)
        patches_flat = patches.reshape(B * T, N, D)
        
        # Unpatchify each frame
        frames = self.patchifier.unpatchify(
            patches_flat,
            frame_size=self.frame_size,
            patch_size=self.patch_size
        )  # (B*T, C, H, W)
        
        # Reshape back to (B, T, C, H, W)
        C, H, W = frames.shape[1:]
        images = frames.reshape(B, T, C, H, W)
        
        return images
    
    def forward(self, recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        """
        Args:
            recon: (B, T, N, D) patches in [0, 1] range
            target: (B, T, N, D) patches in [0, 1] range
            mask: Optional (B, T, N) mask (ignored for LPIPS, used for MSE)
        
        Returns:
            tuple: (total_loss, loss_dict)
                - total_loss: scalar tensor
                - loss_dict: dict with 'mse', 'lpips', 'total' values
        """
        # Compute MSE loss on patches (with mask support)
        mse_loss = self.mse_loss(recon, target, mask)
        
        # Convert patches to images for LPIPS
        if self.patch_size is not None:
            recon_images = self.unpatchify_batch(recon)  # (B, T, C, H, W)
            target_images = self.unpatchify_batch(target)  # (B, T, C, H, W)
        else:
            # Already in image format
            recon_images = recon
            target_images = target
        
        # Reshape from (B, T, C, H, W) to (B*T, C, H, W) for LPIPS
        B, T, C, H, W = recon_images.shape
        recon_flat = recon_images.reshape(B * T, C, H, W)
        target_flat = target_images.reshape(B * T, C, H, W)
        
        # Normalize to [-1, 1] for LPIPS
        recon_normalized = recon_flat * 2 - 1
        target_normalized = target_flat * 2 - 1
        
        # Clamp to ensure valid range
        recon_normalized = recon_normalized.clamp(-1, 1)
        target_normalized = target_normalized.clamp(-1, 1)
        
        # Compute LPIPS
        lpips_values = self.lpips_fn(recon_normalized, target_normalized)  # (B*T, 1, 1, 1)
        lpips_value = lpips_values.mean()
        
        # Combined loss
        total_loss = mse_loss + self.lpips_weight * lpips_value
        
        return total_loss, {
            'mse': mse_loss.item(),
            'lpips': lpips_value.item(),
            'total': total_loss.item()
        }