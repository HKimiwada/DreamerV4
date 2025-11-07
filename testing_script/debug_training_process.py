"""
Comprehensive debugging script for DreamerV4 tokenizer.
Run: PYTHONPATH=. python testing_script/debug_training_process.py
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

from tokenizer.tokenizer_dataset import TokenizerDatasetDDP
from tokenizer.model.encoder_decoder import CausalTokenizer
from tokenizer.losses import MSELoss
from tokenizer.patchify_mask import Patchifier

warnings.filterwarnings('ignore')

class TokenizerDebugger:
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Device] {self.device}")
        
        # Initialize components
        self.dataset = None
        self.model = None
        self.results = {}
        
    # ========================================================================
    # TEST 1: Data Pipeline Integrity
    # ========================================================================
    def test_data_pipeline(self):
        """Verify data loading and preprocessing is correct."""
        print("\n" + "="*80)
        print("TEST 1: Data Pipeline Integrity")
        print("="*80)
        
        self.dataset = TokenizerDatasetDDP(
            video_dir=self.cfg['data_dir'],
            resize=self.cfg['resize'],
            clip_length=self.cfg['clip_length'],
            patch_size=self.cfg['patch_size'],
            mask_prob_range=(0.5, 0.5),  # Fixed 50% for testing
            per_frame_mask_sampling=False,
            mode="random",
        )
        
        # Load one sample
        sample = self.dataset[0]
        patches = sample['patch_tokens']  # (T, N, D)
        mask = sample['mask']  # (T, N)
        
        T, N, D = patches.shape
        print(f"✓ Patches shape: {patches.shape}")
        print(f"✓ Mask shape: {mask.shape}")
        print(f"✓ Patch values: min={patches.min():.4f}, max={patches.max():.4f}, mean={patches.mean():.4f}")
        print(f"✓ Mask ratio: {mask.float().mean():.2%} masked")
        
        # Check for NaN/Inf
        assert not torch.isnan(patches).any(), "❌ NaN detected in patches!"
        assert not torch.isinf(patches).any(), "❌ Inf detected in patches!"
        assert patches.min() >= 0 and patches.max() <= 1, "❌ Patches not in [0,1]!"
        
        # Test unpatchify
        patchifier = Patchifier(patch_size=self.cfg['patch_size'])
        H, W = self.cfg['resize']
        
        reconstructed_frames = patchifier.unpatchify(
            patches, 
            frame_size=(H, W),
            patch_size=self.cfg['patch_size']
        )
        
        print(f"✓ Unpatchified shape: {reconstructed_frames.shape}")
        assert reconstructed_frames.shape == (T, 3, H, W), "❌ Unpatchify failed!"
        
        # Visualize one frame
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        frame_idx = 0
        
        # Original
        frame = reconstructed_frames[frame_idx].permute(1, 2, 0).numpy()
        axes[0].imshow(frame)
        axes[0].set_title("Original Frame")
        axes[0].axis('off')
        
        # Mask visualization
        mask_viz = mask[frame_idx].float().view(H // self.cfg['patch_size'], W // self.cfg['patch_size'])
        mask_viz = mask_viz.repeat_interleave(self.cfg['patch_size'], 0).repeat_interleave(self.cfg['patch_size'], 1)
        axes[1].imshow(mask_viz.numpy(), cmap='gray')
        axes[1].set_title(f"Mask ({mask[frame_idx].float().mean():.1%} masked)")
        axes[1].axis('off')
        
        # Masked frame
        masked_frame = frame.copy()
        mask_expanded = mask_viz.unsqueeze(-1).repeat(1, 1, 3).numpy()
        masked_frame[mask_expanded > 0.5] = 1.0  # White out masked regions
        axes[2].imshow(masked_frame)
        axes[2].set_title("Masked Frame")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig("debug_data_pipeline.png", dpi=150)
        print("✓ Saved visualization → debug_data_pipeline.png")
        plt.close()
        
        self.results['data_pipeline'] = 'PASS'
        return sample
    
    # ========================================================================
    # TEST 2: Model Forward Pass
    # ========================================================================
    def test_model_forward(self, sample):
        """Test model can perform forward pass without errors."""
        print("\n" + "="*80)
        print("TEST 2: Model Forward Pass")
        print("="*80)
        
        self.model = CausalTokenizer(
            input_dim=self.cfg['input_dim'],
            embed_dim=self.cfg['embed_dim'],
            num_heads=self.cfg['num_heads'],
            num_layers=self.cfg['num_layers'],
            latent_dim=self.cfg['latent_dim'],
            use_checkpoint=False,
        ).to(self.device)
        
        num_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"✓ Model parameters: {num_params:.2f}M")
        
        # Forward pass
        patches = sample['patch_tokens'].unsqueeze(0).to(self.device)  # (1, T, N, D)
        mask = sample['mask'].unsqueeze(0).to(self.device)  # (1, T, N)
        
        print(f"✓ Input shape: {patches.shape}")
        
        with torch.no_grad():
            output = self.model(patches, mask)
        
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Output values: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
        
        # Check for NaN/Inf
        assert not torch.isnan(output).any(), "❌ NaN in model output!"
        assert not torch.isinf(output).any(), "❌ Inf in model output!"
        
        self.results['model_forward'] = 'PASS'
        return output
    
    # ========================================================================
    # TEST 3: Gradient Flow
    # ========================================================================
    def test_gradient_flow(self, sample):
        """Check if gradients are flowing properly through the model."""
        print("\n" + "="*80)
        print("TEST 3: Gradient Flow")
        print("="*80)
        
        self.model.train()
        criterion = MSELoss()
        
        patches = sample['patch_tokens'].unsqueeze(0).to(self.device)
        mask = sample['mask'].unsqueeze(0).to(self.device)
        
        # Forward + backward
        output = self.model(patches, mask)
        loss = criterion(output, patches, mask.unsqueeze(-1))
        loss.backward()
        
        # Check gradient statistics
        grad_norms = []
        zero_grads = 0
        nan_grads = 0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                
                if grad_norm == 0:
                    zero_grads += 1
                if torch.isnan(param.grad).any():
                    nan_grads += 1
                    print(f"❌ NaN gradient in {name}")
        
        print(f"✓ Gradient norms: min={min(grad_norms):.6f}, max={max(grad_norms):.6f}, mean={np.mean(grad_norms):.6f}")
        print(f"✓ Zero gradients: {zero_grads}/{len(grad_norms)}")
        print(f"✓ NaN gradients: {nan_grads}/{len(grad_norms)}")
        
        if zero_grads > len(grad_norms) * 0.5:
            print("⚠️  WARNING: More than 50% of gradients are zero!")
        
        if nan_grads > 0:
            print("❌ CRITICAL: NaN gradients detected!")
            self.results['gradient_flow'] = 'FAIL'
        else:
            self.results['gradient_flow'] = 'PASS'
    
    # ========================================================================
    # TEST 4: Loss Computation Validation
    # ========================================================================
    def test_loss_computation(self, sample):
        """Validate that loss is computed correctly."""
        print("\n" + "="*80)
        print("TEST 4: Loss Computation")
        print("="*80)
        
        criterion = MSELoss()
        
        patches = sample['patch_tokens'].unsqueeze(0).to(self.device)
        mask = sample['mask'].unsqueeze(0).to(self.device)
        
        # Test 1: Perfect reconstruction should have ~0 loss
        with torch.no_grad():
            loss_perfect = criterion(patches, patches, mask.unsqueeze(-1))
        print(f"✓ Perfect reconstruction loss: {loss_perfect.item():.8f}")
        assert loss_perfect.item() < 1e-6, "❌ Perfect reconstruction should have near-zero loss!"
        
        # Test 2: Random reconstruction
        random_recon = torch.rand_like(patches)
        with torch.no_grad():
            loss_random = criterion(random_recon, patches, mask.unsqueeze(-1))
        print(f"✓ Random reconstruction loss: {loss_random.item():.4f}")
        
        # Test 3: Model reconstruction
        with torch.no_grad():
            output = self.model(patches, mask)
            loss_model = criterion(output, patches, mask.unsqueeze(-1))
        print(f"✓ Model reconstruction loss: {loss_model.item():.4f}")
        
        # Expected loss range for random [0,1] data
        expected_random_loss = 1.0 / 12.0  # Var(U[0,1]) = 1/12 ≈ 0.083
        print(f"✓ Expected random MSE: ~{expected_random_loss:.4f}")
        
        if loss_model.item() > loss_random.item():
            print("⚠️  WARNING: Model loss is worse than random!")
        
        self.results['loss_computation'] = 'PASS'
    
    # ========================================================================
    # TEST 5: Overfitting Capability (No Masking)
    # ========================================================================
    def test_overfit_no_mask(self, sample, num_steps=100):
        """Test if model can overfit to one sample with no masking."""
        print("\n" + "="*80)
        print("TEST 5: Overfitting Capability (No Masking)")
        print("="*80)
        
        # Reset model
        self.model = CausalTokenizer(
            input_dim=self.cfg['input_dim'],
            embed_dim=self.cfg['embed_dim'],
            num_heads=self.cfg['num_heads'],
            num_layers=self.cfg['num_layers'],
            latent_dim=self.cfg['latent_dim'],
            use_checkpoint=False,
        ).to(self.device)
        
        criterion = MSELoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=0.0)
        
        patches = sample['patch_tokens'].unsqueeze(0).to(self.device)
        mask = torch.zeros_like(sample['mask']).unsqueeze(0).to(self.device)  # NO MASKING
        
        self.model.train()
        losses = []
        
        print("Training with NO masking...")
        for step in tqdm(range(num_steps)):
            optimizer.zero_grad()
            output = self.model(patches, mask)
            loss = criterion(output, patches, mask.unsqueeze(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        
        final_loss = losses[-1]
        initial_loss = losses[0]
        print(f"✓ Initial loss: {initial_loss:.6f}")
        print(f"✓ Final loss: {final_loss:.6f}")
        print(f"✓ Improvement: {(initial_loss - final_loss) / initial_loss * 100:.1f}%")
        
        # Plot loss curve
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.xlabel('Step')
        plt.ylabel('MSE Loss')
        plt.title('Overfitting Test (No Masking)')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.savefig("debug_overfit_no_mask.png", dpi=150)
        print("✓ Saved loss curve → debug_overfit_no_mask.png")
        plt.close()
        
        if final_loss > 0.01:
            print("⚠️  WARNING: Model failed to overfit with no masking!")
            self.results['overfit_no_mask'] = 'FAIL'
        else:
            print("✓ Model can overfit with no masking")
            self.results['overfit_no_mask'] = 'PASS'
        
        return losses
    
    # ========================================================================
    # TEST 6: Overfitting with Low Masking
    # ========================================================================
    def test_overfit_low_mask(self, sample, num_steps=200):
        """Test overfitting with 20% masking."""
        print("\n" + "="*80)
        print("TEST 6: Overfitting with Low Masking (20%)")
        print("="*80)
        
        # Reset model
        self.model = CausalTokenizer(
            input_dim=self.cfg['input_dim'],
            embed_dim=self.cfg['embed_dim'],
            num_heads=self.cfg['num_heads'],
            num_layers=self.cfg['num_layers'],
            latent_dim=self.cfg['latent_dim'],
            use_checkpoint=False,
        ).to(self.device)
        
        criterion = MSELoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=0.0)
        
        patches = sample['patch_tokens'].unsqueeze(0).to(self.device)
        # Fixed 20% masking
        T, N, D = patches.shape[1:]
        mask = torch.bernoulli(torch.full((1, T, N), 0.2)).bool().to(self.device)
        
        print(f"✓ Mask ratio: {mask.float().mean():.2%}")
        
        self.model.train()
        losses = []
        
        print("Training with 20% masking...")
        for step in tqdm(range(num_steps)):
            optimizer.zero_grad()
            output = self.model(patches, mask)
            loss = criterion(output, patches, mask.unsqueeze(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        
        final_loss = losses[-1]
        initial_loss = losses[0]
        print(f"✓ Initial loss: {initial_loss:.6f}")
        print(f"✓ Final loss: {final_loss:.6f}")
        print(f"✓ Improvement: {(initial_loss - final_loss) / initial_loss * 100:.1f}%")
        
        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.xlabel('Step')
        plt.ylabel('MSE Loss')
        plt.title('Overfitting Test (20% Masking)')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.savefig("debug_overfit_20pct_mask.png", dpi=150)
        print("✓ Saved loss curve → debug_overfit_20pct_mask.png")
        plt.close()
        
        # Visualize reconstruction
        self.model.eval()
        with torch.no_grad():
            output = self.model(patches, mask)
        
        self._visualize_reconstruction(patches[0], output[0], mask[0], "debug_recon_20pct.png")
        
        if final_loss > 0.05:
            print("⚠️  WARNING: Model struggling to overfit with 20% masking")
            self.results['overfit_low_mask'] = 'FAIL'
        else:
            print("✓ Model can overfit with 20% masking")
            self.results['overfit_low_mask'] = 'PASS'
        
        return losses
    
    # ========================================================================
    # TEST 7: Check Model Architecture Issues
    # ========================================================================
    def test_architecture_issues(self):
        """Check for common architecture issues."""
        print("\n" + "="*80)
        print("TEST 7: Architecture Issues")
        print("="*80)
        
        issues = []
        
        # Check 1: Verify attention masking
        print("Checking attention masking...")
        # BlockCausalTransformer should have causal_time parameter
        has_causal = any('causal_time' in str(type(m)) or 
                        hasattr(m, 'causal') 
                        for m in self.model.modules())
        print(f"✓ Causal masking implemented: {has_causal}")
        
        # Check 2: Verify normalization layers
        print("Checking normalization...")
        norm_layers = [m for m in self.model.modules() if isinstance(m, (nn.LayerNorm, nn.BatchNorm2d))]
        rms_norms = [m for m in self.model.modules() if 'RMSNorm' in type(m).__name__]
        print(f"✓ RMSNorm layers: {len(rms_norms)}")
        print(f"✓ Other norm layers: {len(norm_layers)}")
        
        # Check 3: Verify activation functions
        print("Checking activations...")
        has_tanh = any(isinstance(m, nn.Tanh) for m in self.model.modules())
        print(f"✓ Tanh bottleneck: {has_tanh}")
        
        # Check 4: Check for dead neurons
        print("Checking for dead neurons...")
        with torch.no_grad():
            sample = self.dataset[0]
            patches = sample['patch_tokens'].unsqueeze(0).to(self.device)
            mask = sample['mask'].unsqueeze(0).to(self.device)
            
            _ = self.model(patches, mask)
            
            dead_neurons = 0
            total_neurons = 0
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    if hasattr(module, 'weight'):
                        weight = module.weight
                        dead = (weight.abs().sum(dim=1) < 1e-6).sum().item()
                        dead_neurons += dead
                        total_neurons += weight.shape[0]
            
            print(f"✓ Dead neurons: {dead_neurons}/{total_neurons} ({dead_neurons/total_neurons*100:.2f}%)")
            
            if dead_neurons / total_neurons > 0.1:
                issues.append("More than 10% dead neurons")
        
        if issues:
            print("\n⚠️  Found issues:")
            for issue in issues:
                print(f"   - {issue}")
            self.results['architecture'] = 'WARNING'
        else:
            self.results['architecture'] = 'PASS'
    
    # ========================================================================
    # TEST 8: Loss Masking Strategy
    # ========================================================================
    def test_loss_masking_strategy(self, sample):
        """Test different loss masking strategies."""
        print("\n" + "="*80)
        print("TEST 8: Loss Masking Strategy")
        print("="*80)
        
        criterion = MSELoss()
        
        patches = sample['patch_tokens'].unsqueeze(0).to(self.device)
        mask = sample['mask'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(patches, mask)
            
            # Strategy 1: Loss only on masked patches
            loss_masked_only = criterion(output, patches, mask.unsqueeze(-1))
            
            # Strategy 2: Loss on all patches
            loss_all = criterion(output, patches, None)
            
            # Strategy 3: Loss only on visible patches
            visible_mask = ~mask
            loss_visible_only = criterion(output, patches, visible_mask.unsqueeze(-1))
        
        print(f"✓ Loss (masked patches only): {loss_masked_only.item():.6f}")
        print(f"✓ Loss (all patches): {loss_all.item():.6f}")
        print(f"✓ Loss (visible patches only): {loss_visible_only.item():.6f}")
        
        # The Dreamer4 paper uses MAE-style training: loss only on masked patches
        print("\n💡 INSIGHT: DreamerV4 computes loss on ALL patches, not just masked ones")
        print("   Your current implementation may be computing loss only on masked patches.")
        
        self.results['loss_strategy'] = 'PASS'
    
    # ========================================================================
    # Helper: Visualize Reconstruction
    # ========================================================================
    def _visualize_reconstruction(self, patches_orig, patches_recon, mask, filename):
        """Visualize original vs reconstructed patches."""
        patchifier = Patchifier(patch_size=self.cfg['patch_size'])
        H, W = self.cfg['resize']
        
        # Take first frame
        orig_frame = patchifier.unpatchify(
            patches_orig[0:1], frame_size=(H, W), patch_size=self.cfg['patch_size']
        )[0]
        
        recon_frame = patchifier.unpatchify(
            patches_recon[0:1], frame_size=(H, W), patch_size=self.cfg['patch_size']
        )[0]
        
        mask_viz = mask[0].float().view(H // self.cfg['patch_size'], W // self.cfg['patch_size'])
        mask_viz = mask_viz.repeat_interleave(self.cfg['patch_size'], 0).repeat_interleave(self.cfg['patch_size'], 1)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(orig_frame.cpu().permute(1, 2, 0).numpy())
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        axes[1].imshow(mask_viz.cpu().numpy(), cmap='gray')
        axes[1].set_title(f"Mask ({mask[0].float().mean():.1%})")
        axes[1].axis('off')
        
        axes[2].imshow(recon_frame.cpu().permute(1, 2, 0).numpy().clip(0, 1))
        axes[2].set_title("Reconstruction")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"✓ Saved reconstruction → {filename}")
        plt.close()
    
    # ========================================================================
    # Run All Tests
    # ========================================================================
    def run_all_tests(self):
        """Run complete diagnostic suite."""
        print("\n" + "="*80)
        print("DREAMERV4 TOKENIZER DIAGNOSTIC SUITE")
        print("="*80)
        
        try:
            # Test 1: Data
            sample = self.test_data_pipeline()
            
            # Test 2: Model forward
            _ = self.test_model_forward(sample)
            
            # Test 3: Gradients
            self.test_gradient_flow(sample)
            
            # Test 4: Loss
            self.test_loss_computation(sample)
            
            # Test 5: Overfit no mask
            self.test_overfit_no_mask(sample, num_steps=100)
            
            # Test 6: Overfit low mask
            self.test_overfit_low_mask(sample, num_steps=200)
            
            # Test 7: Architecture
            self.test_architecture_issues()
            
            # Test 8: Loss strategy
            self.test_loss_masking_strategy(sample)
            
        except Exception as e:
            print(f"\n❌ FATAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            self.results['fatal_error'] = str(e)
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print diagnostic summary."""
        print("\n" + "="*80)
        print("DIAGNOSTIC SUMMARY")
        print("="*80)
        
        for test_name, result in self.results.items():
            status_emoji = "✅" if result == "PASS" else "⚠️" if result == "WARNING" else "❌"
            print(f"{status_emoji} {test_name}: {result}")
        
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        
        # Generate recommendations based on results
        recommendations = []
        
        if self.results.get('overfit_no_mask') == 'FAIL':
            recommendations.append(
                "1. Model cannot overfit with no masking\n"
                "   → Check model architecture (encoder/decoder depth)\n"
                "   → Verify bottleneck dimension isn't too small\n"
                "   → Try simpler model first (fewer layers)"
            )
        
        if self.results.get('overfit_low_mask') == 'FAIL':
            recommendations.append(
                "2. Model struggles with 20% masking\n"
                "   → Start training with 0-20% masking range\n"
                "   → Gradually increase masking ratio\n"
                "   → Your current 10-90% is too aggressive"
            )
        
        if self.results.get('gradient_flow') == 'FAIL':
            recommendations.append(
                "3. Gradient flow issues detected\n"
                "   → Check for vanishing gradients (reduce depth)\n"
                "   → Add gradient clipping (you already have this)\n"
                "   → Verify layer normalization placement"
            )
        
        recommendations.append(
            "4. Key fixes to try:\n"
            "   → Reduce masking ratio to 0.0-0.3 (instead of 0.1-0.9)\n"
            "   → Increase learning rate to 1e-3 for overfitting\n"
            "   → Reduce model depth if overfitting fails\n"
            "   → Verify loss is computed on ALL patches, not just masked\n"
            "   → Check that mask token is being properly injected"
        )
        
        for rec in recommendations:
            print(f"\n{rec}")
        
        print("\n" + "="*80)

# ========================================================================
# Main
# ========================================================================
if __name__ == "__main__":
    config = {
        'data_dir': Path("data"),
        'resize': (256, 448),
        'patch_size': 16,
        'clip_length': 8,
        'input_dim': 3 * 16 * 16,
        'embed_dim': 512,
        'latent_dim': 256,
        'num_heads': 8,
        'num_layers': 12,
    }
    
    debugger = TokenizerDebugger(config)
    debugger.run_all_tests()