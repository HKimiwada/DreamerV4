import os
import json
import torch
from PIL import Image
import torchvision.transforms as T
from models import MatrixGameDistillPipeline

# --- 1. CONFIGURATION & MAPPING ---
# Matrix-Game 2.0 Minecraft Mapping (Standard 16Hz sync)
KEYBOARD_MAPPING = {
    "w": 0, "s": 1, "a": 2, "d": 3, 
    "space": 4, "shift": 5, "attack": 6, "use": 7
}

class VPTActionMapper:
    def __init__(self, keyboard_dim=12, mouse_dim=2):
        self.keyboard_dim = keyboard_dim
        self.mouse_dim = mouse_dim

    def parse_line(self, json_line):
        data = json.loads(json_line)
        
        # Parse Keyboard (Multi-hot)
        kb_tensor = torch.zeros(self.keyboard_dim)
        keys_pressed = [k.lower() for k in data.get("keyboard", {}).get("keys", [])]
        for key, idx in KEYBOARD_MAPPING.items():
            if key in keys_pressed:
                kb_tensor[idx] = 1.0
        
        # Parse Mouse (Continuous deltas)
        # Note: VPT provides 'dx' and 'dy'. Matrix-Game expects these as camera pitch/yaw deltas.
        mouse_dx = data.get("mouse", {}).get("dx", 0.0)
        mouse_dy = data.get("mouse", {}).get("dy", 0.0)
        mouse_tensor = torch.tensor([mouse_dx, mouse_dy], dtype=torch.float32)
        
        return kb_tensor, mouse_tensor

# --- 2. INITIALIZE PIPELINE ---
print("Loading Matrix-Game 2.0 (1.8B Distilled)...")
pipeline = MatrixGameDistillPipeline.from_pretrained(
    model_path="./wm_weights/base_distilled_model",
    vae_path="./wm_weights/Wan2.1_VAE.pth",
    text_encoder_path="./wm_weights/xlm-roberta-large",
    clip_path="./wm_weights/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    device="cuda"
)

# --- 3. PREPARE DATA ---
# Load starting frame
# Ensure your frame naming (000001.png) matches your local file
first_frame_path = "testing_clips/000001.png"
first_frame = Image.open(first_frame_path).convert("RGB")
transform = T.Compose([
    T.Resize((480, 832)), # Matrix-Game 2.0 native resolution
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
init_image = transform(first_frame).unsqueeze(0).cuda()

# Load and Map Actions
mapper = VPTActionMapper()
kb_actions = []
mouse_actions = []

with open("testing_clips/actions.jsonl", "r") as f:
    lines = f.readlines()
    for line in lines[:32]: # Test on first 32 frames
        kb, mouse = mapper.parse_line(line)
        kb_actions.append(kb)
        mouse_actions.append(mouse)

# Stack into [Batch, Time, Dimension] tensors
keyboard_tensor = torch.stack(kb_actions).unsqueeze(0).cuda()
mouse_tensor = torch.stack(mouse_actions).unsqueeze(0).cuda()

# --- 4. RUN INFERENCE ---
print(f"Generating imagination for {len(lines[:32])} frames...")
with torch.no_grad():
    # Matrix-Game 2.0 uses 'keyboard_actions' and 'mouse_actions' specifically
    generated_video = pipeline.generate_long_video(
        init_image=init_image,
        keyboard_actions=keyboard_tensor,
        mouse_actions=mouse_tensor,
        num_frames=len(kb_actions),
        num_inference_steps=4,  # Distilled version only needs 4-8 steps
        guidance_scale=5.0,
        fps=25                  # Matrix-Game 2.0 is tuned for 25 FPS
    )

# --- 5. SAVE RESULTS ---
# The pipeline usually returns a list of PIL images or a video object
output_path = "imagination_training/wm/output_results/reconstruction_test.mp4"
generated_video.save(output_path)
print(f"Success! Video saved to {output_path}")