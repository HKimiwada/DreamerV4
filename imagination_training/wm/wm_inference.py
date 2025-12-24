# Inference Script for Matrix-Game 2.0 Distilled Model
import torch
from models import MatrixGameDistillPipeline # Use Distill for Matrix-Game 2.0

# 1. Load the Model Components
pipeline = MatrixGameDistillPipeline.from_pretrained(
    model_path="./wm_weights/base_distilled_model",
    vae_path="./wm_weights/Wan2.1_VAE.pth",
    text_encoder_path="./wm_weights/xlm-roberta-large",
    clip_path="./wm_weights/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    device="cuda"
)

# 2. Load your custom start frame from testing_clips
from PIL import Image
import torchvision.transforms as T

first_frame = Image.open("testing_clips/frame_0001.png").convert("RGB")
transform = T.Compose([T.Resize((480, 832)), T.ToTensor()])
init_image = transform(first_frame).unsqueeze(0).cuda()

# 3. Load Action Sequence from actions.jsonl
# You need to parse your JSONL into a tensor of shape [batch, seq_len, action_dim]
actions = load_your_vpt_actions("testing_clips/actions.jsonl") 

# 4. Generate the "Imagination" Rollout
# For Matrix-Game 2.0 Distilled, use few-step (usually 4 steps) for speed
generated_video = pipeline.generate_long_video(
    init_image=init_image,
    actions=actions,
    num_frames=len(actions),
    num_inference_steps=4, 
    guidance_scale=5.0
)

# 5. Save results
generated_video.save("reconstruction_test.mp4")