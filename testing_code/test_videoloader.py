# python testing_code/test_videoloader.py
from pathlib import Path
from tokenizer.dataset import VideoLoader

video_dir = Path("data")  # Replace with your video directory path
loader = VideoLoader(video_dir=video_dir, target_fps=20.0, resize=(384, 640), max_frames=64)
    
for idx in range(3):
    frames, metadata = loader[idx]
    print(f"Video {idx}:")
    print(f"  Frames shape: {frames.shape}")
    print(f"  Metadata: {metadata}")