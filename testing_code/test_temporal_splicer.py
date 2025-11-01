# python testing_code/test_temporal_splicer.py
from pathlib import Path
from tokenizer.dataset import VideoLoader
from tokenizer.temporal_slicer import TemporalSlicer

video_dir = Path("data")

# Stage A loader
loader = VideoLoader(video_dir, target_fps=20.0, resize=(384, 640))

# Stage B slicer
slicer = TemporalSlicer(loader, clip_length=64, stride=64, mode="sequential")

count = 0
for sample in slicer.generate_all():
    clip = sample["frames"]
    meta = sample["metadata"]
    print(f"{meta['video_id']}  start={meta['start']}  end={meta['end']}  clip={clip.shape}")
    count += 1
    if count >= 5:
        break
print(f"Generated {count} example clips.")
