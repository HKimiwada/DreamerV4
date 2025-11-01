"""
Data preprocessing pipeline: convert raw VPT gameplay into tensor clips that can be used to train causal tokenizer. 
Overview:
    1. Load raw VPT gameplay data from disk and convert to tensors -> dataset.VideoLoader
    2. Stadardize tensors (resize, normalize, clip into sequences): clip into sequence -> temporal_slicer.TemporalSlicer
    3. Patchify and mask frames for masked-autoencoding training.
    4. Store or Stream batches efficiently for tokenizer.

Classes:
    TemporalSlicer: Loads full-length gameplay tensors (created by VideoLoader) and slices them into fixed-length clips.
        Splits long decoded videos from VideoLoader into uniform, fixed-length clips
        (e.g. 64 or 192 frames from the Dreamer4 paper) suitable for tokenizer training.
    
        mode: sequential (sliding windows) or random (random offset per video)
"""
from pathlib import Path
from typing import Optional, Generator, Dict, Any
import torch

class TemporalSlicer:
    def __init__(
        self,
        video_loader,
        clip_length: int = 64,
        stride: Optional[int] = None,
        mode: str = "sequential",
        drop_last: bool = True,
    ):
        """
        Args:
            video_loader: instance of VideoLoader (Stage A).
            clip_length: number of frames per clip.
            stride: step size between clip starts.
                Defaults to clip_length (non-overlapping). Smaller = overlap.
            mode: 'sequential' → sliding windows,
                  'random' → random starting offsets each call.
            drop_last: drop remainder shorter than clip_length (True) or pad (False).
        """
        self.loader = video_loader
        self.clip_length = clip_length
        self.stride = stride or clip_length
        self.mode = mode
        self.drop_last = drop_last

    # -------------------------------------------------------------

    def slice_video(
        self, frames: torch.Tensor, meta: Dict[str, Any]
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Slice one video tensor (T, 3, H, W) into uniform clips.
        Yields dicts: {"frames": Tensor, "metadata": dict}
        """
        total_frames = frames.shape[0]
        if self.drop_last:
            last_start = total_frames - self.clip_length
        else:
            last_start = total_frames - 1

        # Sequential windows
        if self.mode == "sequential":
            for start in range(0, last_start + 1, self.stride):
                end = min(start + self.clip_length, total_frames)
                clip = frames[start:end]

                # Pad if needed
                if clip.shape[0] < self.clip_length:
                    pad_frames = self.clip_length - clip.shape[0]
                    pad = clip[-1:].repeat(pad_frames, 1, 1, 1)
                    clip = torch.cat([clip, pad], dim=0)

                yield {
                    "frames": clip,
                    "metadata": {
                        "video_id": meta["video_id"],
                        "start": start,
                        "end": end,
                        "orig_frame_count": meta["orig_frame_count"],
                        "clip_length": clip.shape[0],
                    },
                }

        # Random single window
        elif self.mode == "random":
            import random

            if total_frames < self.clip_length:
                if self.drop_last:
                    return  # skip too short
                else:
                    pad = frames[-1:].repeat(self.clip_length - total_frames, 1, 1, 1)
                    frames = torch.cat([frames, pad], dim=0)
                    start = 0
            else:
                start = random.randint(0, total_frames - self.clip_length)
            end = start + self.clip_length
            clip = frames[start:end]
            yield {
                "frames": clip,
                "metadata": {
                    "video_id": meta["video_id"],
                    "start": start,
                    "end": end,
                    "orig_frame_count": meta["orig_frame_count"],
                    "clip_length": clip.shape[0],
                },
            }
        else:
            raise ValueError(f"Unknown slicing mode: {self.mode}")

    # -------------------------------------------------------------

    def generate_all(self) -> Generator[Dict[str, Any], None, None]:
        """Iterate over all videos from the loader and yield sliced clips."""
        for i in range(len(self.loader)):
            frames, meta = self.loader[i]
            yield from self.slice_video(frames, meta)

