"""
Data preprocessing pipeline: convert raw VPT gameplay into tensor clips that can be used to train causal tokenizer. 
Overview:
    1. Load raw VPT gameplay data from disk and convert to tensors -> dataset.VideoLoader
    2. Stadardize tensors (resize, normalize, clip into sequences): clip into sequence -> temporal_slicer.TemporalSlicer
    3. Patchify and mask frames for masked-autoencoding training.
    4. Store or Stream batches efficiently for tokenizer.

Classes:
    TemporalSlicer: Loads full-length gameplay tensors (created by VideoLoader) and slices them into fixed-length clips.
"""