"""
High-level overview:
    Given (patch_tokens, mask), how does the model encode visible patches, 
    compress them temporally, and then reconstruct the missing ones?
encoder_decoder.py:
    Defines how the tokenizer as a whole works (high-level pipeline). 
    Uses modules defined in transformer_blocks.py. 
"""
