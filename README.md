# Dreamer4 Implementation
Custom implementation of "Training Agents Inside of Scalable World Models" originally published by Google DeepMind.
Trained on 8x 16GB V100 GPUs using data from zhwang4ai/OpenAI-Minecraft-Contractor.

## Key Differences from original paper:
- Uses MSE loss only when training tokenizer (no dynamic LPIPS integration as of yet)
- Tokenizer/World Model/Imagination Training etc... all overfit on one video to prove the pipeline works on limited compute resources.

## Results
### Tokenizer Performance
Example #1: Reconstruction from Tokenizer
<video src="/inference/results/tokenizer/reconstructed_output.mp4" controls="true" width="600"></video>

Example #2: Reconstruction from Tokenizer
<video src="/inference/results/tokenizer/v2_reconstructed_output.mp4" controls="true" width="600"></video>

### World Model Performance