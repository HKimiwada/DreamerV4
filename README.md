# Dreamer4 Implementation
Custom implementation of "Training Agents Inside of Scalable World Models" originally published by Google DeepMind.
Trained on 8x 16GB V100 GPUs using data from zhwang4ai/OpenAI-Minecraft-Contractor.

## Key Differences from original paper:
- Uses MSE loss only when training tokenizer (no dynamic LPIPS integration as of yet)
- Tokenizer/World Model/Imagination Training etc... all overfit on one video to prove the pipeline works on limited compute resources.

## Results
### Tokenizer Performance
Example #1: Reconstruction from Tokenizer \n
https://github.com/user-attachments/assets/1bb7deb3-aaa0-4f28-abaf-b1ccf875d183.mp4

Example #2: Reconstruction from Tokenizer \n
https://github.com/user-attachments/assets/2a936c75-4fa1-49be-9aa1-11f7a0f476af.mp4

### World Model Performance
