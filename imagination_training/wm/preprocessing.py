import json
import numpy as np
from PIL import Image
import torch
import os

def process_vpt_actions(jsonl_path):
    """
    Convert VPT action format to Matrix-Game-2.0 format.
    VPT uses mouse (dx, dy, buttons) and keyboard (keys).
    """
    actions = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # Extract mouse and keyboard inputs
            mouse = data['mouse']
            keyboard = data['keyboard']
            
            # Matrix-Game-2.0 expects action embeddings
            # You'll need to create action vectors from mouse/keyboard
            action = {
                'mouse_dx': mouse['dx'],
                'mouse_dy': mouse['dy'],
                'mouse_buttons': mouse['buttons'],
                'keyboard_keys': keyboard['keys'],
                'yaw': data['yaw'],
                'pitch': data['pitch'],
            }
            actions.append(action)
    
    return actions

def load_video_frames(frame_dir, num_frames=None):
    """Load video frames from directory."""
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(('.png', '.jpg'))])
    
    if num_frames:
        frame_files = frame_files[:num_frames]
    
    frames = []
    for f in frame_files:
        img = Image.open(os.path.join(frame_dir, f))
        frames.append(img)
    
    return frames

def create_action_sequence(actions_list):
    """
    Convert VPT actions to action tensors for Matrix-Game-2.0.
    
    Input: List of action dictionaries from process_vpt_actions()
    Output: Tensor of shape [num_frames, ACTION_DIM]
    
    Matrix-Game-2.0 expects action vectors that encode:
    - Mouse movement (dx, dy)
    - Keyboard state (W, A, S, D, Space, etc.)
    - Camera rotation (yaw, pitch changes)
    """
    action_tensors = []
    
    # Define action space dimensions
    # Adjust based on Matrix-Game-2.0's actual action space
    ACTION_DIM = 16  # Example: might need adjustment
    
    for action_data in actions_list:
        # Create action vector
        action_vec = np.zeros(ACTION_DIM, dtype=np.float32)
        
        # Encode mouse movement (normalized to [-1, 1])
        action_vec[0] = np.clip(action_data['mouse_dx'] / 100.0, -1, 1)
        action_vec[1] = np.clip(action_data['mouse_dy'] / 100.0, -1, 1)
        
        # Encode keyboard (binary flags for common Minecraft keys)
        keys = action_data['keyboard_keys']
        action_vec[2] = 1.0 if 'key.keyboard.w' in keys else 0.0  # Forward
        action_vec[3] = 1.0 if 'key.keyboard.a' in keys else 0.0  # Left
        action_vec[4] = 1.0 if 'key.keyboard.s' in keys else 0.0  # Backward
        action_vec[5] = 1.0 if 'key.keyboard.d' in keys else 0.0  # Right
        action_vec[6] = 1.0 if 'key.keyboard.space' in keys else 0.0  # Jump
        action_vec[7] = 1.0 if 'key.keyboard.left.shift' in keys else 0.0  # Sneak
        action_vec[8] = 1.0 if 'key.keyboard.left.control' in keys else 0.0  # Sprint
        
        # Mouse buttons
        action_vec[9] = 1.0 if 0 in action_data['mouse_buttons'] else 0.0  # Left click
        action_vec[10] = 1.0 if 1 in action_data['mouse_buttons'] else 0.0  # Right click
        
        # Camera angles (normalized)
        action_vec[11] = np.clip(action_data['yaw'] / 180.0, -1, 1)
        action_vec[12] = np.clip(action_data['pitch'] / 90.0, -1, 1)
        
        # Reserved for future use
        action_vec[13] = 0.0
        action_vec[14] = 0.0
        action_vec[15] = 0.0
        
        action_tensors.append(torch.from_numpy(action_vec))
    
    return torch.stack(action_tensors)

def preprocess_vpt_for_matrix_game(data_dir, output_dir=None, num_frames=None):
    """
    Complete preprocessing pipeline for VPT data to Matrix-Game-2.0 format.
    
    Args:
        data_dir: Directory containing frames and actions.jsonl
        output_dir: Directory to save preprocessed data (defaults to data_dir)
        num_frames: Optional limit on number of frames to process
    
    Returns:
        dict with paths to conditioning_image and action_sequence
    """
    if output_dir is None:
        output_dir = data_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading frames from {data_dir}...")
    frames = load_video_frames(data_dir, num_frames=num_frames)
    print(f"Loaded {len(frames)} frames")
    
    print(f"Processing actions from {os.path.join(data_dir, 'actions.jsonl')}...")
    actions_list = process_vpt_actions(os.path.join(data_dir, 'actions.jsonl'))
    print(f"Processed {len(actions_list)} action entries")
    
    # Save first frame as conditioning image
    conditioning_image_path = os.path.join(output_dir, 'conditioning_image.png')
    first_frame = frames[0]
    first_frame.save(conditioning_image_path)
    print(f"Saved conditioning image to {conditioning_image_path}")
    
    # Create action sequence tensor
    print("Creating action sequence tensor...")
    action_sequence = create_action_sequence(actions_list)
    action_sequence_path = os.path.join(output_dir, 'action_sequence.pt')
    torch.save(action_sequence, action_sequence_path)
    print(f"Saved action sequence to {action_sequence_path}")
    print(f"Action sequence shape: {action_sequence.shape}")
    
    # Print some statistics
    print("\n=== Action Statistics ===")
    print(f"Total frames: {len(action_sequence)}")
    print(f"Action dimension: {action_sequence.shape[1]}")
    print(f"Mouse movement range: dx=[{action_sequence[:, 0].min():.3f}, {action_sequence[:, 0].max():.3f}], "
          f"dy=[{action_sequence[:, 1].min():.3f}, {action_sequence[:, 1].max():.3f}]")
    
    # Count key presses
    w_presses = action_sequence[:, 2].sum().item()
    space_presses = action_sequence[:, 6].sum().item()
    print(f"'W' key pressed in {w_presses}/{len(action_sequence)} frames ({100*w_presses/len(action_sequence):.1f}%)")
    print(f"'Space' key pressed in {space_presses}/{len(action_sequence)} frames ({100*space_presses/len(action_sequence):.1f}%)")
    
    return {
        'conditioning_image': conditioning_image_path,
        'action_sequence': action_sequence_path,
        'num_frames': len(frames),
        'action_shape': action_sequence.shape
    }


if __name__ == "__main__":
    # Process your VPT data
    result = preprocess_vpt_for_matrix_game(
        data_dir='testing_clips',
        output_dir='testing_clips',  # Optional: specify different output directory
        num_frames=None  # Optional: limit number of frames
    )
    
    print("\n=== Preprocessing Complete ===")
    print(f"Conditioning image: {result['conditioning_image']}")
    print(f"Action sequence: {result['action_sequence']}")
    print(f"Total frames: {result['num_frames']}")
    print(f"Action tensor shape: {result['action_shape']}")
    
    print("\nReady for inference! You can now use:")
    print("  - conditioning_image.png as the initial frame")
    print("  - action_sequence.pt as the action conditioning")