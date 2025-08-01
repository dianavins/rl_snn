#!/usr/bin/env python3
"""Create proper Sequential DQN from a trained SB3 model"""

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from collections import OrderedDict
import ale_py

class ProperSequentialDQN(nn.Module):
    """Proper Sequential DQN that exactly matches SB3 architecture"""
    
    def __init__(self, input_channels: int = 4, n_actions: int = 6):
        super().__init__()
        
        # Match SB3 exactly - use regular ReLU, not inplace
        self.network = nn.Sequential(OrderedDict([
            # Convolutional layers (match SB3 NatureCNN)
            ('conv1', nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)),
            ('relu1', nn.ReLU()),  # NOT inplace like SB3
            
            ('conv2', nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            ('relu2', nn.ReLU()),  # NOT inplace like SB3
            
            ('conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            ('relu3', nn.ReLU()),  # NOT inplace like SB3
            
            # Flatten layer
            ('flatten', nn.Flatten(start_dim=1, end_dim=-1)),
            
            # Linear layers
            ('fc1', nn.Linear(3136, 512)),  # 64 * 7 * 7 = 3136
            ('relu4', nn.ReLU()),  # NOT inplace like SB3
            
            ('fc2', nn.Linear(512, n_actions)),
        ]))
        
    def forward(self, x):
        return self.network(x)

def create_sequential_from_sb3():
    """Create and train a proper Sequential DQN"""
    print("=== Creating Proper Sequential DQN from SB3 ===")
    
    # Register ALE environments
    try:
        ale_py.register_all()
    except AttributeError:
        pass
        
    # Create environment 
    try:
        env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=0)
        env = VecFrameStack(env, n_stack=4)
        print("SUCCESS: Environment created")
    except Exception as e:
        print(f"FAILED: Could not create environment: {e}")
        return None, None
    
    # Create SB3 DQN and train briefly
    print("\n1. Creating and training SB3 DQN...")
    try:
        # Create SB3 model
        sb3_model = DQN(
            'CnnPolicy', 
            env, 
            verbose=1,
            learning_rate=0.0001,
            buffer_size=10000,  # Smaller buffer for this demo
            learning_starts=1000,
            target_update_interval=1000,
            train_freq=4,
            gradient_steps=1,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.02,
            device='cpu'  # Force CPU to avoid CUDA issues
        )
        
        print("SUCCESS: SB3 DQN created")
        
        # Train for a bit (you can skip this and load a pre-trained model)
        print("Training SB3 model briefly...")
        sb3_model.learn(total_timesteps=5000)
        print("SUCCESS: SB3 model trained")
        
        # Save the trained SB3 model
        sb3_model.save("fresh_sb3_pong_dqn")
        print("SUCCESS: SB3 model saved")
        
    except Exception as e:
        print(f"FAILED: SB3 training failed: {e}")
        return None, None
    
    # Create Sequential network
    print("\n2. Creating Sequential network...")
    sequential_model = ProperSequentialDQN()
    print("SUCCESS: Sequential network created")
    
    # Transfer weights from SB3 to Sequential
    print("\n3. Transferring weights...")
    try:
        sb3_state_dict = sb3_model.q_net.state_dict()
        
        # Weight mapping from SB3 to Sequential
        weight_mapping = {
            # Convolutional layers (from features_extractor.cnn)
            'features_extractor.cnn.0.weight': 'network.conv1.weight',
            'features_extractor.cnn.0.bias': 'network.conv1.bias',
            'features_extractor.cnn.2.weight': 'network.conv2.weight', 
            'features_extractor.cnn.2.bias': 'network.conv2.bias',
            'features_extractor.cnn.4.weight': 'network.conv3.weight',
            'features_extractor.cnn.4.bias': 'network.conv3.bias',
            
            # Linear layers (from features_extractor.linear and q_net)
            'features_extractor.linear.0.weight': 'network.fc1.weight',
            'features_extractor.linear.0.bias': 'network.fc1.bias',
            'q_net.0.weight': 'network.fc2.weight',
            'q_net.0.bias': 'network.fc2.bias',
        }
        
        # Convert weights
        sequential_state_dict = {}
        for sb3_key, seq_key in weight_mapping.items():
            if sb3_key in sb3_state_dict:
                sequential_state_dict[seq_key] = sb3_state_dict[sb3_key].clone()
                print(f"  Mapped {sb3_key} -> {seq_key}: {sb3_state_dict[sb3_key].shape}")
            else:
                print(f"  WARNING: {sb3_key} not found in SB3 state dict")
                
        # Load weights into Sequential model
        sequential_model.load_state_dict(sequential_state_dict)
        print("SUCCESS: Weights transferred")
        
        # Verify the transfer by comparing outputs
        print("\n4. Verifying weight transfer...")
        test_input = torch.randn(1, 4, 84, 84)
        
        # Get SB3 output
        sb3_model.q_net.eval()
        with torch.no_grad():
            sb3_output = sb3_model.q_net(test_input)
            
        # Get Sequential output
        sequential_model.eval()
        with torch.no_grad():
            seq_output = sequential_model(test_input)
            
        # Compare
        diff = torch.abs(sb3_output - seq_output).max().item()
        print(f"Max output difference: {diff}")
        
        if diff < 1e-6:
            print("SUCCESS: Perfect weight transfer!")
        else:
            print(f"WARNING: Output difference {diff} - may indicate architecture mismatch")
            
    except Exception as e:
        print(f"FAILED: Weight transfer failed: {e}")
        return None, None
    
    # Save the Sequential model
    print("\n5. Saving Sequential model...")
    try:
        torch.save({
            'model_state_dict': sequential_model.state_dict(),
            'model_architecture': 'ProperSequentialDQN',
            'input_channels': 4,
            'n_actions': 6,
            'training_info': 'Converted from trained SB3 DQN'
        }, 'proper_sequential_pong_dqn.pt')
        print("SUCCESS: Sequential model saved as 'proper_sequential_pong_dqn.pt'")
        
    except Exception as e:
        print(f"FAILED: Could not save model: {e}")
    
    return sb3_model, sequential_model

if __name__ == "__main__":
    sb3_model, sequential_model = create_sequential_from_sb3()
    
    if sb3_model and sequential_model:
        print("\n=== SUCCESS ===")
        print("Created proper Sequential DQN from trained SB3 model")
        print("Model saved as 'proper_sequential_pong_dqn.pt'")
        print("\nNext: Test this model on Pong to see if it performs better")
    else:
        print("\n=== FAILED ===")
        print("Could not create proper Sequential DQN")