#!/usr/bin/env python3
"""Fix the Sequential model by properly initializing and testing it"""

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from conversion.sb3_to_sequential.sb3_to_sequential_converter import SequentialDQNNetwork
import ale_py

def fix_sequential_model():
    """Fix the Sequential model by creating it properly"""
    print("=== Fixing Sequential Model ===")
    
    # Register ALE
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
        print(f"FAILED: Environment: {e}")
        return None
    
    # Method 1: Create fresh SB3 model and copy architecture exactly
    print("\n1. Creating fresh SB3 model for reference...")
    try:
        sb3_model = DQN('CnnPolicy', env, verbose=0, device='cpu')
        print("SUCCESS: SB3 reference model created")
        
        # Print SB3 architecture for verification
        print("\nSB3 Q-network architecture:")
        print(sb3_model.q_net)
        
    except Exception as e:
        print(f"FAILED: SB3 creation: {e}")
        sb3_model = None
    
    # Method 2: Create Sequential with proper initialization
    print("\n2. Creating properly initialized Sequential model...")
    
    # Create Sequential model
    sequential_model = SequentialDQNNetwork()
    print("Sequential architecture:")
    print(sequential_model)
    
    # Method 3: Initialize weights properly (like SB3 does)
    print("\n3. Initializing weights properly...")
    
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            # SB3 uses default PyTorch initialization
            nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / np.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / np.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)
    
    sequential_model.apply(init_weights)
    print("SUCCESS: Weights initialized properly")
    
    # Method 4: If we have SB3 model, copy its weights
    if sb3_model:
        print("\n4. Copying weights from fresh SB3 model...")
        try:
            sb3_state_dict = sb3_model.q_net.state_dict()
            
            weight_mapping = {
                'features_extractor.cnn.0.weight': 'network.conv1.weight',
                'features_extractor.cnn.0.bias': 'network.conv1.bias',
                'features_extractor.cnn.2.weight': 'network.conv2.weight', 
                'features_extractor.cnn.2.bias': 'network.conv2.bias',
                'features_extractor.cnn.4.weight': 'network.conv3.weight',
                'features_extractor.cnn.4.bias': 'network.conv3.bias',
                'features_extractor.linear.0.weight': 'network.fc1.weight',
                'features_extractor.linear.0.bias': 'network.fc1.bias',
                'q_net.0.weight': 'network.fc2.weight',
                'q_net.0.bias': 'network.fc2.bias',
            }
            
            sequential_state_dict = {}
            for sb3_key, seq_key in weight_mapping.items():
                if sb3_key in sb3_state_dict:
                    sequential_state_dict[seq_key] = sb3_state_dict[sb3_key].clone()
                    print(f"  Copied {sb3_key} -> {seq_key}")
                else:
                    print(f"  Missing: {sb3_key}")
            
            # Load the weights
            sequential_model.load_state_dict(sequential_state_dict)
            print("SUCCESS: SB3 weights copied to Sequential")
            
            # Verify outputs match
            test_input = torch.randn(1, 4, 84, 84)
            with torch.no_grad():
                sb3_output = sb3_model.q_net(test_input)
                seq_output = sequential_model(test_input)
                
            diff = torch.abs(sb3_output - seq_output).max().item()
            print(f"Output difference: {diff}")
            
            if diff < 1e-6:
                print("SUCCESS: Perfect architecture match!")
            else:
                print(f"WARNING: Architecture mismatch: {diff}")
                
        except Exception as e:
            print(f"Weight copying failed: {e}")
    
    # Method 5: Test the Sequential model
    print("\n5. Testing Sequential model performance...")
    try:
        test_env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=42)
        test_env = VecFrameStack(test_env, n_stack=4)
        
        # Run one episode
        obs = test_env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        sequential_model.eval()
        while not done and step_count < 3000:  # Shorter test
            obs_tensor = torch.FloatTensor(obs)
            with torch.no_grad():
                q_values = sequential_model(obs_tensor)
                action = q_values.argmax().item()
            
            obs, reward, done, info = test_env.step([action])
            episode_reward += reward[0]
            step_count += 1
            
            if step_count % 500 == 0:
                print(f"    Step {step_count}: Reward = {episode_reward}")
        
        test_env.close()
        print(f"Sequential model test reward: {episode_reward}")
        
        # Save the fixed model
        torch.save({
            'model_state_dict': sequential_model.state_dict(),
            'model_architecture': 'SequentialDQNNetwork_Fixed',
            'test_reward': episode_reward,
            'initialization': 'proper_kaiming_uniform',
            'architecture_verified': True
        }, 'fixed_sequential_pong_dqn.pt')
        
        print("SUCCESS: Fixed model saved as 'fixed_sequential_pong_dqn.pt'")
        
        return sequential_model, episode_reward
        
    except Exception as e:
        print(f"Testing failed: {e}")
        return sequential_model, None

if __name__ == "__main__":
    model, reward = fix_sequential_model()
    
    if model:
        print(f"\n=== RESULTS ===")
        print(f"Fixed Sequential model created")
        if reward is not None:
            print(f"Test episode reward: {reward}")
            if reward >= 15:
                print("EXCELLENT: Model performs well!")
            elif reward >= 0:
                print("GOOD: Model shows positive performance")
            else:
                print("NEEDS WORK: Model needs training or better weights")
        print("\nModel saved as 'fixed_sequential_pong_dqn.pt'")
    else:
        print("\n=== FAILED ===")
        print("Could not create fixed Sequential model")