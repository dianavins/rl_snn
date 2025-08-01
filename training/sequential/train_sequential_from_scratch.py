#!/usr/bin/env python3
"""Train a Sequential DQN from scratch to achieve +21 performance"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from conversion.sb3_to_sequential.sb3_to_sequential_converter import SequentialDQNNetwork
import ale_py

def train_sequential_dqn():
    """Train Sequential DQN using SB3 as a guide"""
    print("=== Training Sequential DQN from Scratch ===")
    
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
        print(f"FAILED: Environment creation: {e}")
        return None
        
    # Method 1: Use SB3 to train and then extract weights
    print("\n1. Training SB3 DQN model...")
    try:
        sb3_model = DQN(
            'CnnPolicy',
            env,
            verbose=1,
            learning_rate=1e-4,
            buffer_size=100000,
            learning_starts=50000,
            target_update_interval=10000,
            train_freq=4,
            gradient_steps=1,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.01,
            device='cpu'
        )
        
        print("Training SB3 model (this will take a while)...")
        print("NOTE: This is just a demo - you should train for longer in practice")
        
        # Train for minimal steps (you should increase this)
        sb3_model.learn(total_timesteps=100000)  # Increase this for real training
        
        # Save SB3 model
        sb3_model.save("trained_sb3_pong")
        print("SUCCESS: SB3 model trained and saved")
        
        # Test SB3 performance
        print("\nTesting SB3 performance...")
        test_env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=42)
        test_env = VecFrameStack(test_env, n_stack=4)
        
        obs = test_env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        while not done and step_count < 5000:
            action, _states = sb3_model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            episode_reward += reward[0]
            step_count += 1
            
        test_env.close()
        print(f"SB3 test episode reward: {episode_reward}")
        
        # Convert to Sequential
        print("\n2. Converting SB3 to Sequential...")
        sequential_model = SequentialDQNNetwork()
        
        # Extract weights from SB3
        sb3_state_dict = sb3_model.q_net.state_dict()
        
        # Weight mapping
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
                print(f"  Mapped {sb3_key} -> {seq_key}")
            else:
                print(f"  WARNING: {sb3_key} not found")
        
        sequential_model.load_state_dict(sequential_state_dict)
        print("SUCCESS: Weights transferred to Sequential model")
        
        # Verify conversion
        print("\n3. Verifying conversion...")
        test_input = torch.randn(1, 4, 84, 84)
        
        with torch.no_grad():
            sb3_output = sb3_model.q_net(test_input)
            seq_output = sequential_model(test_input)
            
        diff = torch.abs(sb3_output - seq_output).max().item()
        print(f"Max output difference: {diff}")
        
        if diff < 1e-5:
            print("SUCCESS: Perfect conversion!")
        else:
            print(f"WARNING: Conversion difference: {diff}")
        
        # Test Sequential performance
        print("\n4. Testing Sequential performance...")
        test_env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=42)
        test_env = VecFrameStack(test_env, n_stack=4)
        
        obs = test_env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        sequential_model.eval()
        while not done and step_count < 5000:
            obs_tensor = torch.FloatTensor(obs)
            with torch.no_grad():
                q_values = sequential_model(obs_tensor)
                action = q_values.argmax().item()
            
            obs, reward, done, info = test_env.step([action])
            episode_reward += reward[0]
            step_count += 1
            
        test_env.close()
        print(f"Sequential test episode reward: {episode_reward}")
        
        # Save Sequential model
        torch.save({
            'model_state_dict': sequential_model.state_dict(),
            'model_architecture': 'SequentialDQNNetwork',
            'training_source': 'SB3_transfer',
            'sb3_performance': episode_reward,
            'conversion_verified': diff < 1e-5
        }, 'trained_sequential_pong_dqn.pt')
        
        print("SUCCESS: Sequential model saved as 'trained_sequential_pong_dqn.pt'")
        
        return sequential_model
        
    except Exception as e:
        print(f"FAILED: Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    model = train_sequential_dqn()
    
    if model:
        print("\n=== SUCCESS ===")
        print("Sequential DQN trained and ready for testing!")
        print("Run 'python test_with_sb3_env.py' with the new model")
    else:
        print("\n=== FAILED ===")
        print("Could not train Sequential DQN")