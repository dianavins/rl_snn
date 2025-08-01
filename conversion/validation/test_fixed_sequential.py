#!/usr/bin/env python3
"""Test the fixed Sequential model with proper input preprocessing"""

import torch
import numpy as np
from stable_baselines3.common.env_util import make_atari_env  
from stable_baselines3.common.vec_env import VecFrameStack
from conversion.sb3_to_sequential.sb3_to_sequential_converter import SequentialDQNNetwork
import ale_py

def test_fixed_sequential():
    """Test the fixed Sequential model with proper preprocessing"""
    print("=== Testing Fixed Sequential Model ===")
    
    # Register ALE
    try:
        ale_py.register_all()
    except AttributeError:
        pass
    
    # Load the fixed model
    print("1. Loading fixed Sequential model...")
    try:
        checkpoint = torch.load('fixed_sequential_pong_dqn.pt', map_location='cpu', weights_only=False)
        model = SequentialDQNNetwork()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("SUCCESS: Fixed Sequential model loaded")
        print(f"Model info: {checkpoint.get('architecture_verified', 'Unknown')}")
        
    except Exception as e:
        print(f"FAILED: Could not load model: {e}")
        return None
    
    # Create environment
    print("\n2. Creating test environment...")
    try:
        env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=42)
        env = VecFrameStack(env, n_stack=4)
        print("SUCCESS: Environment created")
        
    except Exception as e:
        print(f"FAILED: Environment creation: {e}")
        return None
    
    # Run test episodes
    print("\n3. Running test episodes...")
    episode_rewards = []
    
    for episode in range(5):
        print(f"\nEpisode {episode + 1}:")
        
        obs = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        print(f"  Initial obs shape: {obs.shape}")
        
        while not done and step_count < 5000:
            # CRITICAL: Fix input shape - obs is [1, 84, 84, 4] but model expects [1, 4, 84, 84]
            if len(obs.shape) == 4 and obs.shape[-1] == 4:
                # obs is [batch, height, width, channels] -> need [batch, channels, height, width]
                obs_tensor = torch.FloatTensor(obs).permute(0, 3, 1, 2)
            else:
                # obs is already [batch, channels, height, width]
                obs_tensor = torch.FloatTensor(obs)
            
            # Normalize to [0, 1] range (important!)
            obs_tensor = obs_tensor / 255.0
            
            with torch.no_grad():
                q_values = model(obs_tensor)
                action = q_values.argmax().item()
            
            obs, reward, done, info = env.step([action])
            episode_reward += reward[0]
            step_count += 1
            
            if step_count % 1000 == 0:
                print(f"    Step {step_count}: Reward = {episode_reward}")
        
        print(f"  Final reward: {episode_reward} (steps: {step_count})")
        episode_rewards.append(episode_reward)
    
    env.close()
    
    # Results
    print("\n=== RESULTS ===")
    print(f"Episode rewards: {episode_rewards}")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Best reward: {max(episode_rewards)}")
    print(f"Worst reward: {min(episode_rewards)}")
    
    if np.mean(episode_rewards) >= 21:
        print("EXCELLENT: Model achieves +21 average reward!")
    elif np.mean(episode_rewards) >= 15:
        print("VERY GOOD: Model achieves high positive reward")
    elif np.mean(episode_rewards) >= 5:
        print("GOOD: Model shows positive performance")
    elif np.mean(episode_rewards) >= 0:
        print("OKAY: Model shows non-negative performance")
    else:
        print("POOR: Model needs improvement")
        
    return episode_rewards

if __name__ == "__main__":
    rewards = test_fixed_sequential()
    
    if rewards:
        avg_reward = np.mean(rewards)
        print(f"\nFINAL RESULT: Fixed Sequential DQN achieved {avg_reward:.2f} average reward")
        
        if avg_reward >= 21:
            print("SUCCESS: Ready for SNN conversion!")
        else:
            print("NEEDS WORK: Requires further training or fixing")
    else:
        print("\nTEST FAILED: Could not evaluate model")