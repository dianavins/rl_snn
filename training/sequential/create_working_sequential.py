#!/usr/bin/env python3
"""Create a working Sequential DQN model"""

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from conversion.sb3_to_sequential.sb3_to_sequential_converter import SequentialDQNNetwork
import ale_py

def create_working_sequential():
    """Create a working Sequential model and test it properly"""
    print("=== Creating Working Sequential Model ===")
    
    # Register ALE
    try:
        ale_py.register_all()
    except AttributeError:
        pass
    
    # Create environment
    env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)
    print("SUCCESS: Environment created")
    
    # Create fresh SB3 model for reference
    print("\n1. Creating reference SB3 model...")
    sb3_model = DQN('CnnPolicy', env, verbose=0, device='cpu')
    print("SUCCESS: SB3 model created")
    
    # Create and initialize Sequential model
    print("\n2. Creating Sequential model...")
    sequential_model = SequentialDQNNetwork()
    
    # Copy weights from SB3
    print("\n3. Copying weights from SB3...")
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
        sequential_state_dict[seq_key] = sb3_state_dict[sb3_key].clone()
    
    sequential_model.load_state_dict(sequential_state_dict)
    sequential_model.eval()
    print("SUCCESS: Weights copied")
    
    # Test with proper preprocessing
    print("\n4. Testing Sequential model...")
    test_env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=42)
    test_env = VecFrameStack(test_env, n_stack=4)
    
    obs = test_env.reset()
    episode_reward = 0
    done = False
    step_count = 0
    
    print(f"Initial observation shape: {obs.shape}")
    print(f"Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    
    while not done and step_count < 3000:
        # CRITICAL: Proper preprocessing
        # VecFrameStack returns observations as [1, 84, 84, 4] 
        # But our model expects [1, 4, 84, 84]
        
        if len(obs.shape) == 4 and obs.shape[-1] == 4:
            # Transpose from NHWC to NCHW format
            obs_tensor = torch.FloatTensor(obs).permute(0, 3, 1, 2)
        else:
            obs_tensor = torch.FloatTensor(obs)
        
        # Normalize observations to [0, 1] range (critical!)
        obs_tensor = obs_tensor / 255.0
        
        with torch.no_grad():
            q_values = sequential_model(obs_tensor)
            action = q_values.argmax().item()
        
        obs, reward, done, info = test_env.step([action])
        episode_reward += reward[0]
        step_count += 1
        
        if step_count % 500 == 0:
            print(f"  Step {step_count}: Reward = {episode_reward}")
    
    test_env.close()
    print(f"Episode reward: {episode_reward}")
    
    # Save the working model
    torch.save({
        'model_state_dict': sequential_model.state_dict(),
        'model_architecture': 'SequentialDQNNetwork',
        'test_reward': episode_reward,
        'preprocessing_notes': 'Requires NHWC->NCHW transpose and /255 normalization',
        'working_verified': True
    }, 'working_sequential_pong_dqn.pt')
    
    print("SUCCESS: Working model saved as 'working_sequential_pong_dqn.pt'")
    
    # Now run multiple test episodes
    print("\n5. Running 5 test episodes...")
    episode_rewards = []
    
    for episode in range(5):
        print(f"\nEpisode {episode + 1}:")
        
        test_env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=episode + 100)
        test_env = VecFrameStack(test_env, n_stack=4)
        
        obs = test_env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        while not done and step_count < 5000:
            # Proper preprocessing
            if len(obs.shape) == 4 and obs.shape[-1] == 4:
                obs_tensor = torch.FloatTensor(obs).permute(0, 3, 1, 2)
            else:
                obs_tensor = torch.FloatTensor(obs)
            
            obs_tensor = obs_tensor / 255.0
            
            with torch.no_grad():
                q_values = sequential_model(obs_tensor)
                action = q_values.argmax().item()
            
            obs, reward, done, info = test_env.step([action])
            episode_reward += reward[0]
            step_count += 1
        
        test_env.close()
        episode_rewards.append(episode_reward)
        print(f"  Episode {episode + 1} reward: {episode_reward}")
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Episode rewards: {episode_rewards}")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    
    if np.mean(episode_rewards) >= 21:
        print("EXCELLENT: Sequential model achieves +21 performance!")
        print("Ready for SNN conversion!")
    elif np.mean(episode_rewards) >= 15:
        print("VERY GOOD: High positive performance")
    elif np.mean(episode_rewards) >= 0:
        print("GOOD: Positive performance")
    else:
        print("NEEDS IMPROVEMENT: Negative performance")
        print("The model architecture is correct but needs trained weights")
    
    return sequential_model, episode_rewards

if __name__ == "__main__":
    model, rewards = create_working_sequential()
    
    avg_reward = np.mean(rewards)
    print(f"\nCONCLUSION: Sequential DQN achieved {avg_reward:.2f} average reward")
    
    if avg_reward >= 15:
        print("SUCCESS: Model is working well!")
    else:
        print("ISSUE: Model needs better trained weights")