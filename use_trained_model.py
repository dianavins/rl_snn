#!/usr/bin/env python3
"""Use the trained SB3 model to create a proper Sequential DQN"""

import torch
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from conversion.sb3_to_sequential.sb3_to_sequential_converter import SequentialDQNNetwork
import ale_py

def use_trained_model():
    """Load trained SB3 model and convert to Sequential"""
    print("=== Using Trained SB3 Model ===")
    
    # Register ALE
    try:
        ale_py.register_all()
    except AttributeError:
        pass
    
    # Create environment for model loading
    env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=0)  
    env = VecFrameStack(env, n_stack=4)
    print("SUCCESS: Environment created")
    
    # Load trained SB3 model
    print("\n1. Loading trained SB3 model...")
    try:
        # Try the main trained model
        trained_sb3 = DQN.load("PongNoFrameskip-v4", env=env, device='cpu')
        print("SUCCESS: Trained SB3 model loaded from PongNoFrameskip-v4.zip")
        
    except Exception as e:
        print(f"Failed to load PongNoFrameskip-v4: {e}")
        try:
            # Try the pruned model
            trained_sb3 = DQN.load("pruned_pong_model", env=env, device='cpu')
            print("SUCCESS: Trained SB3 model loaded from pruned_pong_model.zip")
        except Exception as e2:
            print(f"Failed to load pruned_pong_model: {e2}")
            print("FAILED: Could not load any trained model")
            return None, None
    
    # Test the trained SB3 model first
    print("\n2. Testing trained SB3 model performance...")
    try:
        test_env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=42)
        test_env = VecFrameStack(test_env, n_stack=4)
        
        obs = test_env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        while not done and step_count < 5000:
            action, _states = trained_sb3.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            episode_reward += reward[0]
            step_count += 1
        
        test_env.close()
        print(f"Trained SB3 episode reward: {episode_reward}")
        
        if episode_reward >= 15:
            print("EXCELLENT: Trained SB3 model performs well!")
        else:
            print("WARNING: Trained SB3 model may not be fully trained")
            
    except Exception as e:
        print(f"SB3 testing failed: {e}")
    
    # Create Sequential model and copy trained weights
    print("\n3. Creating Sequential model with trained weights...")
    sequential_model = SequentialDQNNetwork()
    
    # Get trained weights
    trained_state_dict = trained_sb3.q_net.state_dict()
    
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
        if sb3_key in trained_state_dict:
            sequential_state_dict[seq_key] = trained_state_dict[sb3_key].clone()
            print(f"  Copied trained weight: {sb3_key} -> {seq_key}")
        else:
            print(f"  WARNING: Missing {sb3_key}")
    
    sequential_model.load_state_dict(sequential_state_dict)
    sequential_model.eval()
    print("SUCCESS: Trained weights loaded into Sequential model")
    
    # Verify the conversion
    print("\n4. Verifying trained weight conversion...")
    test_input = torch.randn(1, 4, 84, 84)
    
    with torch.no_grad():
        sb3_output = trained_sb3.q_net(test_input)
        seq_output = sequential_model(test_input)
        
    diff = torch.abs(sb3_output - seq_output).max().item()
    print(f"Output difference: {diff}")
    
    if diff < 1e-5:
        print("SUCCESS: Perfect conversion!")
    else:
        print(f"WARNING: Small difference: {diff}")
    
    # Test Sequential model performance
    print("\n5. Testing Sequential model with trained weights...")
    episode_rewards = []
    
    for episode in range(5):
        print(f"\nEpisode {episode + 1}:")
        
        test_env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=episode + 200)
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
            
            if step_count % 1000 == 0:
                print(f"    Step {step_count}: Reward = {episode_reward}")
        
        test_env.close()
        episode_rewards.append(episode_reward)
        print(f"  Episode {episode + 1} reward: {episode_reward}")
    
    # Save the trained Sequential model
    torch.save({
        'model_state_dict': sequential_model.state_dict(),
        'model_architecture': 'SequentialDQNNetwork',
        'source': 'trained_sb3_model',
        'episode_rewards': episode_rewards,
        'average_reward': np.mean(episode_rewards),
        'conversion_verified': diff < 1e-5
    }, 'trained_sequential_pong_dqn.pt')
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Episode rewards: {episode_rewards}")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    
    if np.mean(episode_rewards) >= 21:
        print("EXCELLENT: Sequential model achieves +21 performance!")
        print("SUCCESS: Ready for SNN conversion!")
    elif np.mean(episode_rewards) >= 15:
        print("VERY GOOD: High positive performance")
    elif np.mean(episode_rewards) >= 0:
        print("GOOD: Positive performance")
    else:
        print("ISSUE: Still negative performance - may need different model")
    
    print("Model saved as 'trained_sequential_pong_dqn.pt'")
    
    return sequential_model, episode_rewards

if __name__ == "__main__":
    model, rewards = use_trained_model()
    
    if model and rewards:
        avg_reward = np.mean(rewards)
        print(f"\nCONCLUSION: Trained Sequential DQN achieved {avg_reward:.2f} average reward")
        
        if avg_reward >= 21:
            print("SUCCESS: Model ready for SNN conversion!")
        elif avg_reward >= 15:
            print("VERY GOOD: Model performs well!")
        else:
            print("NEEDS WORK: May need better training")
    else:
        print("\nFAILED: Could not create trained Sequential model")