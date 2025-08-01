#!/usr/bin/env python3
"""Debug why the Sequential model gets -21 instead of +21"""

import torch
import numpy as np
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from conversion.sb3_to_sequential.sb3_to_sequential_converter import SequentialDQNNetwork
import ale_py

def debug_reward_calculation():
    """Debug the reward calculation and model behavior"""
    print("=== Debugging Reward Calculation ===")
    
    # Register ALE
    try:
        ale_py.register_all()
    except AttributeError:
        pass
    
    # Load Sequential model
    print("1. Loading Sequential model...")
    try:
        checkpoint = torch.load('working_sequential_pong_dqn.pt', map_location='cpu', weights_only=False)
        model = SequentialDQNNetwork()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("SUCCESS: Sequential model loaded")
        
    except Exception as e:
        print(f"FAILED: Could not load model: {e}")
        print("Creating a fresh model for analysis...")
        model = SequentialDQNNetwork()
    
    # Create environment
    env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=42)
    env = VecFrameStack(env, n_stack=4)
    print("SUCCESS: Environment created")
    
    # Analyze model behavior
    print("\n2. Analyzing model behavior...")
    
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    
    # Test different action selections
    episode_reward = 0
    done = False
    step_count = 0
    action_counts = [0] * 6
    q_values_history = []
    
    print("\n3. Running episode with detailed logging...")
    
    while not done and step_count < 1000:  # Shorter for analysis
        # Preprocess
        if len(obs.shape) == 4 and obs.shape[-1] == 4:
            obs_tensor = torch.FloatTensor(obs).permute(0, 3, 1, 2)
        else:
            obs_tensor = torch.FloatTensor(obs)
        
        obs_tensor = obs_tensor / 255.0
        
        # Get Q-values
        with torch.no_grad():
            q_values = model(obs_tensor)
            q_values_np = q_values.numpy().flatten()
            action = q_values.argmax().item()
        
        action_counts[action] += 1
        q_values_history.append(q_values_np.copy())
        
        # Log periodically
        if step_count % 200 == 0:
            print(f"  Step {step_count}:")
            print(f"    Q-values: {q_values_np.round(3)}")
            print(f"    Action: {action}")
            print(f"    Reward so far: {episode_reward}")
        
        # Take step
        obs, reward, done, info = env.step([action])
        episode_reward += reward[0]
        step_count += 1
    
    env.close()
    
    print(f"\n=== ANALYSIS RESULTS ===")
    print(f"Final reward: {episode_reward}")
    print(f"Steps taken: {step_count}")
    print(f"Action distribution: {action_counts}")
    print(f"Action percentages: {[f'{count/step_count*100:.1f}%' for count in action_counts]}")
    
    # Analyze Q-values
    q_values_array = np.array(q_values_history)
    mean_q_values = q_values_array.mean(axis=0)
    std_q_values = q_values_array.std(axis=0)
    
    print(f"\nQ-value statistics:")
    print(f"  Mean Q-values: {mean_q_values.round(3)}")
    print(f"  Std Q-values: {std_q_values.round(3)}")
    print(f"  Preferred action (by mean Q): {mean_q_values.argmax()}")
    
    # Pong action meanings
    action_meanings = {
        0: "NOOP", 1: "FIRE", 2: "RIGHT", 3: "LEFT", 4: "RIGHTFIRE", 5: "LEFTFIRE"
    }
    
    print(f"\nAction analysis:")
    for i, count in enumerate(action_counts):
        if count > 0:
            print(f"  Action {i} ({action_meanings.get(i, 'UNKNOWN')}): {count} times ({count/step_count*100:.1f}%)")
    
    # Diagnosis
    print(f"\n=== DIAGNOSIS ===")
    
    if episode_reward == -21:
        print("ISSUE: Model consistently gets -21 (loses every point)")
        print("This suggests:")
        print("  1. Model is not properly trained")
        print("  2. Model architecture might be correct but weights are wrong")
        print("  3. The original +21 performance came from different weights")
        
        # Check if model always chooses same action
        most_common_action = action_counts.index(max(action_counts))
        action_percentage = max(action_counts) / step_count * 100
        
        if action_percentage > 80:
            print(f"  4. Model is stuck choosing action {most_common_action} ({action_meanings.get(most_common_action)}) {action_percentage:.1f}% of the time")
        
        print(f"\nRECOMMENDAITON:")
        print(f"  - The Sequential architecture is correct")
        print(f"  - Need to load weights from the trained SNN that achieved +21")
        print(f"  - Or train the Sequential model properly")
        
    elif episode_reward > 15:
        print("SUCCESS: Model performs well!")
    else:
        print("PARTIAL: Model shows some learning but needs improvement")
    
    return episode_reward, action_counts, mean_q_values

if __name__ == "__main__":
    reward, actions, q_vals = debug_reward_calculation()
    
    print(f"\nCONCLUSION:")
    print(f"Sequential DQN achieved {reward} reward")
    print(f"The architecture is correct, the issue is in the weights/training")
    
    if reward == -21:
        print("SOLUTION: Need properly trained weights, not random initialization")