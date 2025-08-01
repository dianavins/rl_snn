#!/usr/bin/env python3
"""Test that crisdco uses the exact same environment as SB3 training"""

import torch
import numpy as np
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import ale_py

def test_environment_match():
    """Test environment compatibility between v4 and v5"""
    print("=== Testing Environment Version Compatibility ===")
    
    # Register ALE
    try:
        ale_py.register_all()
        print("SUCCESS: ALE environments registered")
    except Exception as e:
        print(f"WARNING: ALE registration: {e}")
    
    # Test environments
    test_environments = [
        "PongNoFrameskip-v4",  # SB3 standard
        "Pong-v4",             # Alternative v4
        "ALE/Pong-v5"          # New format
    ]
    
    env_results = {}
    
    for env_name in test_environments:
        print(f"\n--- Testing {env_name} ---")
        try:
            # Create environment exactly like SB3/mahowald
            env = make_atari_env(env_name, n_envs=1, seed=0)
            env = VecFrameStack(env, n_stack=4)
            
            # Reset and get initial observation
            obs = env.reset()
            print(f"✓ Environment created: {env_name}")
            print(f"  Observation shape: {obs.shape}")
            print(f"  Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
            print(f"  Action space: {env.action_space}")
            
            # Take a few random actions to test
            episode_rewards = []
            for episode in range(3):
                episode_reward = 0
                done = False
                steps = 0
                obs = env.reset()
                
                while not done and steps < 1000:
                    action = env.action_space.sample()
                    obs, reward, done, info = env.step(action)
                    # Fix: reward is already a scalar in vectorized env
                    reward_scalar = reward if isinstance(reward, (int, float)) else reward[0]
                    episode_reward += reward_scalar
                    steps += 1
                
                episode_rewards.append(episode_reward)
            
            env_results[env_name] = {
                'success': True,
                'obs_shape': obs.shape,
                'obs_range': (obs.min(), obs.max()),
                'episode_rewards': episode_rewards,
                'avg_reward': np.mean(episode_rewards)
            }
            
            print(f"  Test episodes: {[int(r) for r in episode_rewards]}")
            print(f"  Average reward: {np.mean(episode_rewards):.2f}")
            
            env.close()
            
        except Exception as e:
            print(f"✗ Failed {env_name}: {e}")
            env_results[env_name] = {'success': False, 'error': str(e)}
    
    # Compare results
    print(f"\n=== Environment Comparison ===")
    
    successful_envs = {k: v for k, v in env_results.items() if v['success']}
    
    if len(successful_envs) < 2:
        print("Cannot compare - need at least 2 working environments")
        return env_results
    
    # Compare observation shapes and ranges
    reference_env = list(successful_envs.keys())[0]
    reference_data = successful_envs[reference_env]
    
    print(f"Using {reference_env} as reference:")
    print(f"  Shape: {reference_data['obs_shape']}")
    print(f"  Range: [{reference_data['obs_range'][0]:.3f}, {reference_data['obs_range'][1]:.3f}]")
    
    compatible_envs = []
    for env_name, data in successful_envs.items():
        if env_name == reference_env:
            compatible_envs.append(env_name)
            continue
            
        shape_match = data['obs_shape'] == reference_data['obs_shape']
        range_similar = (abs(data['obs_range'][0] - reference_data['obs_range'][0]) < 1.0 and
                        abs(data['obs_range'][1] - reference_data['obs_range'][1]) < 50.0)
        
        print(f"\n{env_name} vs {reference_env}:")
        print(f"  Shape match: {'✓' if shape_match else '✗'} ({data['obs_shape']} vs {reference_data['obs_shape']})")
        print(f"  Range similar: {'✓' if range_similar else '✗'} ({data['obs_range']} vs {reference_data['obs_range']})")
        
        if shape_match and range_similar:
            compatible_envs.append(env_name)
            print(f"  ✓ COMPATIBLE with SB3 training")
        else:
            print(f"  ✗ May have compatibility issues")
    
    # Recommendations
    print(f"\n=== Recommendations ===")
    
    if "PongNoFrameskip-v4" in compatible_envs:
        print("✓ PERFECT: PongNoFrameskip-v4 works - use this (matches SB3 exactly)")
        recommended = "PongNoFrameskip-v4"
    elif "Pong-v4" in compatible_envs:
        print("✓ GOOD: Pong-v4 works - should be compatible with SB3")
        recommended = "Pong-v4"
    elif "ALE/Pong-v5" in compatible_envs:
        print("⚠ CAUTION: Only Pong-v5 works - may have slight differences from SB3 v4")
        recommended = "ALE/Pong-v5"
    else:
        print("✗ PROBLEM: No compatible environments found")
        recommended = None
    
    if recommended:
        print(f"\nRECOMMENDED: Use '{recommended}' for HiAER-Spike conversion")
        
        # Update the fix script recommendation
        print(f"\nTo ensure compatibility, modify fix_HiAER_Spike_roms.py:")
        print(f"  Change env_names_to_try to start with '{recommended}'")
    
    return env_results, recommended

if __name__ == "__main__":
    results, recommended = test_environment_match()
    
    print(f"\n=== SUMMARY ===")
    successful = [k for k, v in results.items() if v.get('success', False)]
    print(f"Working environments: {successful}")
    print(f"Recommended for SB3 compatibility: {recommended}")
    
    if recommended:
        print(f"\n✓ crisdco can use {recommended} for HiAER-Spike conversion")
    else:
        print(f"\n✗ Environment compatibility issues detected")