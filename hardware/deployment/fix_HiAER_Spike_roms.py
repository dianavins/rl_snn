#!/usr/bin/env python3
"""Fixed version of HiAER_Spike.py with proper ROM handling for crisdco"""

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv, VecFrameStack

# Import spikingjelly components
from spikingjelly.clock_driven import ann2snn, functional
from torch.utils.data import DataLoader, TensorDataset
import hardware.hs_api as hs_api
from hardware.hs_api.converter import *

from spikingjelly.clock_driven.neuron import IFNode as ClockDrivenIFNode                                                                           
from spikingjelly.activation_based.neuron import IFNode as ActivationBasedIFNode 

def setup_environment():
    """Setup Atari environment with proper ROM registration"""
    print("=== Setting up Atari environment ===")
    
    # Step 1: Register ALE environments
    try:
        import ale_py
        ale_py.register_all()
        print("SUCCESS: ALE environments registered")
    except Exception as e:
        print(f"WARNING: ALE registration issue: {e}")
    
    # Step 2: Try different environment names - prioritize v4 to match SB3 training
    env_names_to_try = [
        "PongNoFrameskip-v4",    # Primary: matches SB3 training exactly
        "Pong-v4",               # Backup: standard v4
        "PongDeterministic-v4",  # Backup: deterministic v4
        "ALE/Pong-v5",           # Last resort: newer version
    ]
    
    env = None
    successful_env_name = None
    
    for env_name in env_names_to_try:
        try:
            print(f"Trying environment: {env_name}")
            env = make_atari_env(env_name, n_envs=1, seed=0)
            env = VecFrameStack(env, n_stack=4)
            successful_env_name = env_name
            print(f"SUCCESS: Created environment {env_name}")
            break
        except Exception as e:
            print(f"FAILED {env_name}: {e}")
            continue
    
    if env is None:
        raise RuntimeError("Could not create any Pong environment. ROM installation may be required.")
    
    return env, successful_env_name

def main():
    """Main HiAER-Spike conversion with proper ROM handling"""
    print("=== HiAER-Spike Conversion with ROM Fix ===")
    
    # Setup environment first
    try:
        env, env_name = setup_environment()
        print(f"Using environment: {env_name}")
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        print("\nTroubleshooting steps:")
        print("1. Run: python -m autorom --accept-license")
        print("2. Or run: python setup_crisdco_environment.py")
        print("3. Check ROM installation with: ale-import-roms")
        return
    
    # Load ANN model
    print("\n=== Loading ANN Model ===")
    try:
        ann_model_path = "/home/dvins/rl_snn/PongNoFrameskip-v4.zip"  # Update path as needed
        print(f"Loading model from: {ann_model_path}")
        
        ann_model = DQN.load(ann_model_path, env=env)
        print("SUCCESS: ANN model loaded")
        
        # Extract Sequential layers
        ann_model = nn.Sequential(
            ann_model.policy.q_net.features_extractor.cnn[0],
            ann_model.policy.q_net.features_extractor.cnn[2], 
            ann_model.policy.q_net.features_extractor.cnn[4],
            ann_model.policy.q_net.features_extractor.linear[0],
            ann_model.policy.q_net.q_net[0],
        )
        print("SUCCESS: Sequential ANN extracted")
        print(f"ANN model: {ann_model}")
        
    except Exception as e:
        print(f"ERROR loading ANN model: {e}")
        print("Using fallback Sequential model...")
        
        # Fallback: use our trained Sequential model
        try:
            from conversion.sb3_to_sequential.sb3_to_sequential_converter import SequentialDQNNetwork
            checkpoint = torch.load('trained_sequential_pong_dqn.pt', map_location='cpu', weights_only=False)
            sequential_model = SequentialDQNNetwork()
            sequential_model.load_state_dict(checkpoint['model_state_dict'])
            
            # Convert to simple Sequential for SNN conversion
            ann_model = nn.Sequential(
                sequential_model.network.conv1,
                sequential_model.network.conv2,
                sequential_model.network.conv3,
                sequential_model.network.flatten,
                sequential_model.network.fc1,
                sequential_model.network.fc2
            )
            print("SUCCESS: Using trained Sequential model as fallback")
            
        except Exception as e2:
            print(f"CRITICAL ERROR: Could not load any model: {e2}")
            return
    
    # Continue with SNN conversion
    print("\n=== Converting to SNN ===")
    
    # Add IFNode layers after each layer except last
    snn_layers = []
    for i, layer in enumerate(ann_model):
        snn_layers.append(layer)
        if i < len(ann_model) - 1:  # Don't add IFNode after last layer
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                snn_layers.append(ClockDrivenIFNode())
    
    snn_model = nn.Sequential(*snn_layers)
    print("SUCCESS: SNN model created")
    print(f"SNN model: {snn_model}")
    
    # Test the models
    print("\n=== Testing Models ===")
    
    try:
        # Get test observation from environment
        obs = env.reset()
        print(f"Environment observation shape: {obs.shape}")
        
        # Preprocess for model input
        if len(obs.shape) == 4 and obs.shape[-1] == 4:
            obs_tensor = torch.FloatTensor(obs).permute(0, 3, 1, 2) / 255.0
        else:
            obs_tensor = torch.FloatTensor(obs) / 255.0
            
        print(f"Model input shape: {obs_tensor.shape}")
        
        # Test ANN
        with torch.no_grad():
            ann_output = ann_model(obs_tensor)
            print(f"ANN output: {ann_output}")
        
        # Test SNN (simple version without time steps for now)
        with torch.no_grad():
            functional.reset_net(snn_model)
            snn_output = snn_model(obs_tensor)
            print(f"SNN output: {snn_output}")
        
        print("SUCCESS: Both models working!")
        
    except Exception as e:
        print(f"ERROR in model testing: {e}")
        import traceback
        traceback.print_exc()
    
    # Save models
    print("\n=== Saving Models ===")
    
    try:
        torch.save(ann_model.state_dict(), 'hiear_ann_model.pt')
        torch.save(snn_model.state_dict(), 'hiear_snn_model.pt')
        torch.save({
            'ann_model': ann_model,
            'snn_model': snn_model,
            'environment': env_name
        }, 'hiear_complete_models.pt')
        
        print("SUCCESS: Models saved")
        print("Files created:")
        print("  - hiear_ann_model.pt (ANN weights)")
        print("  - hiear_snn_model.pt (SNN weights)")  
        print("  - hiear_complete_models.pt (Complete models)")
        
    except Exception as e:
        print(f"ERROR saving models: {e}")
    
    # Clean up
    env.close()
    print("\nHiAER-Spike conversion complete!")

if __name__ == "__main__":
    main()