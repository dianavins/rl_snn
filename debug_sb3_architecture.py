#!/usr/bin/env python3
"""Debug the original SB3 DQN architecture to understand the mismatch"""

import torch
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import ale_py
from sb3_to_sequential_converter import SequentialDQNNetwork, SB3ToSequentialConverter

def debug_sb3_vs_sequential():
    """Compare SB3 DQN and Sequential architectures"""
    print("=== SB3 DQN vs Sequential Architecture Debug ===")
    
    # Register ALE environments
    ale_py.register_all()
    
    # Create environment for testing
    try:
        env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=0)
        env = VecFrameStack(env, n_stack=4)
        print("✓ Environment created successfully")
    except:
        print("✗ Could not create environment, using dummy inputs")
        env = None
    
    # 1. Load original SNN model that performed well
    print("\n1. Loading original SNN model...")
    try:
        snn_checkpoint = torch.load('snn_pong_q_net_full.pt', map_location='cpu', weights_only=False)
        print(f"✓ SNN model loaded: {type(snn_checkpoint)}")
        
        if isinstance(snn_checkpoint, dict) and 'model' in snn_checkpoint:
            snn_model = snn_checkpoint['model']
        else:
            snn_model = snn_checkpoint
            
        print(f"SNN model type: {type(snn_model)}")
        print(f"SNN model keys: {list(snn_model.state_dict().keys()) if hasattr(snn_model, 'state_dict') else 'No state_dict'}")
        
    except Exception as e:
        print(f"✗ Could not load SNN model: {e}")
        snn_model = None
    
    # 2. Load Sequential model that performed poorly
    print("\n2. Loading Sequential model...")
    try:
        seq_checkpoint = torch.load('sequential_pong_dqn.pt', map_location='cpu', weights_only=False)
        seq_model = SequentialDQNNetwork()
        seq_model.load_state_dict(seq_checkpoint['model_state_dict'])
        print("✓ Sequential model loaded successfully")
        print(f"Sequential model architecture:\n{seq_model}")
        
    except Exception as e:
        print(f"✗ Could not load Sequential model: {e}")
        seq_model = None
        
    # 3. Try to create original SB3 DQN for comparison
    print("\n3. Creating fresh SB3 DQN...")
    try:
        if env is not None:
            sb3_model = DQN('CnnPolicy', env, verbose=0)
            print("✓ Fresh SB3 DQN created")
            print(f"SB3 policy architecture:\n{sb3_model.policy}")
            print(f"SB3 Q-network: {sb3_model.q_net}")
        else:
            sb3_model = None
            
    except Exception as e:
        print(f"✗ Could not create SB3 DQN: {e}")
        sb3_model = None
        
    # 4. Compare architectures if we have them
    if snn_model and seq_model:
        print("\n4. Comparing SNN vs Sequential architectures...")
        
        # Test with same input
        test_input = torch.randn(1, 4, 84, 84)
        
        # SNN forward pass
        snn_model.eval()
        with torch.no_grad():
            snn_output = snn_model(test_input)
            print(f"SNN output shape: {snn_output.shape}")
            print(f"SNN Q-values: {snn_output.numpy().flatten()}")
            
        # Sequential forward pass  
        seq_model.eval()
        with torch.no_grad():
            seq_output = seq_model(test_input)
            print(f"Sequential output shape: {seq_output.shape}")
            print(f"Sequential Q-values: {seq_output.numpy().flatten()}")
            
        # Compare outputs
        diff = torch.abs(snn_output - seq_output).max().item()
        print(f"Max output difference: {diff}")
        
        if diff < 1e-6:
            print("✓ Outputs match - architecture conversion is correct")
        else:
            print("✗ Outputs differ - there's an architecture mismatch!")
            
    # 5. Compare with SB3 if available
    if sb3_model and seq_model:
        print("\n5. Comparing SB3 vs Sequential...")
        
        test_input = torch.randn(1, 4, 84, 84)
        
        # SB3 forward pass
        with torch.no_grad():
            sb3_features = sb3_model.policy.features_extractor(test_input)
            sb3_q_values = sb3_model.policy.q_net(sb3_features)
            print(f"SB3 Q-values: {sb3_q_values.numpy().flatten()}")
            
        # Sequential forward pass
        with torch.no_grad():
            seq_output = seq_model(test_input)
            print(f"Sequential Q-values: {seq_output.numpy().flatten()}")
            
        # Compare
        diff = torch.abs(sb3_q_values - seq_output).max().item()
        print(f"SB3 vs Sequential max difference: {diff}")
        
    # 6. Architecture inspection
    print("\n6. Detailed architecture inspection...")
    
    if seq_model:
        print("Sequential model layers:")
        for name, layer in seq_model.named_modules():
            if len(list(layer.children())) == 0:  # Leaf modules only
                print(f"  {name}: {layer}")
                
    if sb3_model:
        print("\nSB3 model layers:")
        print("Features extractor:")
        for name, layer in sb3_model.policy.features_extractor.named_modules():
            if len(list(layer.children())) == 0:
                print(f"  {name}: {layer}")
        print("Q-network:")
        for name, layer in sb3_model.policy.q_net.named_modules():
            if len(list(layer.children())) == 0:
                print(f"  {name}: {layer}")
                
    # 7. Check for preprocessing differences
    print("\n7. Checking preprocessing...")
    if env:
        obs = env.reset()
        print(f"Environment observation shape: {obs.shape}")
        print(f"Environment observation range: [{obs.min():.3f}, {obs.max():.3f}]")
        
        # Test preprocessing
        test_obs = torch.FloatTensor(obs)
        print(f"Tensor observation range: [{test_obs.min():.3f}, {test_obs.max():.3f}]")
        
    return snn_model, seq_model, sb3_model

if __name__ == "__main__":
    snn_model, seq_model, sb3_model = debug_sb3_vs_sequential()
    
    print("\n=== SUMMARY ===")
    print(f"SNN model loaded: {'✓' if snn_model else '✗'}")
    print(f"Sequential model loaded: {'✓' if seq_model else '✗'}")
    print(f"SB3 model created: {'✓' if sb3_model else '✗'}")
    
    print("\nNext steps:")
    print("1. If models loaded, check output differences")
    print("2. If outputs differ, fix architecture mismatch")
    print("3. Test fixed model on Pong environment")