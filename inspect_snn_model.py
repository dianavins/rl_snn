#!/usr/bin/env python3
"""Inspect the SNN model file to understand its structure"""

import torch
import numpy as np

def inspect_snn_model():
    """Inspect the SNN model that achieved +21 performance"""
    print("=== Inspecting SNN Model ===")
    
    try:
        # Try to load without spikingjelly first
        checkpoint = torch.load('snn_pong_q_net_full.pt', map_location='cpu', weights_only=False)
        print(f"Checkpoint type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            for key, value in checkpoint.items():
                print(f"\n{key}: {type(value)}")
                
                if hasattr(value, 'state_dict'):
                    print(f"  Has state_dict with keys: {list(value.state_dict().keys())[:10]}...")
                elif hasattr(value, 'keys'):
                    print(f"  Dict-like with keys: {list(value.keys())[:10]}...")
                elif torch.is_tensor(value):
                    print(f"  Tensor with shape: {value.shape}")
                    
        else:
            print(f"Direct model object")
            if hasattr(checkpoint, 'state_dict'):
                state_dict = checkpoint.state_dict()
                print(f"State dict keys: {list(state_dict.keys())}")
                
                # Look for weight patterns
                conv_weights = [k for k in state_dict.keys() if 'conv' in k.lower() and 'weight' in k]
                linear_weights = [k for k in state_dict.keys() if ('linear' in k.lower() or 'fc' in k.lower()) and 'weight' in k]
                
                print(f"\nConv weights: {conv_weights}")
                print(f"Linear weights: {linear_weights}")
                
                # Print shapes
                for key in list(state_dict.keys())[:15]:  # First 15 keys
                    tensor = state_dict[key]
                    print(f"  {key}: {tensor.shape}")
                    
    except Exception as e:
        print(f"Could not load SNN model: {e}")
        
    # Also inspect our current Sequential model
    print("\n=== Inspecting Current Sequential Model ===")
    try:
        seq_checkpoint = torch.load('sequential_pong_dqn.pt', map_location='cpu', weights_only=False)
        print(f"Sequential checkpoint keys: {list(seq_checkpoint.keys())}")
        
        if 'model_state_dict' in seq_checkpoint:
            state_dict = seq_checkpoint['model_state_dict']
            print(f"Sequential state dict keys: {list(state_dict.keys())}")
            
            # Print shapes
            for key in state_dict.keys():
                tensor = state_dict[key]
                print(f"  {key}: {tensor.shape}")
                
    except Exception as e:
        print(f"Could not load Sequential model: {e}")

if __name__ == "__main__":
    inspect_snn_model()