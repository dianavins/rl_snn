#!/usr/bin/env python3
"""
SB3 to HiAER Spike conversion demonstration
Shows how the original SB3 model can be converted to HiAER data structures
without requiring the full HiAER dependencies
"""

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from collections import OrderedDict, defaultdict
import json

# Register ALE environments
try:
    import ale_py
    if hasattr(ale_py, 'register_all'):
        ale_py.register_all()
    else:
        import gymnasium as gym
        gym.register_envs(ale_py)
    print("SUCCESS: ALE environments registered")
except Exception as e:
    print(f"WARNING: ALE registration issue: {e}")

class SB3ToHiAERDemo:
    """
    Demonstrates the conversion from SB3 DQN to HiAER Spike format
    Creates the necessary data structures without requiring full HiAER dependencies
    """
    
    def __init__(self, alpha=4, v_threshold=2**19):
        self.alpha = alpha
        self.v_threshold = v_threshold
        self.axon_dict = {}
        self.connections = {}
        self.config = {}
        
    def load_sb3_model(self, model_path):
        """Load and extract layers from SB3 DQN model"""
        print(f"Loading SB3 model from: {model_path}")
        
        # Create dummy environment for loading
        env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=0)
        env = VecFrameStack(env, n_stack=4)
        
        # Load the trained model
        model = DQN.load(model_path, env=env)
        
        # Extract the Q-network components
        q_net = model.policy.q_net
        
        print("Original SB3 Model Structure:")
        print("=" * 50)
        print("Q-Network:")
        print(q_net)
        print("\nFeature Extractor:")
        print(q_net.features_extractor)
        print("\nQ-Value Network:")
        print(q_net.q_net)
        
        # Extract layers for conversion
        layers = []
        layer_info = []
        
        # Get CNN layers
        if hasattr(q_net.features_extractor, 'cnn'):
            print("\nCNN Layers:")
            for i, layer in enumerate(q_net.features_extractor.cnn):
                print(f"  {i}: {layer}")
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    layers.append(layer)
                    layer_info.append(f"features_extractor.cnn.{i}")
        
        # Get linear layers from feature extractor
        if hasattr(q_net.features_extractor, 'linear'):
            print("\nFeature Extractor Linear Layers:")
            for i, layer in enumerate(q_net.features_extractor.linear):
                print(f"  {i}: {layer}")
                if isinstance(layer, (nn.Linear)):
                    layers.append(layer)
                    layer_info.append(f"features_extractor.linear.{i}")
        
        # Get Q-value network layers
        print("\nQ-Value Network Layers:")
        for i, layer in enumerate(q_net.q_net):
            print(f"  {i}: {layer}")
            if isinstance(layer, (nn.Linear)):
                layers.append(layer)
                layer_info.append(f"q_net.{i}")
        
        env.close()
        return layers, layer_info
    
    def analyze_layer_dimensions(self, layers):
        """Analyze the dimensions and structure of extracted layers"""
        print("\n" + "=" * 60)
        print("LAYER ANALYSIS FOR HIAER CONVERSION")
        print("=" * 60)
        
        input_shape = (4, 84, 84)  # Pong input: 4 frames, 84x84 pixels
        current_shape = input_shape
        
        for i, layer in enumerate(layers):
            print(f"\nLayer {i}: {type(layer).__name__}")
            print(f"  Input shape: {current_shape}")
            
            if isinstance(layer, nn.Conv2d):
                print(f"  Conv2d: {layer.in_channels} -> {layer.out_channels}")
                print(f"  Kernel: {layer.kernel_size}, Stride: {layer.stride}, Padding: {layer.padding}")
                print(f"  Weight shape: {layer.weight.shape}")
                print(f"  Bias: {'Yes' if layer.bias is not None else 'No'}")
                
                # Calculate output shape
                in_c, in_h, in_w = current_shape
                out_c = layer.out_channels
                k_h, k_w = layer.kernel_size
                s_h, s_w = layer.stride
                p_h, p_w = layer.padding
                
                out_h = (in_h + 2 * p_h - k_h) // s_h + 1
                out_w = (in_w + 2 * p_w - k_w) // s_w + 1
                current_shape = (out_c, out_h, out_w)
                
                print(f"  Output shape: {current_shape}")
                print(f"  Parameters: {layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)}")
                
            elif isinstance(layer, nn.Linear):
                print(f"  Linear: {layer.in_features} -> {layer.out_features}")
                print(f"  Weight shape: {layer.weight.shape}")
                print(f"  Bias: {'Yes' if layer.bias is not None else 'No'}")
                
                # For linear layers, we need to flatten if coming from conv
                if len(current_shape) > 1:
                    flattened_size = np.prod(current_shape)
                    print(f"  Flattening {current_shape} -> ({flattened_size},)")
                    current_shape = (flattened_size,)
                
                current_shape = (layer.out_features,)
                print(f"  Output shape: {current_shape}")
                print(f"  Parameters: {layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)}")
        
        return current_shape
    
    def create_hiaer_structure(self, layers):
        """Create HiAER-compatible data structures"""
        print("\n" + "=" * 60) 
        print("CREATING HIAER DATA STRUCTURES")
        print("=" * 60)
        
        # Initialize with input layer
        input_shape = (4, 84, 84)
        input_size = np.prod(input_shape)
        
        print(f"Input layer: {input_size} axons ({input_shape})")
        
        # Create input axons
        for i in range(input_size):
            self.axon_dict[f"input_{i}"] = []
        
        current_shape = input_shape
        neuron_counter = 0
        
        for layer_idx, layer in enumerate(layers):
            print(f"\nProcessing Layer {layer_idx}: {type(layer).__name__}")
            
            if isinstance(layer, nn.Conv2d):
                # Handle convolutional layer
                weight = layer.weight.data
                bias = layer.bias.data if layer.bias is not None else None
                
                in_c, in_h, in_w = current_shape
                out_c, in_c_w, k_h, k_w = weight.shape
                s_h, s_w = layer.stride
                p_h, p_w = layer.padding
                
                out_h = (in_h + 2 * p_h - k_h) // s_h + 1
                out_w = (in_w + 2 * p_w - k_w) // s_w + 1
                
                print(f"  Conv2d: {current_shape} -> ({out_c}, {out_h}, {out_w})")
                print(f"  Creating {out_c * out_h * out_w} neurons")
                
                # Create neurons and connections for each output position
                for out_ch in range(out_c):
                    for out_y in range(out_h):
                        for out_x in range(out_w):
                            neuron_id = f"L{layer_idx}_C{out_ch}_Y{out_y}_X{out_x}"
                            
                            # Create neuron connections
                            connections = []
                            
                            # Connect to input region
                            for in_ch in range(in_c_w):  # Use weight's input channels
                                for k_y in range(k_h):
                                    for k_x in range(k_w):
                                        # Calculate input position
                                        in_y = out_y * s_h - p_h + k_y
                                        in_x = out_x * s_w - p_w + k_x
                                        
                                        # Check bounds
                                        if 0 <= in_y < in_h and 0 <= in_x < in_w:
                                            if layer_idx == 0:
                                                # First layer connects to input
                                                input_idx = in_ch * in_h * in_w + in_y * in_w + in_x
                                                axon_id = f"input_{input_idx}"
                                            else:
                                                # Later layers connect to previous layer neurons
                                                axon_id = f"L{layer_idx-1}_C{in_ch}_Y{in_y}_X{in_x}"
                                            
                                            # Get weight and quantize
                                            w = weight[out_ch, in_ch, k_y, k_x].item()
                                            w_quantized = int(w * self.alpha)
                                            
                                            connections.append((axon_id, w_quantized))
                                            
                                            # Add to axon dictionary
                                            if axon_id not in self.axon_dict:
                                                self.axon_dict[axon_id] = []
                                            self.axon_dict[axon_id].append((neuron_id, w_quantized))
                            
                            # Add bias if present
                            if bias is not None:
                                bias_val = int(bias[out_ch].item() * self.alpha)
                                bias_axon = f"bias_L{layer_idx}_C{out_ch}"
                                connections.append((bias_axon, bias_val))
                                
                                if bias_axon not in self.axon_dict:
                                    self.axon_dict[bias_axon] = []
                                self.axon_dict[bias_axon].append((neuron_id, bias_val))
                            
                            # Store neuron with its connections
                            self.connections[neuron_id] = (connections, "LIF_neuron")
                            neuron_counter += 1
                
                current_shape = (out_c, out_h, out_w)
                
            elif isinstance(layer, nn.Linear):
                # Handle linear layer
                weight = layer.weight.data
                bias = layer.bias.data if layer.bias is not None else None
                
                # Flatten input if needed
                if len(current_shape) > 1:
                    input_size = np.prod(current_shape)
                    print(f"  Flattening: {current_shape} -> {input_size}")
                else:
                    input_size = current_shape[0]
                
                output_size = layer.out_features
                print(f"  Linear: {input_size} -> {output_size}")
                print(f"  Creating {output_size} neurons")
                
                # Create neurons for each output
                for out_idx in range(output_size):
                    neuron_id = f"L{layer_idx}_N{out_idx}"
                    
                    connections = []
                    
                    # Connect to all inputs
                    for in_idx in range(input_size):
                        if layer_idx == 0:
                            axon_id = f"input_{in_idx}"
                        else:
                            # Previous layer was conv or linear
                            if len(current_shape) > 1:
                                # Previous was conv - need to map flat index to conv coordinates
                                prev_c, prev_h, prev_w = current_shape
                                c = in_idx // (prev_h * prev_w)
                                remaining = in_idx % (prev_h * prev_w)
                                y = remaining // prev_w
                                x = remaining % prev_w
                                axon_id = f"L{layer_idx-1}_C{c}_Y{y}_X{x}"
                            else:
                                # Previous was linear
                                axon_id = f"L{layer_idx-1}_N{in_idx}"
                        
                        # Get weight and quantize
                        w = weight[out_idx, in_idx].item()
                        w_quantized = int(w * self.alpha)
                        
                        connections.append((axon_id, w_quantized))
                        
                        # Add to axon dictionary
                        if axon_id not in self.axon_dict:
                            self.axon_dict[axon_id] = []
                        self.axon_dict[axon_id].append((neuron_id, w_quantized))
                    
                    # Add bias if present
                    if bias is not None:
                        bias_val = int(bias[out_idx].item() * self.alpha)
                        bias_axon = f"bias_L{layer_idx}_N{out_idx}"
                        connections.append((bias_axon, bias_val))
                        
                        if bias_axon not in self.axon_dict:
                            self.axon_dict[bias_axon] = []
                        self.axon_dict[bias_axon].append((neuron_id, bias_val))
                    
                    # Store neuron - use ANN for final layer, LIF for others
                    neuron_type = "ANN_neuron" if layer_idx == len(layers) - 1 else "LIF_neuron"
                    self.connections[neuron_id] = (connections, neuron_type)
                    neuron_counter += 1
                
                current_shape = (output_size,)
        
        print(f"\nConversion Summary:")
        print(f"  Total neurons: {neuron_counter}")
        print(f"  Total axons: {len(self.axon_dict)}")
        print(f"  Output layer size: {current_shape[0]}")
        
        # Create config
        self.config = {
            'neuron_type': "LI&F",
            'global_neuron_params': {
                'v_thr': self.v_threshold
            }
        }
        
        return neuron_counter
    
    def save_hiaer_format(self, filename):
        """Save the HiAER data structures to file"""
        hiaer_data = {
            'axon_dict': {k: v for k, v in self.axon_dict.items()},
            'connections': {k: {'connections': v[0], 'neuron_type': v[1]} for k, v in self.connections.items()},
            'config': self.config,
            'conversion_params': {
                'alpha': self.alpha,
                'v_threshold': self.v_threshold
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(hiaer_data, f, indent=2)
        
        print(f"HiAER data structures saved to: {filename}")
    
    def demonstrate_usage(self):
        """Demonstrate how the converted network would be used"""
        print("\n" + "=" * 60)
        print("HIAER NETWORK USAGE EXAMPLE")
        print("=" * 60)
        
        print("The converted network can be used with HiAER API as follows:")
        print("""
# Create CRI network (when HiAER dependencies are available)
from hs_api.api import CRI_network

hiaer_network = CRI_network(
    axons=axon_dict,
    connections=connections,
    config=config,
    target='simpleSim',
    outputs=output_neurons,
    simDump=False
)

# Run inference
input_spikes = ['input_0', 'input_1', ...]  # Active input axons
potentials, spikes = hiaer_network.step(input_spikes, membranePotential=True)
        """)
        
        # Show some example data structures
        print("Example axon connections (first 5):")
        for i, (axon_id, connections) in enumerate(list(self.axon_dict.items())[:5]):
            print(f"  {axon_id}: {len(connections)} connections")
            if connections:
                print(f"    -> {connections[0]} (and {len(connections)-1} more)")
        
        print(f"\nExample neurons (first 3):")
        for i, (neuron_id, (connections, neuron_type)) in enumerate(list(self.connections.items())[:3]):
            print(f"  {neuron_id} ({neuron_type}): {len(connections)} inputs")

def main():
    """Main demonstration function"""
    print("=" * 80)
    print("SB3 TO HIAER SPIKE CONVERSION DEMONSTRATION")
    print("=" * 80)
    
    # Initialize converter
    converter = SB3ToHiAERDemo(alpha=4, v_threshold=2**19)
    
    # Load SB3 model
    model_path = "PongNoFrameskip-v4.zip"
    try:
        layers, layer_info = converter.load_sb3_model(model_path)
        print(f"\nExtracted {len(layers)} layers for conversion:")
        for i, (layer, info) in enumerate(zip(layers, layer_info)):
            print(f"  {i}: {info} -> {type(layer).__name__}")
        
    except Exception as e:
        print(f"ERROR: Could not load SB3 model: {e}")
        print("\nTrying alternative model paths...")
        
        # Try other available models
        import os
        model_candidates = []
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith('.zip') and 'pong' in file.lower():
                    model_candidates.append(os.path.join(root, file))
        
        if model_candidates:
            print("Found potential models:")
            for candidate in model_candidates:
                print(f"  {candidate}")
            try:
                layers, layer_info = converter.load_sb3_model(model_candidates[0])
            except Exception as e2:
                print(f"Could not load alternative model: {e2}")
                return
        else:
            print("No suitable model files found")
            return
    
    # Analyze layer structure
    output_shape = converter.analyze_layer_dimensions(layers)
    
    # Create HiAER data structures
    num_neurons = converter.create_hiaer_structure(layers)
    
    # Save results
    converter.save_hiaer_format("sb3_to_hiaer_conversion.json")
    
    # Demonstrate usage
    converter.demonstrate_usage()
    
    print(f"\n" + "=" * 80)
    print("CONVERSION COMPLETE!")
    print(f"Successfully converted SB3 DQN to HiAER format with {num_neurons} neurons")
    print("The conversion demonstrates how to bypass the sequential intermediate step")
    print("and directly create HiAER-compatible data structures from SB3 models.")
    print("=" * 80)

if __name__ == "__main__":
    main()