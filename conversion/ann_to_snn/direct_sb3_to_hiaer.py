#!/usr/bin/env python3
"""
Direct SB3 to HiAER Spike conversion pipeline
Bypasses the sequential intermediate step and converts directly from SB3 DQN to HiAER format
"""

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from collections import OrderedDict, defaultdict

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

# Import HiAER API components
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    from hardware.hs_api.api import CRI_network
    from hardware.hs_api.neuron_models import LIF_neuron, ANN_neuron
    from hardware.hs_api.converter import Quantize_Network
    print("SUCCESS: HiAER API imported")
except ImportError as e:
    print(f"ERROR: HiAER API not available: {e}")
    print("Project root:", project_root)
    print("Current working directory:", os.getcwd())
    exit(1)

class DirectSB3ToHiAERConverter:
    """
    Direct converter from SB3 DQN models to HiAER Spike format
    Extracts convolutional and linear layers and creates appropriate HiAER data structures
    """
    
    def __init__(self, alpha=4, v_threshold=2**19):
        self.alpha = alpha
        self.v_threshold = v_threshold
        self.neuron_id_counter = 0
        self.axon_dict = {}
        self.connections = {}
        self.layer_outputs = {}
        
    def extract_sb3_layers(self, sb3_model_path):
        """Extract layers from SB3 DQN model"""
        print(f"Loading SB3 model from: {sb3_model_path}")
        
        # Create dummy environment for loading
        env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=0)
        env = VecFrameStack(env, n_stack=4)
        
        # Load the trained model
        model = DQN.load(sb3_model_path, env=env)
        
        # Extract the Q-network
        q_net = model.policy.q_net
        
        # Extract feature extractor (CNN part)
        features_extractor = q_net.features_extractor
        
        # Extract the final Q-value network (fully connected part)  
        q_value_net = q_net.q_net
        
        print("SB3 Model Architecture:")
        print("Feature Extractor:", features_extractor)
        print("Q-Value Network:", q_value_net)
        
        # Build sequential network from extracted components
        layers = []
        
        # Add CNN layers from features_extractor
        if hasattr(features_extractor, 'cnn'):
            for layer in features_extractor.cnn:
                if isinstance(layer, (nn.Conv2d, nn.ReLU, nn.Flatten)):
                    layers.append(layer)
        
        # Add linear layers  
        if hasattr(features_extractor, 'linear'):
            for layer in features_extractor.linear:
                if isinstance(layer, (nn.Linear, nn.ReLU)):
                    layers.append(layer)
                    
        # Add final Q-value layers
        for layer in q_value_net:
            if isinstance(layer, (nn.Linear, nn.ReLU)):
                layers.append(layer)
        
        # Create sequential model
        sequential_model = nn.Sequential(*layers)
        sequential_model.eval()
        
        print(f"Extracted Sequential Model:")
        for i, layer in enumerate(sequential_model):
            print(f"  {i}: {layer}")
            
        env.close()
        return sequential_model
    
    def create_hiaer_neurons(self, layer, layer_idx, input_shape, output_shape):
        """Create HiAER neurons for a layer"""
        neurons = {}
        
        if isinstance(layer, nn.Conv2d):
            # For conv layers, create neurons for each output feature map position
            out_channels, out_h, out_w = output_shape
            for c in range(out_channels):
                for h in range(out_h):
                    for w in range(out_w):
                        neuron_id = f"L{layer_idx}_C{c}_H{h}_W{w}"
                        # Use LIF neurons for conv layers (spike-based processing)
                        neurons[neuron_id] = LIF_neuron(
                            leak_v=0,
                            threshold_v=self.v_threshold, 
                            reset_v=0
                        )
                        
        elif isinstance(layer, nn.Linear):
            # For linear layers, create neurons for each output unit
            out_features = output_shape[0] if isinstance(output_shape, tuple) else output_shape
            for i in range(out_features):
                neuron_id = f"L{layer_idx}_N{i}"
                # Use ANN neurons for final layers if it's the output layer
                if layer_idx == len(self.layers) - 1:
                    neurons[neuron_id] = ANN_neuron()
                else:
                    neurons[neuron_id] = LIF_neuron(
                        leak_v=0,
                        threshold_v=self.v_threshold,
                        reset_v=0
                    )
        
        return neurons
    
    def create_hiaer_connections(self, layer, layer_idx, input_shape, output_shape, neurons):
        """Create HiAER connections for a layer"""
        connections = {}
        
        if isinstance(layer, nn.Conv2d):
            # Extract convolution parameters
            weight = layer.weight.data
            bias = layer.bias.data if layer.bias is not None else None
            kernel_size = layer.kernel_size
            stride = layer.stride
            padding = layer.padding
            
            out_channels, in_channels, kh, kw = weight.shape
            input_h, input_w = input_shape[-2:]
            
            # Calculate output dimensions
            out_h = (input_h + 2 * padding[0] - kh) // stride[0] + 1
            out_w = (input_w + 2 * padding[1] - kw) // stride[1] + 1
            
            # Create connections for each output neuron
            for out_c in range(out_channels):
                for out_h_idx in range(out_h):
                    for out_w_idx in range(out_w):
                        neuron_id = f"L{layer_idx}_C{out_c}_H{out_h_idx}_W{out_w_idx}"
                        
                        # Find input connections for this output neuron
                        input_connections = []
                        
                        for in_c in range(in_channels):
                            for kh_idx in range(kh):
                                for kw_idx in range(kw):
                                    # Calculate input position
                                    in_h_idx = out_h_idx * stride[0] - padding[0] + kh_idx
                                    in_w_idx = out_w_idx * stride[1] - padding[1] + kw_idx
                                    
                                    # Check bounds
                                    if 0 <= in_h_idx < input_h and 0 <= in_w_idx < input_w:
                                        # Input axon identifier
                                        if layer_idx == 0:
                                            axon_id = f"input_C{in_c}_H{in_h_idx}_W{in_w_idx}"
                                        else:
                                            axon_id = f"L{layer_idx-1}_C{in_c}_H{in_h_idx}_W{in_w_idx}"
                                        
                                        # Get weight value and quantize it
                                        weight_val = weight[out_c, in_c, kh_idx, kw_idx].item()
                                        quantized_weight = int(weight_val * self.alpha)
                                        
                                        input_connections.append((axon_id, quantized_weight))
                        
                        # Add bias if present
                        if bias is not None:
                            bias_val = bias[out_c].item() 
                            quantized_bias = int(bias_val * self.alpha)
                            bias_axon = f"bias_L{layer_idx}_C{out_c}"
                            input_connections.append((bias_axon, quantized_bias))
                            
                            # Create bias axon
                            if bias_axon not in self.axon_dict:
                                self.axon_dict[bias_axon] = [(neuron_id, quantized_bias)]
                            else:
                                self.axon_dict[bias_axon].append((neuron_id, quantized_bias))
                        
                        # Store connections for this neuron
                        connections[neuron_id] = (input_connections, neurons[neuron_id])
                        
                        # Update axon dictionary
                        for axon_id, weight_val in input_connections:
                            if axon_id not in self.axon_dict:
                                self.axon_dict[axon_id] = []
                            self.axon_dict[axon_id].append((neuron_id, weight_val))
                            
        elif isinstance(layer, nn.Linear):
            # Extract linear layer parameters
            weight = layer.weight.data  # [out_features, in_features]
            bias = layer.bias.data if layer.bias is not None else None
            
            out_features, in_features = weight.shape
            
            # Create connections for each output neuron
            for out_idx in range(out_features):
                neuron_id = f"L{layer_idx}_N{out_idx}"
                
                input_connections = []
                
                # Connect to all input features
                for in_idx in range(in_features):
                    # Input axon identifier
                    if layer_idx == 0:
                        axon_id = f"input_N{in_idx}"
                    else:
                        axon_id = f"L{layer_idx-1}_N{in_idx}"
                    
                    # Get weight and quantize
                    weight_val = weight[out_idx, in_idx].item()
                    quantized_weight = int(weight_val * self.alpha)
                    
                    input_connections.append((axon_id, quantized_weight))
                
                # Add bias if present
                if bias is not None:
                    bias_val = bias[out_idx].item()
                    quantized_bias = int(bias_val * self.alpha)
                    bias_axon = f"bias_L{layer_idx}_N{out_idx}"
                    input_connections.append((bias_axon, quantized_bias))
                    
                    # Create bias axon
                    if bias_axon not in self.axon_dict:
                        self.axon_dict[bias_axon] = [(neuron_id, quantized_bias)]
                    else:
                        self.axon_dict[bias_axon].append((neuron_id, quantized_bias))
                
                # Store connections for this neuron
                connections[neuron_id] = (input_connections, neurons[neuron_id])
                
                # Update axon dictionary
                for axon_id, weight_val in input_connections:
                    if axon_id not in self.axon_dict:
                        self.axon_dict[axon_id] = []
                    self.axon_dict[axon_id].append((neuron_id, weight_val))
        
        return connections
    
    def calculate_layer_output_shape(self, layer, input_shape):
        """Calculate output shape for a layer given input shape"""
        if isinstance(layer, nn.Conv2d):
            # For conv2d: input_shape = (C, H, W)
            in_channels, in_h, in_w = input_shape
            out_channels = layer.out_channels
            kernel_size = layer.kernel_size
            stride = layer.stride  
            padding = layer.padding
            
            out_h = (in_h + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
            out_w = (in_w + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
            
            return (out_channels, out_h, out_w)
            
        elif isinstance(layer, nn.Linear):
            return (layer.out_features,)
            
        elif isinstance(layer, nn.Flatten):
            # Flatten everything except batch dimension
            return (np.prod(input_shape),)
            
        elif isinstance(layer, nn.ReLU):
            # ReLU doesn't change shape
            return input_shape
            
        else:
            raise ValueError(f"Unsupported layer type: {type(layer)}")
    
    def convert_model(self, sequential_model, input_shape=(4, 84, 84)):
        """Convert sequential model to HiAER format"""
        print("Converting model to HiAER format...")
        
        self.layers = [layer for layer in sequential_model if not isinstance(layer, nn.ReLU)]
        current_shape = input_shape
        
        # Create input axons
        input_size = np.prod(input_shape)
        for i in range(input_size):
            # Map flat index to multidimensional coordinates
            coords = np.unravel_index(i, input_shape)
            if len(coords) == 3:  # (C, H, W)
                axon_id = f"input_C{coords[0]}_H{coords[1]}_W{coords[2]}"
            else:
                axon_id = f"input_N{i}"
            self.axon_dict[axon_id] = []
        
        # Process each layer
        all_neurons = {}
        output_neurons = []
        
        for layer_idx, layer in enumerate(self.layers):
            if isinstance(layer, nn.ReLU):
                continue  # Skip activation layers
                
            print(f"Processing layer {layer_idx}: {layer}")
            print(f"  Input shape: {current_shape}")
            
            # Calculate output shape
            output_shape = self.calculate_layer_output_shape(layer, current_shape)
            print(f"  Output shape: {output_shape}")
            
            # Create neurons for this layer
            layer_neurons = self.create_hiaer_neurons(layer, layer_idx, current_shape, output_shape)
            all_neurons.update(layer_neurons)
            
            # Create connections for this layer  
            layer_connections = self.create_hiaer_connections(
                layer, layer_idx, current_shape, output_shape, layer_neurons
            )
            self.connections.update(layer_connections)
            
            # Update current shape for next layer
            current_shape = output_shape
            
            # If this is the last layer, mark neurons as outputs
            if layer_idx == len(self.layers) - 1:
                output_neurons.extend(list(layer_neurons.keys()))
        
        print(f"Conversion complete!")
        print(f"Total neurons: {len(all_neurons)}")
        print(f"Total axons: {len(self.axon_dict)}")
        print(f"Output neurons: {len(output_neurons)}")
        
        return self.axon_dict, self.connections, output_neurons

def main():
    """Main conversion function"""
    print("=== Direct SB3 to HiAER Conversion ===")
    
    # Initialize converter
    converter = DirectSB3ToHiAERConverter(alpha=4, v_threshold=2**19)
    
    # Extract SB3 model layers (use the ZIP file path)
    sb3_model_path = "PongNoFrameskip-v4.zip"
    try:
        sequential_model = converter.extract_sb3_layers(sb3_model_path)
    except Exception as e:
        print(f"ERROR: Could not load SB3 model: {e}")
        print("Available model files:")
        import os
        for root, dirs, files in os.walk("models"):
            for file in files:
                if file.endswith(('.zip', '.pt', '.pth')):
                    print(f"  {os.path.join(root, file)}")
        return
    
    # Convert to HiAER format
    try:
        axon_dict, connections, output_neurons = converter.convert_model(sequential_model)
    except Exception as e:
        print(f"ERROR during conversion: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create HiAER network configuration
    config = {
        'neuron_type': "LI&F",
        'global_neuron_params': {
            'v_thr': converter.v_threshold
        }
    }
    
    print("\nCreating CRI network...")
    try:
        # Create the CRI network
        hiaer_network = CRI_network(
            axons=axon_dict,
            connections=connections,
            config=config,
            target='simpleSim',
            outputs=output_neurons,
            simDump=False
        )
        
        print("SUCCESS: HiAER network created!")
        print(f"Network ready for simulation with {len(output_neurons)} outputs")
        
        # Test the network with dummy input
        print("\nTesting network with dummy input...")
        dummy_inputs = list(axon_dict.keys())[:10]  # Use first 10 axons as test
        result = hiaer_network.step(dummy_inputs, membranePotential=True)
        print(f"Test result: {len(result)} responses")
        
        return hiaer_network
        
    except Exception as e:
        print(f"ERROR creating CRI network: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    network = main()
    if network:
        print("Conversion successful! Network ready for use.")
    else:
        print("Conversion failed. Check errors above.")